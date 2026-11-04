"""Claude gateway (docs/rive-design.md section 6.16): the only way experiments reach the Claude API.

- The key is read from ~/.config/rive/claude.env (ANTHROPIC_API_KEY=...) or the environment; it is never logged,
  and a request that contains it is refused.
- Every request is checked before it is sent against a monthly budget (user, 2026-11-03: 40 USD a month) and the
  caller's session budget, priced at the worst case: every input token written to the 1-hour cache (2 characters
  per token) and every output token used.
- Every request is logged with its tokens and cost in <log_dir>/usage.jsonl.

Prices are Claude Opus 5.5's from the Claude API skill's model notes on 2026-11-03 (USD per million tokens) and must
be checked again before a formal experiment.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

KEY_FILE = Path("~/.config/rive/claude.env").expanduser()
PRICES = {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0,
                              "cache_write_1h": 8.0}}


class BudgetExceeded(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


def read_key(path=KEY_FILE):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def cost(model, usage):
    p = PRICES[model]
    created = usage.get("cache_creation_input_tokens") or 0
    five = usage.get("ephemeral_5m_input_tokens")
    hour = created - five if five is not None else created  # without the split, the dearer rate
    return (usage["input_tokens"] * p["input"] + (usage.get("cache_read_input_tokens") or 0) * p["cache_read"]
            + (five or 0) * p["cache_write_5m"] + hour * p["cache_write_1h"] + usage["output_tokens"] * p["output"]) / 1e6


class ClaudeGateway:
    def __init__(self, log_dir, monthly_usd=40.0, session_usd=None, model="claude-opus-5-5", client=None,
                 clock=lambda: datetime.now(timezone.utc)):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "usage.jsonl"
        self.monthly, self.session, self.model = monthly_usd, session_usd, model
        self.clock, self.lock, self._client = clock, threading.Lock(), client
        self.session_spent = 0.0

    def client(self):
        if self._client is None:
            import anthropic
            key = read_key()
            if not key:
                raise Refused(f"no ANTHROPIC_API_KEY in the environment or {KEY_FILE}")
            self._client = anthropic.Anthropic(api_key=key, max_retries=3, timeout=900)
        return self._client

    def spent_this_month(self, now):
        total = 0.0
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                e = json.loads(line)
                t = datetime.fromisoformat(e["time"])
                if (t.year, t.month) == (now.year, now.month):
                    total += e["cost_usd"]
        return total

    def worst_case(self, request_chars, max_tokens):
        p = PRICES[self.model]
        return (request_chars / 2 * p["cache_write_1h"] + max_tokens * p["output"]) / 1e6

    def remaining(self):
        """USD left in the session (or the month, if no session budget)."""
        month_left = self.monthly - self.spent_this_month(self.clock())
        return min(month_left, self.session - self.session_spent) if self.session else month_left

    def create(self, purpose, *, system, messages, max_tokens, **kwargs):
        """One streamed Messages request (the final message). kwargs go to messages.stream (tools, thinking,
        output_config, cache_control, ...)."""
        request = json.dumps([system, messages, kwargs], default=str)
        key = read_key()
        if key and key in request:
            raise Refused("the request contains the API key")
        now = self.clock()
        with self.lock:
            worst = self.worst_case(len(request), max_tokens)
            month = self.spent_this_month(now)
            if month + worst > self.monthly or (self.session and self.session_spent + worst > self.session):
                raise BudgetExceeded(f"spent {month:.2f} USD this month and {self.session_spent:.2f} in this session; "
                                     f"this request could cost {worst:.2f} (caps {self.monthly} and {self.session})")
            with self.client().messages.stream(model=self.model, system=system, messages=messages,
                                               max_tokens=max_tokens, **kwargs) as stream:
                message = stream.get_final_message()
            u = message.usage
            split = getattr(u, "cache_creation", None)
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                     "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                     "ephemeral_5m_input_tokens": getattr(split, "ephemeral_5m_input_tokens", None)}
            c = round(cost(self.model, usage), 6)
            self.session_spent += c
            line = {"time": now.isoformat(timespec="seconds"), "model": self.model, "purpose": purpose,
                    "request_id": getattr(message, "_request_id", None), "stop_reason": message.stop_reason,
                    **usage, "cost_usd": c}
            with self.log_path.open("a") as f:
                f.write(json.dumps(line) + "\n")
        return message
