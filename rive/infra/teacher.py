"""Teacher gateway (RIVE v2.4 chapter 16; docs/rive-design.md section 3, item 7): the only way experiments reach the
DeepSeek API.

- The key is read from ~/.config/rive/deepseek.env (DEEPSEEK_API_KEY=...) or the environment; it is never logged.
- Every request is checked against a daily and a monthly budget in CNY before it is sent, priced at the worst
  case (all input uncached, every output token used).
- Every request is logged with its tokens and cost; a repeated request is answered from the cache.
- Requests that contain a forbidden text (hidden tests, sealed task content) are refused.
- Shadow mode (the default) is recorded with each request: callers must keep shadow answers out of the main
  experiments.
- Thinking is on at "high" by default (user, 2026-11-01: budget 100 CNY a month); effort="none" turns it off for
  cheap calls. Reasoning tokens count as output, in the token limit and in the price. The reasoning text is cached
  with the answer and returned on request.

Prices are those of the official page on 2026-11-01 and must be checked again before a formal experiment.
"""
import hashlib
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

KEY_FILE = Path("~/.config/rive/deepseek.env").expanduser()
BASE_URL = "https://api.deepseek.com"
# CNY per million tokens: (off-peak, peak). https://api-docs.deepseek.com/zh-cn/quick_start/pricing/, 2026-11-01.
PRICES = {"deepseek-flash": {"cached": (0.02, 0.04), "input": (1.0, 2.0), "output": (4.0, 8.0)},
          "deepseek-v4-pro": {"cached": (0.15, 0.30), "input": (4.5, 9.0), "output": (13.5, 27.0)}}
BEIJING = timezone(timedelta(hours=8))


class BudgetExceeded(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


def is_peak(when):
    """Mon–Fri 09:00–12:00 and 14:00–18:00 Beijing time. Public holidays are off-peak, but are not known here, so
    they are priced as peak (the safe side)."""
    t = when.astimezone(BEIJING)
    return t.weekday() < 5 and (9 <= t.hour < 12 or 14 <= t.hour < 18)


def cost(model, uncached, cached, output, peak):
    p, i = PRICES[model], int(peak)
    return (uncached * p["input"][i] + cached * p["cached"][i] + output * p["output"][i]) / 1e6


def reasoning_of(response):
    """The reply's reasoning text: the Responses API returns it in "reasoning" output items (section 6.18's
    correction: the gateway used to keep only output_text, and the reasoning was lost)."""
    return "".join(getattr(c, "text", "") or "" for item in getattr(response, "output", None) or []
                   if getattr(item, "type", None) == "reasoning" for c in getattr(item, "content", None) or [])


def read_key(path=KEY_FILE):
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class TeacherGateway:
    def __init__(self, log_dir, monthly_cny=100.0, daily_cny=10.0, model="deepseek-flash", shadow=True, client=None,
                 clock=lambda: datetime.now(timezone.utc)):
        self.dir = Path(log_dir)
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "usage.jsonl"
        self.monthly, self.daily, self.model, self.shadow = monthly_cny, daily_cny, model, shadow
        self.clock, self.lock, self._client = clock, threading.Lock(), client
        self.reserved = 0.0  # worst-case CNY of the requests in flight, so parallel requests cannot pass the caps

    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = read_key()
            if not key:
                raise Refused(f"no DEEPSEEK_API_KEY in the environment or {KEY_FILE}")
            self._client = OpenAI(api_key=key, base_url=BASE_URL, timeout=300, max_retries=3)
        return self._client

    def spent(self, now):
        """CNY logged this month and today (Beijing calendar)."""
        month = day = 0.0
        b = now.astimezone(BEIJING)
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                e = json.loads(line)
                t = datetime.fromisoformat(e["time"]).astimezone(BEIJING)
                if (t.year, t.month) == (b.year, b.month):
                    month += e["cost_cny"]
                    if t.date() == b.date():
                        day += e["cost_cny"]
        return month, day

    def ask(self, instructions, text, purpose, proposal_id=None, max_output_tokens=8000, effort="high",
            forbidden=(), with_reasoning=False):
        """The teacher's answer to `text` (with system `instructions`), thinking at `effort` (none, low, high,
        max). max_output_tokens covers the answer and the reasoning. forbidden: texts that must not leave this
        machine (hidden tests, sealed tasks). with_reasoning: (reasoning, answer) instead; a cached reply from
        before the reasoning was kept is asked again."""
        for f in forbidden:
            if f and (f in text or f in instructions):
                raise Refused(f"the request contains forbidden content ({purpose})")
        if read_key() and (read_key() in text or read_key() in instructions):
            raise Refused("the request contains the API key")
        request = json.dumps([self.model, instructions, text, max_output_tokens, effort])
        key = hashlib.sha256(request.encode()).hexdigest()
        cached = self.dir / "cache" / f"{key[:32]}.json"
        now = self.clock()
        if cached.exists():
            kept = json.loads(cached.read_text())
            if not with_reasoning or "reasoning" in kept:
                self._log(now, purpose, proposal_id, key, {"cache_hit": True, "cost_cny": 0.0})
                return (kept["reasoning"], kept["text"]) if with_reasoning else kept["text"]
        with self.lock:
            month, day = self.spent(now)
            month, day = month + self.reserved, day + self.reserved
            worst = cost(self.model, len(text) + len(instructions) + 100, 0, max_output_tokens, True)  # <= 1 token/char
            if month + worst > self.monthly or day + worst > self.daily:
                raise BudgetExceeded(f"spent or reserved {month:.3f} CNY this month and {day:.3f} today; this "
                                     f"request could cost {worst:.3f} (caps {self.monthly} and {self.daily})")
            self.reserved += worst
        try:  # the request itself runs outside the lock, so several can be in flight
            r = self.client().responses.create(model=self.model, instructions=instructions, input=text,
                                               max_output_tokens=max_output_tokens, reasoning={"effort": effort})
        except Exception:
            with self.lock:
                self.reserved -= worst
            raise
        u = r.usage
        hit = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        peak = is_peak(now)
        fields = {"cache_hit": False, "input_tokens": u.input_tokens, "cached_input_tokens": hit,
                  "output_tokens": u.output_tokens, "reasoning_tokens": reasoning, "peak": peak,
                  "cost_cny": round(cost(self.model, u.input_tokens - hit, hit, u.output_tokens, peak), 6)}
        with self.lock:
            self.reserved -= worst
            self._log(now, purpose, proposal_id, key, fields)
        answer, reasoning = r.output_text, reasoning_of(r)
        cached.write_text(json.dumps({"text": answer, "reasoning": reasoning}, ensure_ascii=False))
        return (reasoning, answer) if with_reasoning else answer

    def _log(self, now, purpose, proposal_id, key, fields):
        line = {"time": now.isoformat(timespec="seconds"), "model": self.model, "purpose": purpose,
                "proposal_id": proposal_id, "prompt_hash": key[:16], "shadow": self.shadow, **fields}
        with self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")
