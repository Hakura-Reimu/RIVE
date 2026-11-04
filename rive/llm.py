"""Client for the local vLLM OpenAI-compatible server. Only the trusted controller uses it."""
import threading

from openai import OpenAI


class LLM:
    def __init__(self, base_url="http://127.0.0.1:8000/v1", model="qwen3.5-4b", timeout=900):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def _count(self, response):
        with self._lock:
            self.usage["calls"] += 1
            if response.usage:
                self.usage["prompt_tokens"] += response.usage.prompt_tokens
                self.usage["completion_tokens"] += response.usage.completion_tokens

    def _chat(self, messages, schema, max_tokens, temperature, thinking, seed):
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}}, **kwargs)
        self._count(r)
        return r

    def chat(self, messages, schema=None, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        """Chat completion; with `schema`, the server constrains the output to that JSON Schema."""
        return self._chat(messages, schema, max_tokens, temperature, thinking, seed).choices[0].message.content or ""

    def generate(self, messages, max_tokens=1024, temperature=0.0, thinking=False, seed=0):
        """Chat completion with this call's own token counts (per-task budgets of Phase C workflows)."""
        r = self._chat(messages, None, max_tokens, temperature, thinking, seed)
        return {"text": r.choices[0].message.content or "", "finish_reason": r.choices[0].finish_reason,
                "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                "completion_tokens": r.usage.completion_tokens if r.usage else 0}

    def complete(self, prompt, max_tokens=1024, temperature=0.0, seed=0):
        """Plain completion of an already rendered prompt (evaluation path, same as the offline harness)."""
        r = self.client.completions.create(model=self.model, prompt=prompt, max_tokens=max_tokens,
                                           temperature=temperature, seed=seed)
        self._count(r)
        return r.choices[0].text
