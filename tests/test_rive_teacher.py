"""The teacher gateway (rive/infra/teacher.py) with a stand-in API client: budget, cost, cache, refusals, log."""
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from rive.infra.teacher import BudgetExceeded, Refused, TeacherGateway, cost, is_peak


class FakeResponses:
    def __init__(self):
        self.requests = []

    def create(self, **kw):
        self.requests.append(kw)
        usage = SimpleNamespace(input_tokens=8000, output_tokens=2000,
                                input_tokens_details=SimpleNamespace(cached_tokens=3000),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=0))
        return SimpleNamespace(usage=usage, output_text=f"answer {len(self.requests)}")


def gateway(tmp_path, when, **kw):
    client = SimpleNamespace(responses=FakeResponses())
    return TeacherGateway(tmp_path, client=client, clock=lambda: when, **kw), client.responses


OFF_PEAK = datetime(2026, 11, 1, 13, 0, tzinfo=timezone.utc)  # Sunday 21:00 Beijing
PEAK = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)  # Monday 10:00 Beijing


def test_peak_hours_and_prices():
    assert not is_peak(OFF_PEAK) and is_peak(PEAK)
    assert not is_peak(datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc))  # Monday 12:30, the lunch gap
    assert cost("deepseek-flash", 5000, 3000, 2000, peak=False) == pytest.approx((5000 + 60 + 8000) / 1e6)
    assert cost("deepseek-flash", 5000, 3000, 2000, peak=True) == pytest.approx(2 * (5000 + 60 + 8000) / 1e6)


def test_requests_are_logged_priced_and_cached(tmp_path):
    gw, api = gateway(tmp_path, OFF_PEAK)
    assert gw.ask("You review experiments.", "Why did change r2 fail?", "diagnosis", "p7") == "answer 1"
    assert api.requests[0]["reasoning"] == {"effort": "high"} and api.requests[0]["model"] == "deepseek-flash"
    assert api.requests[0]["max_output_tokens"] == 8000
    assert gw.ask("You review experiments.", "Why did change r2 fail?", "diagnosis", "p7") == "answer 1"
    assert len(api.requests) == 1  # the repeat came from the cache
    lines = [json.loads(x) for x in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert lines[0]["cost_cny"] == pytest.approx(0.01306) and lines[0]["shadow"] and lines[0]["proposal_id"] == "p7"
    assert lines[1]["cache_hit"] and lines[1]["cost_cny"] == 0


def test_budgets_and_refusals(tmp_path):
    gw, api = gateway(tmp_path, PEAK, daily_cny=0.1)
    for i in range(2):
        gw.ask("i", f"question {i}", "test")  # 0.026 CNY each at peak; the next could cost 0.064 more
    with pytest.raises(BudgetExceeded):
        gw.ask("i", "question 2", "test")
    assert len(api.requests) == 2
    with pytest.raises(Refused):
        gw.ask("i", "the hidden test is assert f(1) == 2", "test", forbidden=["assert f(1) == 2"])
    assert len(api.requests) == 2


def test_requests_in_flight_count_against_the_budget(tmp_path):
    """Requests run outside the lock, so a request still in flight holds its worst case against the caps."""
    started, release = threading.Event(), threading.Event()

    class Slow(FakeResponses):
        def create(self, **kw):
            started.set()
            release.wait(5)
            return super().create(**kw)

    client = SimpleNamespace(responses=Slow())
    gw = TeacherGateway(tmp_path, client=client, clock=lambda: PEAK, daily_cny=0.1)  # worst case 0.064 at peak
    first = threading.Thread(target=gw.ask, args=("i", "question 0", "test"))
    first.start()
    assert started.wait(5)
    with pytest.raises(BudgetExceeded):  # 0.064 reserved + 0.064 > 0.1
        gw.ask("i", "question 1", "test")
    release.set()
    first.join(5)
    assert gw.reserved == pytest.approx(0) and len(client.responses.requests) == 1
    gw.ask("i", "question 1", "test")  # 0.026 spent + 0.064 <= 0.1
    assert len(client.responses.requests) == 2


def test_the_reasoning_is_kept_and_an_old_cache_entry_is_asked_again(tmp_path):
    class Thinking(FakeResponses):
        def create(self, **kw):
            r = super().create(**kw)
            r.output = [SimpleNamespace(type="reasoning", content=[SimpleNamespace(type="reasoning_text", text="17*23")]),
                        SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="no")])]
            return r

    client = SimpleNamespace(responses=Thinking())
    gw = TeacherGateway(tmp_path, client=client, clock=lambda: OFF_PEAK)
    gw.ask("i", "old question", "test")
    (cache,) = (tmp_path / "cache").glob("*.json")
    cache.write_text(json.dumps({"text": "answer 1"}))  # as the gateway cached replies before
    assert gw.ask("i", "old question", "test") == "answer 1" and len(client.responses.requests) == 1
    assert gw.ask("i", "old question", "test", with_reasoning=True) == ("17*23", "answer 2")
    assert gw.ask("i", "old question", "test", with_reasoning=True) == ("17*23", "answer 2")
    assert len(client.responses.requests) == 2


def test_no_key_means_no_client(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("rive.infra.teacher.KEY_FILE", tmp_path / "missing.env")
    monkeypatch.setattr("rive.infra.teacher.read_key", lambda path=None: None)
    gw = TeacherGateway(tmp_path, clock=lambda: OFF_PEAK)
    with pytest.raises(Refused):
        gw.ask("i", "q", "test")
