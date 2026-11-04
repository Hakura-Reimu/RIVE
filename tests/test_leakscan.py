from rive.benchmark.leakscan import LeakScanner

PROTECTED = [{"id": "multiple_7", "messages": [{"role": "user", "content":
              "Find the boiling point of water at an altitude of 5000 meters above sea level please"}],
              "functions": [{"name": "boiling_point_at_altitude", "parameters": {}}],
              "ground_truth": [{"boiling_point_at_altitude": {"substance": ["distilled water"], "altitude": [5000]}}]}]


def test_clean_router_passes():
    s = LeakScanner(PROTECTED)
    assert s.scan({"router.py": "def select(task, tools, history=None):\n    return {'tools': [], 'hint': ''}\n"}) == {
        "violations": [], "warnings": []}


def test_copied_request_and_answer_value_are_violations():
    s = LeakScanner(PROTECTED)
    r = s.scan({"router.py": "# boiling point of water at an altitude of 5000 meters above\nX = 'Distilled Water'\n"})
    kinds = {v["kind"] for v in r["violations"]}
    assert kinds == {"request_text", "answer_value"}


def test_single_word_answer_values_are_not_flagged():
    item = dict(PROTECTED[0], ground_truth=[{"boiling_point_at_altitude": {"substance": ["mitochondria"]}}])
    assert LeakScanner([item]).scan({"router.py": "KEYWORDS = ['mitochondria', 'nucleus']\n"})["violations"] == []


def test_protected_function_name_is_a_warning_unless_visible_in_dev():
    code = {"router.py": "PREFERRED = ['boiling_point_at_altitude']\n"}
    assert LeakScanner(PROTECTED).scan(code)["warnings"]
    assert not LeakScanner(PROTECTED, dev_function_names={"boiling_point_at_altitude"}).scan(code)["warnings"]
