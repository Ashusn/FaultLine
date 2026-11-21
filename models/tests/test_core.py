import json

import pytest

from flm import claims, metrics
from flm.common import read_json, read_jsonl, wpath
from flm.gates import relative_gate
from flm.headline import change
from flm.render import render_window
from flm.schema import group_key, map_fault, validate
from flm.signals import extract, rule_rank, service_features, signal_f1, FEATURES
from flm.simulate import simulate
from flm.split import SPLITS, leak_report, split_incidents


def test_simulator_is_seeded_and_valid():
    a = simulate(n_per_combo=1, seed=3)
    b = simulate(n_per_combo=1, seed=3)
    assert json.dumps(a) == json.dumps(b)
    for inc in a:
        validate(inc)


def test_db_faults_only_on_db_services():
    from flm.simulate import APPS

    for inc in simulate(n_per_combo=1, seed=0):
        if inc["label"]["fault"] in ("db_slow", "pool_exhaustion"):
            assert inc["label"]["service"] in APPS[inc["app"]]["db"]


def test_map_fault():
    assert map_fault("delay") == ("latency", "delay")
    assert map_fault("F3") == ("code_bug", "f3")
    with pytest.raises(ValueError):
        map_fault("banana")


def test_validate_rejects_bad_label():
    inc = simulate(n_per_combo=1, seed=0)[0]
    inc["label"] = {"service": "nope", "fault": "latency"}
    with pytest.raises(ValueError):
        validate(inc)


def test_rule_baseline_finds_culprit_often():
    incs = simulate(n_per_combo=2, seed=5)
    hits = sum(rule_rank(extract(i), i["services"])[0] == i["label"]["service"] for i in incs)
    assert hits / len(incs) > 0.5


def test_signal_f1():
    ref = {"anomalies": [{"service": "a", "signal": "cpu"}, {"service": "b", "signal": "error_rate"}]}
    assert signal_f1(ref, ref) == 1.0
    assert signal_f1({"anomalies": []}, ref) == 0.0
    assert signal_f1({"anomalies": [{"service": "a", "signal": "cpu"}]}, ref) == pytest.approx(2 / 3)
    assert signal_f1(None, ref) == 0.0


def test_features_shape():
    inc = simulate(n_per_combo=1, seed=1)[0]
    f = service_features(extract(inc), inc["services"])
    assert set(f) == set(inc["services"])
    assert all(len(v) == len(FEATURES) for v in f.values())


def test_render_respects_budget():
    inc = simulate(n_per_combo=1, seed=2)[0]
    t = render_window(inc, 3000)
    assert len(t) <= 3000 and "## metrics" in t and "## logs" in t


def test_split_has_no_leaks(cfg):
    incs = simulate(n_per_combo=3, seed=0)
    s = split_incidents(incs, cfg)
    rep = leak_report(s, cfg)
    assert rep["ok"], rep["problems"]
    assert all(i["app"] == "bookstore" for i in s["test_ood_app"])
    assert all(i["label"]["fault"] in cfg["split"]["holdout_faults"] for i in s["test_ood_fault"])
    assert {group_key(i) for i in s["train"]}.isdisjoint({group_key(i) for i in s["test"]})


def test_leak_report_catches_shared_group(cfg):
    incs = simulate(n_per_combo=2, seed=0, apps=["faultline"], faults=["latency"])
    s = {k: [] for k in SPLITS}
    s["train"] = incs[:1]
    s["test"] = incs[1:2]
    assert not leak_report(s, cfg)["ok"]


def test_metrics():
    assert metrics.recall_at_k(["a", "b", "c"], ["c", "d"], 3) == 0.5
    assert metrics.ndcg_at_k(["a", "b"], ["a"], 10) == 1.0
    assert metrics.ndcg_at_k(["x", "a"], ["a"], 10) < 1.0
    lo, hi = metrics.bootstrap_ci([1, 0] * 50)
    assert lo < 0.5 < hi


def test_gate_passes_equal_and_fails_big_drop(cfg):
    base = [{"id": str(i), "correct": 1.0 if i % 2 else 0.0} for i in range(200)]
    same = [dict(x) for x in base]
    worse = [{"id": x["id"], "correct": 0.0} for x in base]
    c = json.loads(json.dumps(cfg))
    c["eval"]["gate_drop"] = 0.03
    assert relative_gate(c, "verifier", same, base)["passed"]
    assert not relative_gate(c, "verifier", worse, base)["passed"]


def test_change_verbs():
    assert change("top1", 0.5, 0.7).startswith("raises")
    assert change("top1", 0.7, 0.5).startswith("lowers")
    assert change("top1", 0.5, 0.5).startswith("keeps")


def test_claim_perturbations_and_rule_check():
    import random

    inc = simulate(n_per_combo=1, seed=4, apps=["faultline"], faults=["error"])[2]
    inc["signals"] = extract(inc)
    rows = claims.build_for_incident(inc, random.Random(0), per_item_false=6)
    assert any(r["label"] == 1 for r in rows) and any(r["label"] == 0 for r in rows)
    for r in rows:
        if r["kind"] == "true":
            assert claims.rule_check(r["claim"], r["evidence"]) == 1, r
        if r["kind"] in ("invented_id", "wrong_number"):
            assert claims.rule_check(r["claim"], r["evidence"]) == 0, r


def test_data_steps(built):
    rep = read_json(wpath(built, "results", "data_report.json"))
    assert rep["ok"]
    for n in ("train", "val", "test"):
        assert read_jsonl(wpath(built, "data", "splits", f"{n}.jsonl"))
    corpus = read_jsonl(wpath(built, "data", "retrieval", "corpus.jsonl"))
    kinds = {d["kind"] for d in corpus}
    assert {"runbook_fault", "runbook_service", "incident"} <= kinds
    rr = read_json(wpath(built, "results", "retrieval_report.json"))
    assert rr["missing_runbook_faults"] == []
    test_ids = {i["incident_id"] for i in read_jsonl(wpath(built, "data", "splits", "test.jsonl"))}
    assert not any(d["id"].split(":", 1)[1] in test_ids for d in corpus if d["kind"] == "incident")
    comp = read_jsonl(wpath(built, "data", "compressor", "train.jsonl"))
    assert json.loads(comp[0]["target"])["anomalies"] is not None


def test_classifier_beats_or_matches_rule(built):
    from flm.models import classifier

    out = classifier.train(built, "classifier")
    m = classifier.evaluate(built, "classifier", out["local"], "test")["summary"]
    r = classifier.evaluate(built, "classifier", "rule", "test")["summary"]
    assert m["top1"] >= r["top1"] - 0.1
    assert m["fault_acc"] is not None and m["fault_acc"] > 0.2


def test_verifier_rule_baseline_eval(built):
    from flm.models import verifier

    s = verifier.evaluate(built, "verifier", "rule", "test")["summary"]
    assert 0.4 < s["correct"] <= 1.0
    assert "wrong_direction" in s["catch_by_kind"]


def test_steps_and_only_models(cfg, monkeypatch):
    from flm.steps import step_list

    steps = step_list(cfg)
    assert steps[0] == "check" and steps[-1] == "report"
    assert steps.index("compressor_sft:publish") < steps.index("compressor_grpo:train")
    monkeypatch.setenv("ONLY_MODELS", "classifier")
    assert [s for s in step_list(cfg) if ":" in s and not s.startswith("data")] == [f"classifier:{x}" for x in ("baseline", "train", "push", "verify", "eval", "publish")]
    monkeypatch.setenv("ONLY_MODELS", "nope")
    with pytest.raises(SystemExit):
        step_list(cfg)


def test_compressor_parse_and_rewards():
    from flm.models.compressor import parse, reward_f1, reward_length, reward_valid

    ref = json.dumps({"anomalies": [{"service": "a", "signal": "cpu"}]})
    good = 'sure {"anomalies":[{"service":"a","signal":"cpu"}]}'
    assert parse(good) is not None and parse("no json") is None and parse('{"x":1}') is None
    assert reward_valid([good, "bad"]) == [1.0, 0.0]
    assert reward_f1([good, "bad"], [ref, ref]) == [1.0, 0.0]
    assert reward_length([ref, ref * 5], [ref, ref])[1] < 0


def test_dry_configs_use_dry_names():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    for p in root.glob("*.yaml"):
        c = yaml.safe_load(p.read_text())
        assert c["hub"]["prefix"].startswith("DRY_") == p.stem.endswith("_dry"), p.name


def test_diffcheck_compare():
    from flm.diffcheck import compare

    ch = compare({"a": {"b": 1, "c": 2}}, {"a": {"b": 1, "c": 3}, "d": 4})
    assert ("a.c", 2, 3) in ch and ("d", "<missing>", 4) in ch and len(ch) == 2
