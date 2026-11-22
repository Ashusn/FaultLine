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
    assert [s for s in step_list(cfg) if ":" in s and not s.startswith(("data", "bench"))] == [f"classifier:{x}" for x in ("baseline", "train", "push", "verify", "eval", "publish")]
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


def test_dedicated_source_goes_to_its_own_split(cfg):
    from flm.simulate import simulate

    incs = simulate(n_per_combo=1, seed=0, apps=["faultline"])
    for i in incs[:5]:
        i["source"] = "faultline"
        i["incident_id"] = "real-" + i["incident_id"]
    s = split_incidents(incs, cfg)
    assert len(s["test_faultline"]) == 5
    assert leak_report(s, cfg)["ok"]


MODEL_CLAIMS = {
    "tolerance": 0.5,
    "claims": [
        {"id": "top1", "text": "top1 {value}% of {n}", "model": "classifier", "metric": "top1", "kind": "percent", "split": "test_faultline", "fallback_splits": ["test"]},
        {"id": "top3", "text": "top3 {value}% of {n}", "model": "classifier", "metric": "top3", "kind": "percent", "split": "test_faultline", "fallback_splits": ["test"]},
        {"id": "uncited", "text": "{value} uncited", "source": "external", "file": "agent_eval.json", "key": "uncited_claims"},
        {"id": "compress", "text": "{value}%", "model": "compressor_grpo", "metric": "token_ratio", "kind": "one_minus_percent", "split": "test"},
        {"id": "recall", "text": "{value}%", "model": "embedder", "metric": "recall@10", "kind": "relative_gain_percent", "split": "test"},
    ],
    "latex": "names it in {top1}\\% of {top1_n} injected incidents with {uncited} uncited claims, shrinks by {compress}\\% {proof}",
}


def test_proof_page_and_latex(cfg, tmp_path):
    from flm.common import write_json
    from flm.proof import build_proof

    c = json.loads(json.dumps(cfg))
    c["_work"] = str(tmp_path / "w")
    c["proof"] = MODEL_CLAIMS
    write_json(wpath(c, "results", "classifier_full.json"), {"splits": {"test": {"top1": 0.8123, "top1_ci95": [0.75, 0.86], "top3": 0.97, "top3_ci95": [0.94, 0.99], "n": 500}}})
    write_json(wpath(c, "results", "classifier_base_full.json"), {"splits": {"test": {"top1": 0.7, "top3": 0.9, "n": 500}}})
    write_json(wpath(c, "results", "embedder_full.json"), {"splits": {"test": {"recall@10": 0.66, "n": 80}}})
    write_json(wpath(c, "results", "embedder_base_full.json"), {"splits": {"test": {"recall@10": 0.55, "n": 80}}})
    write_json(wpath(c, "results", "agent_eval.json"), {"uncited_claims": 0, "n": 500})
    out = build_proof(c, tmp_path)
    p = json.loads((out / "proof.json").read_text())
    got = {x["id"]: x["result"] for x in p["claims"]}
    assert got["top1"]["value"] == 81.2 and got["top1"]["ci95"] == [75.0, 86.0]
    assert got["recall"]["value"] == 20.0
    assert got["uncited"]["value"] == 0
    assert got["compress"] is None
    latex = p["latex"]
    assert "81.2}" in latex and "of 500 injected" in latex and "{X}" in latex and "PROOF.md#top1" in latex
    assert (out / "agent_eval.json").exists()
    assert "Not measured yet." in (out / "PROOF.md").read_text()


def test_reproduce_matches_published(built, monkeypatch):
    import shutil

    from flm import pipeline, reproduce
    from flm.common import wdir, write_json
    from flm.models import classifier
    from flm.proof import build_proof

    c = json.loads(json.dumps(built))
    c["proof"] = MODEL_CLAIMS
    out = classifier.train(c, "classifier")
    write_json(wpath(c, "results", "classifier_train.json"), out)
    write_json(wpath(c, "results", "classifier_full.json"), pipeline._full(c, "classifier", out["local"]))
    build_proof(c, wdir(c, "published"))
    keep = wdir(c, "keep_model")
    shutil.copytree(out["local"], keep, dirs_exist_ok=True)

    def fake_model(cfg, name):
        d = wdir(cfg, "repro_models", name)
        shutil.copytree(keep, d, dirs_exist_ok=True)
        return str(d)

    monkeypatch.setattr(reproduce, "fetch_data", lambda cfg: None)
    monkeypatch.setattr(reproduce, "fetch_model", fake_model)
    monkeypatch.setenv("CLAIMS", "top1,top3")
    reproduce.run(c)
    rows = read_json(wpath(c, "results", "reproduce.json"))
    assert [r["id"] for r in rows] == ["top1", "top3"] and all(r["match"] for r in rows)


def test_bench_select_and_score(built):
    from flm import bench
    from flm.data import load_split

    c = json.loads(json.dumps(built))
    c["bench"]["size"] = 25
    pool = [(s, i) for s in bench.POOL for i in load_split(c, s)]
    a = bench.select(c, pool)
    b = bench.select(c, pool)
    assert [i["incident_id"] for _, i in a] == [i["incident_id"] for _, i in b]
    assert len(a) == 25 and len({i["incident_id"] for _, i in a}) == 25
    assert len({i["app"] for _, i in a}) >= 2
    labels = [{"incident_id": i["incident_id"], "service": i["label"]["service"], "fault": i["label"]["fault"]} for _, i in a]
    perfect = [{"incident_id": l["incident_id"], "services": [l["service"]], "fault": l["fault"]} for l in labels]
    assert bench.score(perfect, labels)["top1"] == 1.0
    assert bench.score(perfect[:5], labels)["missing_predictions"] == 20


def test_resume_line_only_uses_measured_numbers(cfg, tmp_path):
    from flm.common import write_json, wpath
    from flm.proof import build_proof

    c = json.loads(json.dumps(cfg))
    c["_work"] = str(tmp_path)
    write_json(wpath(c, "results", "bench", "bench_results.json"), {"version": "v0.1", "n": 40, "systems": {}, "headline": {"system": "end_to_end", "n": 40, "top1": 71.2, "top3": 93.0, "token_reduction": None, "recall10_base": 40.0, "recall10_trained": 55.5}})
    out = build_proof(c, tmp_path)
    latex = json.loads((out / "proof.json").read_text())["latex"]
    assert "{71.2}\\% of 40 held out incidents" in latex
    assert "uncited" not in latex and "compressor" not in latex
    assert "Recall@10 from" in latex and "{55.5}" in latex
    assert "PROOF.md" in latex


def test_reproduce_rescores_bench_predictions(cfg, tmp_path, monkeypatch):
    import shutil

    from flm import bench, reproduce
    from flm.common import wdir, write_json, write_jsonl
    from flm.proof import build_proof

    c = json.loads(json.dumps(cfg))
    c["_work"] = str(tmp_path)
    labels = [{"incident_id": f"i{k}", "service": "a", "fault": "latency"} for k in range(20)]
    preds = [{"incident_id": f"i{k}", "services": ["a" if k % 5 else "b", "a"], "fault": "latency"} for k in range(20)]
    sc = bench.score(preds, labels)
    res = {"version": "v0.1", "n": 20, "systems": {"end_to_end": {"top1": sc["top1"], "top1_ci95": sc["top1_ci95"]}}, "headline": {"system": "end_to_end", "n": 20, "top1": round(100 * sc["top1"], 1), "top3": round(100 * sc["top3"], 1)}}
    write_json(wpath(c, "results", "bench", "bench_results.json"), res)
    write_jsonl(wpath(c, "results", "bench", "pred_end_to_end.jsonl"), preds)
    pub = wdir(c, "published")
    build_proof(c, pub)
    shutil.copytree(wdir(c, "results", "bench"), pub / "results" / "v0.1")
    write_jsonl(pub / "labels.jsonl", labels)
    monkeypatch.setattr(reproduce, "fetch_data", lambda cfg: None)
    monkeypatch.setenv("CLAIMS", "top1,top3")
    reproduce.run(c)
    rows = read_json(wpath(c, "results", "reproduce.json"))
    assert [r["id"] for r in rows] == ["top1", "top3"] and all(r["match"] for r in rows)
    assert rows[0]["published"] == 80.0


def test_upload_run_sends_logs_results_and_state(cfg, tmp_path, monkeypatch):
    from flm import hub
    from flm.common import wdir

    c = json.loads(json.dumps(cfg))
    c["_work"] = str(tmp_path)
    (wdir(c, "logs") / "data__build.log").write_text("leak check PASSED\n")
    write = wdir(c, "results") / "x.json"
    write.write_text("{}")
    seen = {}

    class FakeApi:
        def create_repo(self, repo, **k):
            seen["repo"] = repo

        def upload_folder(self, folder_path, path_in_repo, **k):
            from pathlib import Path

            seen["path"] = path_in_repo
            seen["files"] = sorted(p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file())

    monkeypatch.setattr(hub, "api", lambda: FakeApi())
    assert hub.upload_run(c, final=True)
    rid = hub.run_id(c)
    assert seen["repo"].endswith("DRY_FaultLine_Runs") and seen["path"] == f"runs/{rid}"
    assert "logs/data__build.log" in seen["files"] and "results/x.json" in seen["files"] and "state.json" in seen["files"]
    assert hub.run_id(c) == rid


def test_run_logged_writes_step_log(tmp_path, monkeypatch):
    import run as runner

    log = tmp_path / "steps.log"
    rc = runner._run_logged("nope:step", log)
    assert rc != 0 and "unknown step" in log.read_text()
