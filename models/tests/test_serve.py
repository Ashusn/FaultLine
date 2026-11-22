from fastapi.testclient import TestClient

from flm.serve import app
from flm.signals import extract
from flm.simulate import simulate


def test_rule_fallbacks(monkeypatch):
    for k in ("FLM_CLASSIFIER", "FLM_VERIFIER", "FLM_EMBEDDER", "FLM_COMPRESSOR_URL"):
        monkeypatch.delenv(k, raising=False)
    c = TestClient(app)
    assert c.get("/v1/health").json()["ok"]
    inc = simulate(n_per_combo=1, seed=0)[0]
    r = c.post("/v1/classify", json={"signals": extract(inc), "services": inc["services"]}).json()
    assert r["source"] == "rule" and r["services"][0] in inc["services"]
    v = c.post("/v1/verify", json={"claims": [{"claim": "a cpu rose from 1 to 2 [E1].", "evidence": "[E1] metric a cpu (cpu) mean before=1 after=2"}]}).json()
    assert v["results"][0]["supported"] is True
    assert c.post("/v1/embed", json={"texts": ["x"]}).status_code == 503
    assert c.post("/v1/compress", json={"text": "x"}).status_code == 503
