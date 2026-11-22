import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flm.common import dataset_repo, load_cfg, read_json, repo_id, results_path

TEST = os.environ.get("RESUME_SPLIT", "test")


def pct(x):
    return f"{100 * x:.1f}\\%"


def link(url, text):
    return f"\\href{{{url}}}{{{text}}}"


def get(cfg, name, kind, split, key):
    p = results_path(cfg, name, kind)
    if not p.exists():
        return None
    return read_json(p).get("splits", {}).get(split, {}).get(key)


def main():
    cfg = load_cfg()
    hf = "https://huggingface.co"
    page = f"{hf}/datasets/{dataset_repo(cfg, 'incidents')}/blob/main/report/HEADLINE.md"
    missing = []

    final_p = os.environ.get("FINAL_EVAL")
    if final_p and Path(final_p).exists():
        f = read_json(final_p)
        n, top1, top3, uncited = f["n"], f["top1"], f["top3"], f.get("uncited_claims")
        page_final = f.get("url", page)
        root = f"names the failing service correctly in {link(page_final, pct(top1))} of {n} injected incidents ({link(page_final, pct(top3))} in top 3)"
        if uncited is not None:
            root += f" with {link(page_final, str(uncited))} uncited claims"
    else:
        top1 = get(cfg, "classifier", "full", TEST, "top1")
        top3 = get(cfg, "classifier", "full", TEST, "top3")
        n = get(cfg, "classifier", "full", TEST, "n")
        if top1 is None:
            missing.append("classifier eval (run classifier:eval)")
            root = "names the failing service correctly in X\\% of N held out incidents (Y\\% in top 3)"
        else:
            root = f"names the failing service correctly in {link(page, pct(top1))} of {n} held out incidents ({link(page, pct(top3))} in top 3)"
        missing.append("final 500 incident eval from the FaultLine harness (set FINAL_EVAL=path/to/final_eval.json)")

    comp = next((c for c in ("compressor_grpo", "compressor_sft") if results_path(cfg, c, "full").exists()), None)
    if comp:
        ratio = get(cfg, comp, "full", TEST, "token_ratio")
        f1 = get(cfg, comp, "full", TEST, "signal_f1")
        comp_url = f"{hf}/{repo_id(cfg, comp)}"
        f1_txt = f"{f1:.2f}"
        comp_txt = f"a {link(comp_url, 'Qwen3 1.7B compressor')} that cuts telemetry tokens by {link(page, pct(1 - ratio))} at {link(page, f1_txt)} signal F1"
    else:
        missing.append("compressor eval (run compressor_sft or compressor_grpo)")
        comp_txt = "a Qwen3 1.7B compressor that cuts telemetry tokens by Z\\%"

    em_new = get(cfg, "embedder", "full", TEST, "recall@10")
    em_old = get(cfg, "embedder", "base_full", TEST, "recall@10")
    if em_new is not None and em_old is not None:
        emb_url = f"{hf}/{repo_id(cfg, 'embedder')}"
        emb_txt = f"an {link(emb_url, 'embedder')} that lifts Recall@10 from {link(page, pct(em_old))} to {link(page, pct(em_new))} over bge base"
    else:
        missing.append("embedder eval (run embedder:eval)")
        emb_txt = "an embedder that lifts Recall@10 by W\\% over bge base"

    coll = f"{hf}/collections/{cfg['hub']['namespace']}"
    line = (
        f"\\item \\href{{https://github.com/amareshhebbar/FaultLine}}{{\\textbf{{FaultLine}}}} $\\cdot$ {link(coll, 'Models')} $\\cdot$ {link(page, 'Evals')} \\textit{{(I own the models)}}: "
        f"Root cause AI for microservices that {root}; fine tuned 5 models, including {comp_txt} and {emb_txt}."
    )
    print(line)
    if missing:
        print("\nstill missing:", file=sys.stderr)
        for m in missing:
            print(" ", m, file=sys.stderr)
    print("\nsplit used:", TEST, "(set RESUME_SPLIT=test_ood_app to report the held out app)", file=sys.stderr)


if __name__ == "__main__":
    main()
