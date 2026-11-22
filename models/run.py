import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import subprocess
import sys
import time

from flm.common import ROOT, load_cfg, log
from flm.steps import load_state, mark_done, reset, step_list

USAGE = "usage: python run.py check | pipeline | run <step> | status | reset <step> | steps | proof | reproduce | diffcheck [configs...]"


def run_step(cfg, step):
    from flm import check, data, pipeline

    if step == "check":
        return check.run(cfg)
    if step == "data:build":
        return data.step_build(cfg)
    if step == "data:compressor":
        return data.step_compressor_data(cfg)
    if step == "data:retrieval":
        return data.step_retrieval(cfg)
    if step == "data:claims":
        return data.step_claims(cfg)
    if step == "data:push":
        return data.step_push(cfg)
    if step.startswith("bench:"):
        from flm import bench

        return {"bench:freeze": bench.step_freeze, "bench:run": bench.step_run, "bench:publish": bench.step_publish}[step](cfg)
    if step == "report":
        return pipeline.step_report(cfg)
    name, _, stage = step.partition(":")
    if name in cfg["models"] and stage in pipeline.MODEL_STEPS:
        return pipeline.MODEL_STEPS[stage](cfg, name)
    raise SystemExit(f"unknown step {step!r}")


def _run_logged(step, logfile):
    with open(logfile, "a") as lf:
        lf.write(f"==== {step} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        p = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "run", step], cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        return p.wait()


def pipeline_all(cfg):
    from flm.common import wdir
    from flm.hub import run_id, runs_url, upload_run

    upload = os.environ.get("UPLOAD_LOGS", "1") == "1"
    if upload:
        run_id(cfg)
    done = set(load_state(cfg)["done"])
    for step in step_list(cfg):
        if step in done:
            log(f"skip {step} (done)")
            continue
        log(f"==== {step}")
        t = time.time()
        rc = _run_logged(step, wdir(cfg, "logs") / f"{step.replace(':', '__')}.log")
        cfg = load_cfg()
        dt = time.time() - t
        if rc != 0:
            log(f"STOPPED at {step} (exit {rc})")
            if upload:
                upload_run(cfg)
            sys.exit(rc)
        mark_done(cfg, step, dt)
        cost = dt / 3600 * float(cfg.get("gpu_hourly_usd", 0))
        log(f"{step} finished in {dt / 60:.1f} min, about ${cost:.2f}")
        if upload:
            upload_run(cfg)
    log("pipeline complete")
    if upload:
        ok = upload_run(cfg, final=True) or upload_run(cfg, final=True)
        log("all logs and results are on HuggingFace:", runs_url(cfg))
        if ok and os.environ.get("CLEAN_WORK") == "1":
            import shutil

            shutil.rmtree(cfg["_work"], ignore_errors=True)
            log("local work folder deleted, everything lives on HuggingFace")


def main(argv):
    if not argv:
        raise SystemExit(USAGE)
    cmd = argv[0]
    if cmd == "diffcheck":
        from flm.diffcheck import run as diff

        paths = argv[1:] or sorted(str(p) for p in (ROOT / "configs").glob("*.yaml"))
        diff(paths)
        return
    cfg = load_cfg()
    if cmd == "check":
        run_step(cfg, "check")
        mark_done(cfg, "check", 0)
    elif cmd == "pipeline":
        pipeline_all(cfg)
    elif cmd == "run" and len(argv) == 2:
        run_step(cfg, argv[1])
    elif cmd == "status":
        st = load_state(cfg)
        for s in step_list(cfg):
            print(("done  " if s in st["done"] else "todo  ") + s + (f"  {st['durations'].get(s)}s" if s in st["durations"] else ""))
    elif cmd == "reset" and len(argv) == 2:
        print("reset:", reset(cfg, argv[1]) or "nothing matched")
    elif cmd == "proof":
        from flm.common import wdir
        from flm.headline import build_headline
        from flm.proof import build_proof

        out = build_headline(cfg)
        build_proof(cfg, out)
        print(out / "PROOF.md")
    elif cmd == "reproduce":
        from flm.reproduce import run as reproduce

        reproduce(cfg)
    elif cmd == "steps":
        print("\n".join(step_list(cfg)))
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main(sys.argv[1:])
