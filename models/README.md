# FaultLine Models

This folder is Amaresh's part of FaultLine: the data and the five trained models.
It lives inside the FaultLine repo but runs on its own. It has its own setup, configs, tests and runner.

Ashutosh's AI framework uses these models through one API. He does not need to read anything here except `API.md` and `INCIDENT_SCHEMA.md`.

## What gets built

| Model | What it does | Base (small budget) | Main metric |
| --- | --- | --- | --- |
| embedder | finds the right runbook, past incident or code chunk for an incident | `BAAI/bge-base-en-v1.5` | nDCG@10 |
| reranker | puts the best found chunks on top | `BAAI/bge-reranker-base` | nDCG@10 |
| compressor_sft | turns a huge telemetry window into short JSON signals | `Qwen/Qwen3-1.7B` | signal F1 |
| compressor_grpo | the SFT compressor trained more with a reward we can check | compressor_sft | signal F1 |
| classifier | ranks services and guesses the fault kind in milliseconds | rule baseline | top 1 |
| verifier | checks if a report sentence is backed by its cited evidence | rule baseline | accuracy |

Every model is compared with its base on the same items. Every number comes from a results file.

## Layout

```
models/
  run.py                      check | pipeline | run <step> | status | reset <step> | steps | diffcheck
  setup.sh                    builds .venv with Python 3.11 and pinned packages
  requirements.txt            pinned, tested on Python 3.11
  requirements_vllm.txt       vLLM, installed by setup.sh unless INSTALL_VLLM=0
  .env.example                tokens only, never paths
  configs/
    config.yaml               small budget, the real run
    config_big_budget.yaml    bigger models and more data
    config_dry.yaml           tiny sizes, DRY_ repo names, loose gates
  flm/
    common.py                 config, paths, repo names, json helpers, run metadata
    schema.py                 the incident format (the contract) and fault kinds
    simulate.py               seeded incident simulator for two apps
    sources.py                RCAEval loader, simulator, and JSONL from the FaultLine harness
    signals.py                rule signal extractor, features, rule baseline ranking
    render.py                 turns an incident into the long text the compressor reads
    split.py                  split by incident group, hold out a fault kind and an app, leak check
    retrieval.py              corpus from runbooks, code and past incidents, queries, positives
    claims.py                 true claims and six kinds of broken claims, rule checker
    synthetic.py              optional LLM paraphrased queries with a filter
    data.py                   data steps
    models/                   embedder, reranker, compressor, classifier, verifier
    pipeline.py               baseline, train, push, verify, eval, publish, report
    gates.py                  relative gate with a paired bootstrap interval
    metrics.py                recall, nDCG, MRR, bootstrap intervals
    hub.py                    uploads from /tmp snapshots, retries only server errors
    card.py                   model and dataset cards built from results files
    headline.py               HEADLINE.md and headline.json from results only
    check.py                  environment check, including a real batch size memory test
    diffcheck.py              lists every config value that differs from git HEAD
    serve.py                  the model API for Ashutosh's framework
  runbooks/                   fault and service runbooks, also the RAG corpus
  scripts/
    start_cell.py             paste into a Jupyter cell on the pod
    precheck.sh               compile with 3.11, bash syntax, DRY_ names, diffcheck, CPU tests
    stop.sh                   kills everything including vLLM workers
    health.sh                 quick pod health view
    headline.py               rebuild the eval page by hand
  tests/                      CPU tests, plus slow tests that train tiny models
```

## Step order

```
check
data:build         collect incidents, extract signals, split, leak check
data:compressor    window to signals pairs
data:retrieval     corpus, queries, training pairs
data:claims        verifier pairs
data:push          four datasets to HuggingFace
for each model:
  <model>:baseline   base on val (quick) and on test splits (full), plus extra baselines
  <model>:train
  <model>:push       weights, adapter without README, training curves
  <model>:verify     download from hub, quick eval, gate against base, delete download
  <model>:eval       full eval, results uploaded to the repo eval folder
  <model>:publish    card with results, added to the collection
report               HEADLINE.md, then local weights deleted
```

`state.json` in the work folder records finished steps. Running `pipeline` again resumes at the first unfinished step.
Every step runs in its own process and reads the config fresh.

## On your laptop (free)

```bash
cd FaultLine/models
INSTALL_VLLM=0 bash setup.sh
PYBIN=.venv/bin/python bash scripts/precheck.sh
.venv/bin/python -m pytest -q -m slow
```

The slow tests build tiny random models on your CPU and run every training path once: embedder, reranker, verifier, SFT, GRPO, and the full model step chain with a fake hub.

## On the pod

1. Pick an Ada or Ampere GPU (L40S is fine) with a CUDA 13 driver. CUDA 13 needs driver 580 or newer, and the pinned torch is built for it. If your template has an older driver, run setup with `TORCH_INDEX=https://download.pytorch.org/whl/cu128 INSTALL_VLLM=0` and set `use_vllm: false`.
2. Open Jupyter, paste `scripts/start_cell.py` into one cell, put fresh tokens in it, set `MODE = "dry"`, run it.
3. When the dry run ends with `PROOF OK`, check the dry eval page, then run again with `MODE = "small"`.
4. Watch `summary.txt` on your phone at `AmareshHebbar/FaultLine_Proof`.
5. The pod terminates itself only after the last upload.

Useful commands on the pod:

```bash
cd /workspace/FaultLine/models
export FLM_CONFIG=configs/config.yaml FLM_WORK=/workspace/faultline_models_small
.venv/bin/python run.py status
.venv/bin/python run.py reset compressor_sft:verify
.venv/bin/python run.py reset "embedder:*"
ONLY_MODELS=classifier,verifier .venv/bin/python run.py pipeline
bash scripts/stop.sh
bash scripts/health.sh
```

## Where the data comes from

1. **Simulator.** A seeded simulator makes labelled incidents for two apps: the FaultLine shop and a bookstore. It spreads faults to callers, adds noise and a random red herring, and writes metrics, logs and spans. It lets every step run with no network and no platform.
2. **RCAEval.** 735 real labelled failure cases from Online Boutique, Sock Shop and Train Ticket, MIT license, read from the HuggingFace Parquet copy.
3. **FaultLine harness.** When Ashutosh's eval harness writes incidents in the contract format (`INCIDENT_SCHEMA.md`), add the file paths to `data.jsonl` in the config. No code change is needed.

## Rules from the playbook, and what is new

From the playbook:

* Dry run first with `DRY_` names, tiny sizes and loose gates.
* Gates are relative to the base, never a fixed number.
* No paths in `.env`. The runner sets the work folder per mode.
* Adapter uploads ignore README. Cards use hub ids.
* Uploads go from a `/tmp` snapshot. Only server errors are retried. Not found fails at once.
* Verify downloads are deleted in `finally`. Local weights are deleted after the report.
* The check trains at the real batch size and fails if the GPU is busy or below capability 8.0.
* `MPLBACKEND=Agg` is set first thing.
* Numbers go into cards and pages only from results files.

New in this lab:

* **Split by incident group, not by row.** A group is app, service and fault. No group is in two splits. A test fails the build if one is.
* **Out of distribution tests.** A whole fault kind and a whole app are held out, so we see how the models do on failures they never saw.
* **Paired bootstrap gates.** The gate uses the 95% interval of the difference between model and base on the same items. This replaces the "missed by one problem" problem.
* **diffcheck.** `run.py diffcheck` prints every config value that differs from git HEAD, and the pod runs it before the pipeline.
* **Run metadata.** Every results file has the commit, config name, seed and data hash. Every step logs its time and its GPU cost.
* **Token budget guard.** Compressor rows are shortened to fit `max_length`. The run stops if more than 5% would not fit, instead of training on cut answers.
* **Pinned for 3.11 and tested.** The pins were resolved together with vLLM and the whole test suite passes on Python 3.11 with them.

## Settings that matter most

* `split.holdout_faults` and `split.holdout_apps`: what the out of distribution tests hold out.
* `eval.quick_n`, `eval.min_quick_items`: keep quick evals at 100 items or more on real runs.
* `eval.gate_drop` and `eval.gate_drop_by_model`: how much worse than base a model may be, per model.
* `models.<name>.max_length` and `max_input_chars`: the compressor token budget.
* `retrieval.code_dirs`: point it at Ashutosh's service code once it exists.
