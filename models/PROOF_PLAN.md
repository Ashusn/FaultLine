#### FaultLine: how each resume number gets made and proven

#### The target line

Root cause AI for microservices that names the failing service correctly in X% of 500 injected incidents (Y% in top 3) with 0 uncited claims; fine tuned and published 5 models on HuggingFace, including a Qwen3 1.7B compressor that shrinks telemetry windows by Z% and an embedder that lifts Recall@10 by W% over bge base.

#### Where each number comes from

| Id | Number | Made by | Results file | Public link |
| --- | --- | --- | --- | --- |
| top1 | X | classifier on the `test_faultline` split | `eval/classifier_full.json` | `PROOF.md#top1` |
| top3 | Y | same | same | `PROOF.md#top3` |
| uncited | 0 | Ashutosh's agent eval with the verifier on | `report/agent_eval.json` | `PROOF.md#uncited` |
| compress | Z | compressor_grpo, 1 minus token ratio | `eval/compressor_grpo_full.json` | `PROOF.md#compress` |
| recall | W | embedder against bge base, Recall@10 | `eval/embedder_full.json` and `eval/embedder_base_full.json` | `PROOF.md#recall` |

Until the 500 real incidents exist, top1, top3, compress and recall come from the `test` split (simulator plus RCAEval). The proof page always says which split each number used.

#### Steps

1. **Now, on the pod:** run the small config. You get every model, the eval pages and a first proof page on simulator and RCAEval data.
2. **Ashutosh's harness writes 500 incidents** in the `INCIDENT_SCHEMA.md` format with `"source": "faultline"`. Copy the file to the pod and add its path under `data.jsonl` in the config. These incidents go only into the `test_faultline` split. They are never used for training.
3. **Ashutosh's agent eval writes `agent_eval.json`** with at least `{"uncited_claims": 0, "n": 500, "top1": 0.0, "top3": 0.0}` from the same 500 incidents. Copy it into the work folder under `results/`.
4. **Rerun from data:build:** `run.py reset "data:*"`, then `run.py reset report`, then `run.py pipeline`. The models are evaluated again on `test_faultline`, and the proof page and resume line fill in.
5. **Check it like a stranger would:** run `run.py reproduce` on a fresh pod. Every line must say MATCH.
6. **Paste the resume line** from the bottom of `PROOF.md`. Every number in it already links to its proof.

#### Rules

* No number goes in the resume before it is on the proof page.
* If a number is worse than hoped, publish it anyway and say what you learned.
* The compressor number means tokens saved per telemetry window. If you later measure tokens saved per full agent investigation, add it as a new claim instead of renaming this one.
