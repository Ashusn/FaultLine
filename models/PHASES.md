#### FaultLine Models: phases

#### Phase 0: Lab setup (done in this drop)
The folder, configs, runner, checks, gates, cards, eval page, pod scripts and tests.
Done when: `scripts/precheck.sh` passes and `pytest -m slow` trains every tiny model on CPU.

#### Phase 1: Data (done in this drop, run it on the pod)
The incident contract, the simulator, the RCAEval loader, signals, splits with a leak check, retrieval data, claims data, runbooks, and dataset push.
Done when: `data:build` prints `leak check PASSED` and the four datasets are on HuggingFace.

#### Phase 2: Embedding model and reranker
Code is ready. Run `ONLY_MODELS=embedder,reranker` first. They are the cheapest, so you get a real number early.
Done when: both pass verify and beat or match their base on test.

#### Phase 3: Signal compressor
Code is ready. SFT first, then GRPO from the SFT model.
Done when: signal F1 is up against the base, JSON is valid, token ratio is low, and downstream top 1 does not drop.

#### Phase 4: Root cause classifier
Code is ready. It also runs the study: rule versus classifier, on reference signals and on compressor signals.
Done when: top 1 beats the rule baseline on test and on the held out app.

#### Phase 5: Verifier model
Code is ready. Trained on true claims and six kinds of broken ones.
Done when: catch rate beats the rule checker on wrong direction and unsupported cause, with a low false reject rate.

#### Phase 6: Serving
Code is ready in `flm/serve.py`. Ashutosh's framework calls it.
Done when: his framework runs one investigation through all endpoints.

#### Phase 6b: Public benchmark and resume line (code ready)
`bench:freeze`, `bench:run` and `bench:publish` run after the models. They freeze 500 held out incidents as `FaultLine_Bench` v1.0, score every system on it, and write the resume line with links to the results.
Done when: `BENCH.md` is public, `python -m flm.bench score` gives the same numbers, and `report/resume_line.tex` is pasted into the resume.

#### Phase 7: Real FaultLine data
When his harness writes incidents in `INCIDENT_SCHEMA.md` format, add the file to `data.jsonl` in the config, rerun with the same configs, run the 500 incident eval, and publish.
Done when: the eval page shows numbers on real FaultLine incidents.
