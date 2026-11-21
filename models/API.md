# Model API (for the AI framework)

One HTTP server serves every model. The framework only calls these endpoints.

## Run it

```bash
cd FaultLine/models
export FLM_EMBEDDER=AmareshHebbar/FaultLine_Embed_bge_base
export FLM_QUERY_PREFIX="Represent this sentence for searching relevant passages: "
export FLM_RERANKER=AmareshHebbar/FaultLine_Rerank_bge_base
export FLM_CLASSIFIER=AmareshHebbar/FaultLine_RootCause_Classifier
export FLM_VERIFIER=AmareshHebbar/FaultLine_ClaimVerifier_DeBERTa_v3_base
export FLM_COMPRESSOR=AmareshHebbar/FaultLine_Compressor_Qwen3_1.7B_GRPO
export FLM_COMPRESSOR_URL=http://localhost:8001/v1
.venv/bin/vllm serve $FLM_COMPRESSOR --port 8001 &
.venv/bin/uvicorn flm.serve:app --host 0.0.0.0 --port 8000
```

Any variable left empty falls back to the rule version where one exists (classify, verify), or returns 503.

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/v1/health` | none | `{"ok": true}` |
| GET | `/v1/models` | none | which model serves each endpoint |
| POST | `/v1/compress` | `{"incident": {...}}` or `{"text": "..."}` | `{"signals": {"anomalies": [...]}, "valid": true}` |
| POST | `/v1/embed` | `{"texts": [...], "kind": "query" or "doc"}` | `{"vectors": [[...]], "dim": 768}` |
| POST | `/v1/rerank` | `{"query": "...", "docs": [...], "top_k": 10}` | `{"results": [{"index": 3, "score": 7.1}]}` |
| POST | `/v1/classify` | `{"signals": {...}, "services": [...]}` | `{"services": [...], "faults": [...], "source": "model"}` |
| POST | `/v1/verify` | `{"claims": [{"claim": "...", "evidence": "..."}]}` | `{"results": [{"supported": true, "score": 0.97}]}` |

## How the framework should use them

1. Build the incident window in the format of `INCIDENT_SCHEMA.md` and call `/v1/compress`. Use the returned signals instead of the raw window in the agent prompt.
2. Call `/v1/classify` for a fast first guess, and give it to the agent as a hint, not as the answer.
3. For RAG, embed queries with `kind: query` and documents with `kind: doc`, then call `/v1/rerank` on the top 50.
4. Before writing the report, send every sentence with the text of the evidence it cites to `/v1/verify`. Drop sentences that come back unsupported.
