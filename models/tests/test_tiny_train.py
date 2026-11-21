import json

import pytest

from flm.common import read_jsonl, wpath

pytestmark = pytest.mark.slow


def _texts(cfg):
    out = [d["text"] for d in read_jsonl(wpath(cfg, "data", "retrieval", "corpus.jsonl"))]
    out += [r["input"][:2000] + r["target"] for r in read_jsonl(wpath(cfg, "data", "compressor", "train.jsonl"))[:40]]
    out += [r["claim"] + r["evidence"] for r in read_jsonl(wpath(cfg, "data", "claims", "train.jsonl"))[:200]]
    return out


@pytest.fixture(scope="module")
def tiny(built, tmp_path_factory):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, BertModel, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    texts = _texts(built)
    wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    wp.normalizer = normalizers.BertNormalizer(lowercase=True)
    wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    wp.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    from tokenizers.processors import TemplateProcessing

    wp.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", wp.token_to_id("[CLS]")), ("[SEP]", wp.token_to_id("[SEP]"))])
    btok = PreTrainedTokenizerFast(tokenizer_object=wp, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=256)
    bc = BertConfig(vocab_size=wp.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
    enc, ce = d / "bert", d / "bert_ce"
    BertModel(bc).save_pretrained(enc)
    btok.save_pretrained(enc)
    BertForSequenceClassification(BertConfig(**{**bc.to_dict(), "num_labels": 1})).save_pretrained(ce)
    btok.save_pretrained(ce)
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ctok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>", model_max_length=4096)
    ctok.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    lm = d / "lm"
    qc = Qwen2Config(vocab_size=bpe.get_vocab_size(), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, eos_token_id=ctok.eos_token_id, pad_token_id=ctok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(qc).save_pretrained(lm)
    ctok.save_pretrained(lm)
    c = json.loads(json.dumps(built))
    common = {"max_steps": 3, "batch_size": 4, "logging_steps": 1, "epochs": 1}
    c["models"]["embedder"].update(common, base=str(enc), baselines=[], query_prefix="", doc_prefix="", mini_batch_size=4, max_seq_length=128)
    c["models"]["reranker"].update(common, base=str(ce), candidates=10, max_length=128)
    c["models"]["verifier"].update(common, base=str(ce), max_length=128)
    lmc = {"base": str(lm), "max_steps": 2, "batch_size": 2, "grad_accum": 1, "max_length": 768, "max_new_tokens": 24, "eval_batch": 4, "use_vllm": False, "lora_targets": "all-linear", "gradient_checkpointing": False, "logging_steps": 1}
    c["models"]["compressor_sft"].update(lmc)
    c["models"]["compressor_grpo"].update(lmc, num_generations=2, max_prompts=4, batch_size=2)
    for split in ("train", "val", "test"):
        p = wpath(c, "data", "compressor", f"{split}.jsonl")
        rows = read_jsonl(p)
        from flm.common import write_jsonl

        write_jsonl(p, [{**r, "input": r["input"][:1200]} for r in rows])
    return c


def test_embedder_trains_and_evals(tiny):
    from flm.models import embedder

    out = embedder.train(tiny, "embedder")
    r = embedder.evaluate(tiny, "embedder", out["local"], "val", 10)
    assert r["summary"]["n"] == 10 and 0 <= r["summary"]["ndcg@10"] <= 1


def test_reranker_trains_and_evals(tiny):
    from flm.models import reranker

    out = reranker.train(tiny, "reranker")
    a = reranker.evaluate(tiny, "reranker", out["local"], "val", 10)
    b = reranker.evaluate(tiny, "reranker", "first_stage", "val", 10)
    assert a["summary"]["candidate_recall"] == b["summary"]["candidate_recall"]


def test_verifier_trains_and_evals(tiny):
    from flm.models import verifier

    out = verifier.train(tiny, "verifier")
    r = verifier.evaluate(tiny, "verifier", out["local"], "val", 50)
    assert r["summary"]["n"] == 50


def test_compressor_sft_then_grpo(tiny):
    from flm.models import compressor

    out = compressor.train(tiny, "compressor_sft")
    r = compressor.evaluate(tiny, "compressor_sft", out["local"], "val", 4)
    assert r["summary"]["n"] == 4
    assert compressor.init_ref(tiny, "compressor_grpo") == out["local"]
    g = compressor.train(tiny, "compressor_grpo")
    assert (wpath(tiny, "models", "compressor_grpo", "config.json")).exists()
    assert g["adapter"]


def test_pipeline_gate_with_local_models(tiny):
    from flm.gates import relative_gate
    from flm.models import classifier

    out = classifier.train(tiny, "classifier")
    m = classifier.evaluate(tiny, "classifier", out["local"], "val")
    b = classifier.evaluate(tiny, "classifier", "rule", "val")
    g = relative_gate(tiny, "classifier", m["items"], b["items"])
    assert "delta_ci95" in g and g["n"] == len(m["items"])


def test_full_model_steps_with_fake_hub(tiny, monkeypatch):
    import shutil

    from flm import hub, pipeline
    from flm.common import read_json, results_path

    pushed = []
    monkeypatch.setattr(hub, "push_folder", lambda repo, folder, **kw: pushed.append((repo, kw.get("path_in_repo"))))
    monkeypatch.setattr(hub, "push_text", lambda repo, path, text, **kw: pushed.append((repo, path)))
    monkeypatch.setattr(hub, "add_to_collection", lambda cfg, repos: None)

    def fake_download(cfg, name, dest):
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(read_json(results_path(cfg, name, "train"))["local"], dest)
        return dest, ["config.json"]

    monkeypatch.setattr(hub, "download_model", fake_download)
    tiny["eval"]["gate_drop"] = 1.0
    for name in ("classifier", "verifier", "embedder"):
        for stage in pipeline.MODEL_STEPS:
            pipeline.MODEL_STEPS[stage](tiny, name)
        assert read_json(results_path(tiny, name, "verify"))["passed"]
        assert read_json(results_path(tiny, name, "full"))["splits"]
    assert any(p[1] == "README.md" for p in pushed)
    from flm.headline import build_headline

    out = build_headline(tiny)
    text = (out / "HEADLINE.md").read_text()
    assert "classifier" in text and "Leak check: PASSED" in text
