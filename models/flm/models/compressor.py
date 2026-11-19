import json
import time

from ..common import log, read_jsonl, set_seed, wdir, wpath, write_jsonl
from ..metrics import summarize
from ..schema import SIGNALS
from ..signals import rule_rank, signal_f1
from ..train_utils import bf16_ok, common_args, device, save_history

KEYS = ["signal_f1", "valid_json", "token_ratio", "downstream_top1"]

SYSTEM = (
    "You read a telemetry window from a microservice system and write the incident signals as compact JSON. "
    "Output only JSON of the form {\"anomalies\":[{\"service\":str,\"signal\":str,\"metric\":str,\"before\":num,\"after\":num,\"severity\":num}]} "
    "with at most {k} entries, highest severity first. Allowed signal values: " + ", ".join(SIGNALS + ["other"]) + ". "
    "Only report changes you can see in the window. Do not name a root cause."
)


def system_prompt(cfg):
    return SYSTEM.replace("{k}", str(cfg["signals"]["top_k"]))


def build_prompt(tok, cfg, text):
    msgs = [{"role": "system", "content": system_prompt(cfg)}, {"role": "user", "content": text}]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt(cfg)}\n\n{text}\n\nJSON:"


def parse(text):
    if not text:
        return None
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("anomalies"), list):
        return None
    return obj


def rows(cfg, split, n=None):
    p = wpath(cfg, "data", "compressor", f"{split}.jsonl")
    r = read_jsonl(p) if p.exists() else []
    return r[:n] if n else r


def load_tok(ref):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(ref):
    import torch
    from transformers import AutoModelForCausalLM

    dt = torch.bfloat16 if bf16_ok() else torch.float32
    return AutoModelForCausalLM.from_pretrained(ref, dtype=dt).to(device())


def generate(cfg, name, ref, prompts):
    m = cfg["models"][name]
    max_new = int(m.get("max_new_tokens", 512))
    if m.get("use_vllm"):
        try:
            from vllm import LLM, SamplingParams

            llm = LLM(model=ref, max_model_len=int(m.get("max_length", 4096)) + max_new, gpu_memory_utilization=float(m.get("vllm_gpu_util", 0.85)), seed=cfg.get("seed", 0))
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new))
            return [o.outputs[0].text for o in outs]
        except ImportError:
            log("vllm not installed, using transformers generate")
    import torch

    tok = load_tok(ref)
    tok.padding_side = "left"
    model = load_model(ref)
    model.eval()
    out = []
    bs = int(m.get("eval_batch", 8))
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i : i + bs], return_tensors="pt", padding=True, truncation=True, max_length=int(m.get("max_length", 4096)), add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True))
    return out


def evaluate(cfg, name, ref, split, n=None):
    data = rows(cfg, split, n)
    if not data:
        return None
    from ..data import load_split

    incs = {i["incident_id"]: i for i in load_split(cfg, split)}
    tok = load_tok(ref)
    m = cfg["models"][name]
    data = fit_rows(cfg, tok, data, int(m.get("max_length", 4096)), reserve_for_completion=int(m.get("max_new_tokens", 512)))
    prompts = [r["prompt"] for r in data]
    t = time.time()
    texts = generate(cfg, name, ref, prompts)
    ms = (time.time() - t) * 1000 / len(data)
    items, samples = [], []
    for r, p, txt in zip(data, prompts, texts):
        pred = parse(txt)
        ref_sig = json.loads(r["target"])
        inc = incs[r["id"]]
        n_in = len(tok(r["input"], add_special_tokens=False)["input_ids"])
        n_out = len(tok(txt, add_special_tokens=False)["input_ids"])
        top = rule_rank(pred, inc["services"])[0] if pred else None
        items.append({"id": r["id"], "signal_f1": signal_f1(pred, ref_sig), "valid_json": 1.0 if pred else 0.0, "token_ratio": n_out / max(1, n_in), "downstream_top1": 1.0 if top == inc["label"]["service"] else 0.0})
        samples.append({"id": r["id"], "output": txt, "pred": pred})
    write_jsonl(wpath(cfg, "results", f"{name}_{split}_samples.jsonl"), samples)
    s = summarize(items, KEYS, seed=cfg.get("seed", 0))
    s["ms_per_item"] = round(ms, 1)
    return {"items": items, "summary": s}


def lora_config(m):
    from peft import LoraConfig

    return LoraConfig(r=int(m.get("lora_r", 16)), lora_alpha=int(m.get("lora_alpha", 32)), lora_dropout=float(m.get("lora_dropout", 0.05)), target_modules=m.get("lora_targets", "all-linear"), task_type="CAUSAL_LM")


def _finish(cfg, name, trainer, tok, m):
    out = wdir(cfg, "models", name)
    adapter = wdir(cfg, "models", f"{name}_adapter")
    model = trainer.model
    if m.get("lora", True) and hasattr(model, "merge_and_unload"):
        model.save_pretrained(str(adapter))
        model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    save_history(trainer, wdir(cfg, "models", f"{name}_training"))
    return {"local": str(out), "adapter": str(adapter) if m.get("lora", True) else None, "training": str(wdir(cfg, "models", f"{name}_training"))}


def init_ref(cfg, name):
    m = cfg["models"][name]
    src = m.get("init_from")
    if not src:
        return m["base"]
    local = wpath(cfg, "models", src, "config.json")
    if local.exists():
        return str(local.parent)
    from ..common import repo_id

    return repo_id(cfg, src)


def fit_rows(cfg, tok, data, budget, reserve_for_completion=None):
    out, shrunk, dropped = [], 0, 0
    for r in data:
        text = r["input"]
        comp = len(tok(r["target"], add_special_tokens=False)["input_ids"]) + 1 if reserve_for_completion is None else reserve_for_completion
        for _ in range(6):
            p = build_prompt(tok, cfg, text)
            n = len(tok(p, add_special_tokens=False)["input_ids"])
            if n + comp <= budget:
                break
            text = text[: int(len(text) * (budget - comp) / max(n, 1) * 0.95)]
        else:
            dropped += 1
            continue
        if text is not r["input"]:
            shrunk += 1
        out.append({"id": r.get("id"), "input": r["input"], "prompt": p, "target": r["target"]})
    log(f"fit to {budget} tokens: {len(out)} kept, {shrunk} inputs shortened, {dropped} dropped")
    if dropped > 0.05 * max(1, len(data)):
        raise SystemExit(f"{dropped} of {len(data)} rows do not fit max_length {budget}; raise max_length or lower max_input_chars")
    return out


def train_sft(cfg, name):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    m = cfg["models"][name]
    set_seed(cfg.get("seed", 0))
    ref = init_ref(cfg, name)
    tok = load_tok(ref)
    data = [{"prompt": r["prompt"], "completion": r["target"]} for r in fit_rows(cfg, tok, rows(cfg, "train"), int(m.get("max_length", 4096)))]
    args = SFTConfig(**common_args(m, wdir(cfg, "runs", name), cfg.get("seed", 0)), max_length=int(m.get("max_length", 4096)), completion_only_loss=True, packing=False, gradient_checkpointing=bool(m.get("gradient_checkpointing", True)))
    trainer = SFTTrainer(model=load_model(ref), args=args, train_dataset=Dataset.from_list(data), processing_class=tok, peft_config=lora_config(m) if m.get("lora", True) else None)
    trainer.train()
    return _finish(cfg, name, trainer, tok, m)


def reward_valid(completions, **kw):
    return [1.0 if parse(c) is not None else 0.0 for c in completions]


def reward_f1(completions, reference, **kw):
    return [signal_f1(parse(c), json.loads(r)) if parse(c) is not None else 0.0 for c, r in zip(completions, reference)]


def reward_length(completions, reference, **kw):
    return [-min(1.0, max(0.0, len(c) / max(1, len(r)) - 1.2)) for c, r in zip(completions, reference)]


def train_grpo(cfg, name):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    m = cfg["models"][name]
    set_seed(cfg.get("seed", 0))
    ref = init_ref(cfg, name)
    tok = load_tok(ref)
    tok.padding_side = "left"
    fitted = fit_rows(cfg, tok, rows(cfg, "train", m.get("max_prompts")), int(m.get("max_length", 4096)), reserve_for_completion=int(m.get("max_new_tokens", 512)))
    data = [{"prompt": r["prompt"], "reference": r["target"]} for r in fitted]
    kw = common_args(m, wdir(cfg, "runs", name), cfg.get("seed", 0))
    args = GRPOConfig(
        **kw,
        num_generations=int(m.get("num_generations", 4)),
        max_completion_length=int(m.get("max_new_tokens", 512)),
        beta=float(m.get("beta", 0.04)),
        temperature=float(m.get("temperature", 0.8)),
        use_vllm=bool(m.get("train_vllm", False)),
        vllm_mode=m.get("vllm_mode", "colocate"),
        reward_weights=[float(x) for x in m.get("reward_weights", [0.3, 1.0, 0.3])],
        gradient_checkpointing=bool(m.get("gradient_checkpointing", True)),
    )
    trainer = GRPOTrainer(model=load_model(ref), reward_funcs=[reward_valid, reward_f1, reward_length], args=args, train_dataset=Dataset.from_list(data), processing_class=tok, peft_config=lora_config(m) if m.get("lora", True) else None)
    trainer.train()
    return _finish(cfg, name, trainer, tok, m)


def train(cfg, name):
    return train_grpo(cfg, name) if cfg["models"][name].get("method_kind") == "grpo" else train_sft(cfg, name)
