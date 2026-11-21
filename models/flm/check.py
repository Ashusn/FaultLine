import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import log, wpath


def _sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def gpu_info():
    import torch

    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability()
    return {"name": torch.cuda.get_device_name(), "capability": f"{cap[0]}.{cap[1]}", "major": cap[0], "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}


def real_batch_test(cfg):
    import torch
    from transformers import AutoModelForCausalLM

    names = [n for n, m in cfg["models"].items() if m.get("type") == "compressor" and m.get("enabled", True) and not m.get("init_from")]
    if not names:
        return
    m = cfg["models"][names[0]]
    model = AutoModelForCausalLM.from_pretrained(m["base"], dtype=torch.bfloat16).cuda()
    from peft import get_peft_model

    from .models.compressor import lora_config

    model = get_peft_model(model, lora_config(m))
    if m.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    bs, L = int(m.get("batch_size", 1)), int(m.get("max_length", 4096))
    x = torch.randint(100, 1000, (bs, L), device="cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(2):
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.time() - t) / 2
    log(f"real batch test {m['base']}: batch {bs} x {L} tokens, {dt:.2f}s per step, {bs * L / dt:.0f} tokens/s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    del model, opt
    torch.cuda.empty_cache()


def run(cfg):
    c = cfg["check"]
    problems = []
    v = sys.version_info
    log("python", f"{v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) != tuple(c.get("python", [3, 11])):
        problems.append(f"python {v.major}.{v.minor}, expected {c.get('python', [3, 11])}")
    if cfg["hub"]["prefix"].startswith("DRY_") != _is_dry_config(cfg):
        problems.append("dry config must use a DRY_ prefix and real configs must not")
    import torch

    log("torch", torch.__version__, "built for cuda", torch.version.cuda)
    log("driver", _sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader") or "none")
    g = gpu_info()
    log("gpu", g)
    if c.get("require_gpu", True):
        if not g:
            problems.append("no CUDA GPU")
        elif g["major"] < 8:
            problems.append(f"GPU capability {g['capability']} is below 8.0, bf16 will not work")
        apps = _sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
        if apps and "No devices" not in apps and any(ch.isdigit() for ch in apps):
            problems.append(f"GPU is not empty: {apps}")
    work = wpath(cfg, "x").parent
    free = shutil.disk_usage(work).free / 1e9
    log("free disk GB", round(free, 1))
    if free < float(c.get("min_free_gb", 20)):
        problems.append(f"only {free:.0f} GB free at {work}")
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi

            log("hf user", HfApi().whoami()["name"])
        except Exception as e:
            problems.append(f"HF token failed: {e}")
    elif c.get("require_hf", True):
        problems.append("HF_TOKEN is not set")
    if problems:
        for p in problems:
            log("CHECK PROBLEM:", p)
        raise SystemExit("check FAILED")
    if g and c.get("real_batch", True):
        real_batch_test(cfg)
    log("everything works")


def _is_dry_config(cfg):
    return Path(cfg["_path"]).stem.endswith("_dry")
