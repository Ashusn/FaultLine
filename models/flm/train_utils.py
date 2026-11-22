import os
from pathlib import Path

from .common import write_json


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def bf16_ok():
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def common_args(m, out_dir, seed):
    cpu = device() == "cpu"
    return {
        "output_dir": str(out_dir),
        "num_train_epochs": float(m.get("epochs", 1)),
        "max_steps": int(m.get("max_steps", -1)),
        "per_device_train_batch_size": int(m.get("batch_size", 8)),
        "gradient_accumulation_steps": int(m.get("grad_accum", 1)),
        "learning_rate": float(m.get("lr", 2e-5)),
        "warmup_steps": int(m.get("warmup_steps", 0)),
        "lr_scheduler_type": m.get("scheduler", "linear"),
        "logging_steps": int(m.get("logging_steps", 10)),
        "save_strategy": "no",
        "report_to": "none",
        "seed": int(seed),
        "bf16": (not cpu) and bf16_ok(),
        "use_cpu": cpu,
        "dataloader_num_workers": 0,
    }


def save_history(trainer, out_dir):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    hist = list(trainer.state.log_history)
    write_json(d / "log_history.json", hist)
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt

        for key in ("loss", "reward"):
            pts = [(h["step"], h[key]) for h in hist if key in h and "step" in h]
            if pts:
                plt.figure(figsize=(6, 3.5))
                plt.plot([p[0] for p in pts], [p[1] for p in pts])
                plt.xlabel("step")
                plt.ylabel(key)
                plt.tight_layout()
                plt.savefig(d / f"{key}.png", dpi=120)
                plt.close()
    except ImportError:
        pass
    return d
