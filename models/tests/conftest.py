import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    from flm.common import load_cfg

    work = tmp_path_factory.mktemp("work")
    os.environ["FLM_WORK"] = str(work)
    c = load_cfg(ROOT / "configs" / "config_dry.yaml")
    c["data"]["rcaeval"]["enabled"] = False
    c["data"]["sim"]["n_per_combo"] = 3
    c["retrieval"]["code_dirs"] = []
    c["check"]["require_gpu"] = False
    c["check"]["require_hf"] = False
    c["eval"]["quick_n"] = 40
    return c


@pytest.fixture(scope="session")
def built(cfg):
    from flm import data

    data.step_build(cfg)
    data.step_compressor_data(cfg)
    data.step_retrieval(cfg)
    data.step_claims(cfg)
    return cfg
