import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flm.common import load_cfg
from flm.headline import build_headline

print(build_headline(load_cfg()) / "HEADLINE.md")
