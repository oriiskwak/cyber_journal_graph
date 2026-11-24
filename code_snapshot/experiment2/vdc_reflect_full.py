"""Seed-42 full-data Qwen CYREN guarded-reflection entry point."""
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["VDC_FULL_RUN"] = "1"
os.environ["VDC_BACKEND"] = "qwen"
os.environ.setdefault("VDC_FORCE_REFLECT", "0")

from experiment_intern.vdc_reflect_smoke import main


if __name__ == "__main__":
    main()
