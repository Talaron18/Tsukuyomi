#!/usr/bin/env python3
"""The LoRA distillation trainer: flow.train.sft (stats, train, merge, gguf,
modelfile). Run from the repo root:

    python scripts/train_sft.py stats --data data/datagen/r1
    python scripts/train_sft.py train --data data/datagen/r1 --out runs/sft/q8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.train.sft import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
