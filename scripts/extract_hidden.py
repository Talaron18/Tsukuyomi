#!/usr/bin/env python3
"""Hidden states for the action head: flow.train.hidden. From the repo root:

    python scripts/extract_hidden.py --export data/datagen/r1/export/action \\
        --data data/datagen/r1 --model Qwen/Qwen3-8B \\
        --adapter runs/sft/q8/adapter --layers -8 --out runs/hidden/q8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.train.hidden import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
