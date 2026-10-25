"""Stage 2 training (docs/embodiment.md, "Stage 2 on one 3090 Ti"): the chat
templates shared by training and serving (chat), the LoRA distillation
trainer (sft), its evaluation (evalsft) and the model loader (hf). Only
chat imports nothing heavy; the rest import torch / transformers / peft
lazily, inside the functions that need them."""
