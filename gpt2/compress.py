# Shrink BabyBata and measure what each cut costs.
# Run AFTER train.py (needs babybata-model/):  python compress.py
#
#   1. Quantization (written by hand): store each weight with fewer bits
#   2. Layer removal: keep 6 of GPT-2's 12 transformer layers, then retrain briefly
#   3. Both together
#
# Every version is scored on the same validation set (lower loss = better)
# and answers the same questions, so you can see the trade-off.

import copy
import os

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from train import OUT_DIR, device, fine_tune, prepare_data, prompt_for, validation_loss

SMALL_DIR = OUT_DIR + "-small"
KEEP_LAYERS = [0, 2, 4, 6, 8, 10]   # every other layer, out of 0..11
RETRAIN_EPOCHS = 6
RETRAIN_LR = 1e-4                   # a bit higher: the cut model has more to repair
TEST_QUESTIONS = ["Who are you?", "What projects has Batbold built?", "How can I contact Batbold?"]


# ---------------------------------------------------------------
# Quantization by hand
#   For each weight matrix, row by row:
#     scale   = biggest |weight| in the row / largest integer we can store
#     stored  = round(weight / scale)        <- small integer, e.g. -127..127
#     used    = stored * scale               <- what the network computes with
#   Rounding error is the only "damage". We simulate it in float so we can
#   measure quality; a real deployment would store the integers + scales.
# ---------------------------------------------------------------
def quantize(model, bits):
    q = copy.deepcopy(model)
    top = 2 ** (bits - 1) - 1                    # 8 bits -> 127, 4 bits -> 7
    with torch.no_grad():
        for p in q.parameters():
            if p.dim() != 2:                     # biases, LayerNorm: tiny, keep exact
                continue
            scale = p.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / top
            p.copy_(torch.clamp(torch.round(p / scale), -top - 1, top) * scale)
    return q


def size_mb(model, bits=32):
    """File size if 2D weights use `bits` each (+ one 16-bit scale per row)
    and everything else stays 32-bit."""
    total_bits = 0
    for p in model.parameters():
        if p.dim() == 2 and bits < 32:
            total_bits += p.numel() * bits + p.shape[0] * 16
        else:
            total_bits += p.numel() * 32
    return total_bits / 8 / 1e6


def answer(model, tokenizer, question):
    model.eval()  # validation_loss() leaves the model in training mode (dropout on)
    ids =tokenizer.encode(prompt_for(question), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=50,
                             do_sample=False,  # greedy: same input -> same output, fair comparison
                             pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return text.split("\nUser:")[0].strip()


def main():
    if not os.path.exists(OUT_DIR):
        raise SystemExit("No BabyBata yet. Run train.py first.")

    tokenizer = GPT2TokenizerFast.from_pretrained(OUT_DIR)
    full = GPT2LMHeadModel.from_pretrained(OUT_DIR).to(device).eval()
    train, val, pad_id = prepare_data(tokenizer)

    # ---- layer removal: cut, then let the remaining layers adapt ----
    print(f"Removing layers: keeping {len(KEEP_LAYERS)} of {full.config.n_layer} -> {KEEP_LAYERS}")
    small = copy.deepcopy(full)
    small.transformer.h = torch.nn.ModuleList([small.transformer.h[i] for i in KEEP_LAYERS])
    small.config.n_layer = len(KEEP_LAYERS)
    for new_index, block in enumerate(small.transformer.h):  # kept layers were 0,2,4.. -> now 0,1,2..
        if hasattr(block.attn, "layer_idx"):
            block.attn.layer_idx = new_index
    small.train()
    print(f"right after cutting: validation loss = {validation_loss(small, val, pad_id):.3f}  (damaged!)")
    print("retraining the smaller model...")
    fine_tune(small, tokenizer, train, val, pad_id, RETRAIN_EPOCHS, RETRAIN_LR, SMALL_DIR)
    small = GPT2LMHeadModel.from_pretrained(SMALL_DIR).to(device).eval()  # best checkpoint

    variants = [
        ("full, 32-bit (original)", full, 32),
        ("full, 8-bit", quantize(full, 8), 8),
        ("full, 4-bit", quantize(full, 4), 4),
        ("6 layers, 32-bit", small, 32),
        ("6 layers, 8-bit", quantize(small, 8), 8),
        ("6 layers, 4-bit", quantize(small, 4), 4),
    ]

    print("\n" + "=" * 72)
    print(f"{'version':26} {'parameters':>12} {'size':>9} {'val loss':>9}")
    print("-" * 72)
    for name, model, bits in variants:
        params = sum(p.numel() for p in model.parameters())
        loss = validation_loss(model, val, pad_id)
        print(f"{name:26} {params / 1e6:10.1f}M {size_mb(model, bits):7.0f}MB {loss:9.3f}")
    print("=" * 72)

    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        for name, model, _ in variants:
            print(f"  {name:26} {answer(model, tokenizer, question)}")

    print(f"\nSaved the 6-layer model to {SMALL_DIR}")


if __name__ == "__main__":
    main()
