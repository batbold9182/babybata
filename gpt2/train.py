# Fine-tune GPT-2 into BabyBata.
# Run:  python train.py        (works on a laptop CPU, much faster on a GPU / Google Colab)
#
# It's the same loop as your from-scratch network:
#   forward -> loss -> backward -> step
# except the starting weights come from GPT-2 instead of random numbers.

import os
import random
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "babybata.txt")        # hand-written + teach mode
GENERATED_FILE = os.path.join(HERE, "data", "generated.txt")  # made by augment.py
OUT_DIR = os.path.join(HERE, "models", "babybata-model")   # older versions: models/archive/

# Where training STARTS from. Plain GPT-2 = a clean start, no old habits
# (v1 had memorized a wrong email). To continue from a saved version instead,
# point this at its folder, e.g. os.path.join(HERE, "babybata-model-v1").
BASE_MODEL = "gpt2"
EPOCHS = 6               # best validation was at epoch 4; later epochs only overfit
FREEZE_EMBEDDINGS = True  # don't train the 38.6M-weight vocabulary table: ~faster, less memory
LEARNING_RATE = 5e-5     # tiny, so we adjust GPT-2's knowledge instead of overwriting it
BATCH_SIZE = 4
MAX_TOKENS = 128
VALIDATION_FRACTION = 0.15
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# 1. DATA: read User/BabyBata pairs from the text file
# ---------------------------------------------------------------
def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l and not l.startswith("#")]
        if len(lines) == 2 and lines[0].startswith("User:") and lines[1].startswith("BabyBata:"):
            question = lines[0][len("User:"):].strip()
            answer = lines[1][len("BabyBata:"):].strip()
            pairs.append((question, answer))
    return pairs


def prompt_for(question):
    """The exact format the model sees. chat.py must use the same one."""
    return f"User: {question}\nBabyBata:"


# ---------------------------------------------------------------
# 2. TOKENIZE, with loss masking:
#    the model is only graded on BabyBata's answer, not on the question.
#    Label -100 means "ignore this position in the loss".
# ---------------------------------------------------------------
def encode(pair, tokenizer):
    question, answer = pair
    prompt_ids = tokenizer.encode(prompt_for(question))
    answer_ids = tokenizer.encode(" " + answer) + [tokenizer.eos_token_id]  # eos = "stop talking"
    ids = (prompt_ids + answer_ids)[:MAX_TOKENS]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_TOKENS]
    return ids, labels


def make_batch(examples, pad_id):
    """Pad every example in the batch to the same length."""
    longest = max(len(ids) for ids, _ in examples)
    input_ids, labels, attention = [], [], []
    for ids, labs in examples:
        pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)         # padding is never graded
        attention.append([1] * len(ids) + [0] * pad)  # and never looked at
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device),
            torch.tensor(attention, device=device))


def batches(examples, pad_id, shuffle):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), BATCH_SIZE):
        yield make_batch([examples[i] for i in order[s:s + BATCH_SIZE]], pad_id)


def validation_loss(model, examples, pad_id):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ids, labels, attn in batches(examples, pad_id, shuffle=False):
            total += model(input_ids=ids, attention_mask=attn, labels=labels).loss.item() * len(ids)
            count += len(ids)
    model.train()
    return total / count


def prepare_data(tokenizer):
    """Same train/validation split every time (compress.py relies on this)."""
    pairs = load_pairs(DATA_FILE)
    if os.path.exists(GENERATED_FILE):
        pairs += load_pairs(GENERATED_FILE)
    random.Random(SEED).shuffle(pairs)
    n_val = max(1, int(len(pairs) * VALIDATION_FRACTION))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    train = [encode(p, tokenizer) for p in train_pairs]
    val = [encode(p, tokenizer) for p in val_pairs]
    pad_id = tokenizer.eos_token_id  # GPT-2 has no pad token; masked out anyway
    return train, val, pad_id


def fine_tune(model, tokenizer, train, val, pad_id, epochs, learning_rate, out_dir):
    """The training loop. Saves to out_dir whenever validation loss improves."""
    # AdamW = gradient descent with momentum and per-weight step sizes (lesson 9 material)
    # Only weights that are being trained (not frozen) go to the optimizer.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    best = validation_loss(model, val, pad_id)
    print(f"before training: validation loss = {best:.3f}")

    for epoch in range(1, epochs + 1):
        start, train_total, seen = time.time(), 0.0, 0
        for ids, labels, attn in batches(train, pad_id, shuffle=True):
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss  # forward + loss
            loss.backward()          # backward: PyTorch does the chain rule for you
            optimizer.step()         # step: move every weight downhill
            optimizer.zero_grad()    # reset gradients for the next batch
            train_total += loss.item() * len(ids)
            seen += len(ids)

        val_loss = validation_loss(model, val, pad_id)
        note = ""
        if val_loss < best:
            best = val_loss
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            note = "  <- best so far, saved"
        print(f"epoch {epoch}: train loss = {train_total / seen:.3f}   "
              f"validation loss = {val_loss:.3f}   ({time.time() - start:.0f}s){note}")
    return best


def main():
    print(f"device: {device}")
    tokenizer = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
    print(f"loaded {BASE_MODEL}: {sum(p.numel() for p in model.parameters()):,} parameters")

    if FREEZE_EMBEDDINGS:
        # The word table (also used as the output layer) and position table stay as they are.
        model.transformer.wte.weight.requires_grad_(False)
        model.transformer.wpe.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {trainable:,} of them (the rest are frozen)")

    train, val, pad_id = prepare_data(tokenizer)
    print(f"{len(train)} training examples, {len(val)} validation examples\n")

    fine_tune(model, tokenizer, train, val, pad_id, EPOCHS, LEARNING_RATE, OUT_DIR)

    if not os.path.exists(OUT_DIR):
        print("\nValidation never improved, so nothing was saved. Try more data or more epochs.")
    else:
        print(f"\nBabyBata saved to {OUT_DIR}.  Talk to it:  python chat.py")


if __name__ == "__main__":
    main()
