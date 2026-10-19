# Fine-tune Qwen2.5-0.5B-Instruct into BabyBata 2.0 with LoRA.
# Run on a GPU (Colab T4):  python train.py        (needs: pip install peft)
#
# Same loop as always: forward -> loss -> backward -> step. What's new:
#   - LoRA: the 494M original weights stay FROZEN. Next to each weight matrix we train
#     two tiny matrices (A and B) whose product is added to it. ~1% of the weights
#     are trained, so it fits on a free GPU and forgets less of what Qwen knew.
#   - Every example includes the system prompt + profile, so the model learns to
#     READ answers from the profile instead of memorizing them.
#   - Only BabyBata's replies are graded (loss masking), never the prompt or questions.

import math
import os
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bot import BASE_MODEL, HERE, MODEL_DIR, load_all_conversations, to_messages

LORA_DIR = os.path.join(HERE, "babybata2-lora")
EPOCHS = 3
LEARNING_RATE = 2e-4      # LoRA uses higher rates than full fine-tuning: few, fresh weights
BATCH_SIZE = 2            # small, because Qwen's 151,936-word vocabulary makes the output layer huge
ACCUMULATE = 4            # ...so add up 4 batches before each step (acts like batch size 8)
MAX_TOKENS = 1536
VALIDATION_FRACTION = 0.1
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


def encode(turns, tokenizer):
    """One conversation -> (token ids, labels). Labels are -100 (ignored) everywhere
    except BabyBata's replies, so the model is only graded on what IT says."""
    ids, labels, done_text = [], [], ""
    for k in range(len(turns)):
        prompt_text = tokenizer.apply_chat_template(to_messages(turns[:k], turns[k][0]),
                                                    tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(to_messages(turns[:k + 1]), tokenize=False)
        prompt_part = tokenizer(prompt_text[len(done_text):], add_special_tokens=False)["input_ids"]
        reply_part = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
        ids += prompt_part + reply_part
        labels += [-100] * len(prompt_part) + reply_part
        done_text = full_text
    return ids[:MAX_TOKENS], labels[:MAX_TOKENS]


def batches(examples, pad_id, shuffle):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), BATCH_SIZE):
        chunk = [examples[i] for i in order[s:s + BATCH_SIZE]]
        longest = max(len(ids) for ids, _ in chunk)
        ids = [i + [pad_id] * (longest - len(i)) for i, _ in chunk]
        labels = [l + [-100] * (longest - len(l)) for _, l in chunk]
        attention = [[1] * len(i) + [0] * (longest - len(i)) for i, _ in chunk]
        yield (torch.tensor(ids, device=device), torch.tensor(labels, device=device),
               torch.tensor(attention, device=device))


def compute_loss(model, ids, attention, labels):
    """Same loss as model(..., labels=labels).loss, but memory-friendly.

    The normal way scores all 151,936 vocabulary words at EVERY position (~1,100 per
    example, mostly the profile), then throws away all but the ~40 reply positions.
    That output alone is ~1.3 GB per batch and ran the T4 out of memory. Here we run
    the transformer, keep only the reply positions, and score just those."""
    core = model.get_base_model()                 # Qwen with the LoRA layers inside it
    hidden = core.model(input_ids=ids, attention_mask=attention).last_hidden_state
    targets = labels[:, 1:]                       # position t predicts token t+1
    graded = targets != -100                      # only BabyBata's reply tokens
    logits = core.lm_head(hidden[:, :-1][graded]).float()
    return torch.nn.functional.cross_entropy(logits, targets[graded])


def validation_loss(model, examples, pad_id):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for ids, labels, attention in batches(examples, pad_id, shuffle=False):
            total += compute_loss(model, ids, attention, labels).item() * len(ids)
            count += len(ids)
    model.train()
    return total / count


def main():
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU. This will be extremely slow; use Colab.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"training {trainable:,} of {total:,} weights ({100 * trainable / total:.1f}%)")

    conversations = load_all_conversations()
    random.Random(SEED).shuffle(conversations)
    n_val = max(1, int(len(conversations) * VALIDATION_FRACTION))
    val = [encode(c, tokenizer) for c in conversations[:n_val]]
    train = [encode(c, tokenizer) for c in conversations[n_val:]]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"{len(train)} training conversations, {len(val)} validation, "
          f"~{sum(len(i) for i, _ in train) // len(train)} tokens each (mostly the profile)\n")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    steps_total = EPOCHS * math.ceil(len(train) / (BATCH_SIZE * ACCUMULATE))
    # learning rate: short warm-up, then smoothly down to 0 (big steps early, fine steps late)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, steps_total) / steps_total)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")  # keeps float16 gradients from underflowing

    best = validation_loss(model, val, pad_id)
    print(f"before training: validation loss = {best:.3f}")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        start, running, seen = time.time(), 0.0, 0
        for n, (ids, labels, attention) in enumerate(batches(train, pad_id, shuffle=True), 1):
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = compute_loss(model, ids, attention, labels)                          # forward + loss
            scaler.scale(loss / ACCUMULATE).backward()                                       # backward
            if n % ACCUMULATE == 0:
                scaler.step(optimizer)                                                       # step
                scaler.update()
                optimizer.zero_grad()
                schedule.step()
            running += loss.item() * len(ids)
            seen += len(ids)
        val_loss = validation_loss(model, val, pad_id)
        note = ""
        if val_loss < best:
            best = val_loss
            model.save_pretrained(LORA_DIR)   # only the small LoRA matrices (a few MB)
            note = "  <- best so far, saved"
        print(f"epoch {epoch}: train loss = {running / seen:.3f}   validation loss = {val_loss:.3f}   "
              f"({time.time() - start:.0f}s){note}")

    if not os.path.exists(LORA_DIR):
        print("validation never improved; saving the last epoch instead")
        model.save_pretrained(LORA_DIR)

    # Merge the best LoRA matrices into Qwen's weights -> one normal model folder.
    print("\nmerging the best LoRA weights into the model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
    merged.to(torch.float16).save_pretrained(MODEL_DIR)   # float16: half the file size
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"BabyBata 2.0 saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
