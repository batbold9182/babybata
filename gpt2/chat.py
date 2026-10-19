# Talk to BabyBata (run train.py first).
# Run:  python chat.py             <- BabyBata (babybata-model/)
#       python chat.py --teach     <- teach mode: correct its answers, saved as training data
#       python chat.py --small     <- the 6-layer version (babybata-model-small/)
#       python chat.py --base      <- plain GPT-2 with the same prompt, to compare
#
# Chatting never changes the model (only the forward pass runs).
# Teach mode just COLLECTS corrections into data/babybata.txt;
# BabyBata learns them the next time you run train.py.

import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.utils import logging

from facts import fill, to_slots
from train import DATA_FILE, OUT_DIR, prompt_for

logging.set_verbosity_error()  # hide download/loading chatter

TEACH_HEADER = "# ---------- Taught in chat (python chat.py --teach) ----------"

if "--base" in sys.argv:
    model_path, name = "gpt2", "plain GPT-2"
elif "--small" in sys.argv:
    model_path, name = OUT_DIR + "-small", "BabyBata (6 layers)"
else:
    model_path, name = OUT_DIR, "BabyBata"
if model_path != "gpt2" and not os.path.exists(os.path.join(model_path, "config.json")):
    sys.exit(f"Model not found in {model_path}\n"
             "Download it from Google Drive (MyDrive/babybata/) and put it in models/.")
teach = "--teach" in sys.argv

tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()


def generate(question):
    """Returns the raw answer, slots still in it (e.g. 'Email me at {EMAIL}')."""
    ids = tokenizer.encode(prompt_for(question), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=80,
            do_sample=True,
            temperature=0.3,         # low = sticks to what it learned, high = more creative (and more made-up facts)
            top_p=0.9,               # only pick from the most likely tokens
            repetition_penalty=1.05, # tiny: names like "Batbold" MUST repeat exactly
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return reply.split("\nUser:")[0].strip()  # in case it starts writing the next question itself


def count_examples():
    with open(DATA_FILE, encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith("User:"))


def save_example(question, answer):
    """Append one conversation to the training data, exact facts turned back into slots."""
    with open(DATA_FILE, encoding="utf-8") as f:
        has_header = TEACH_HEADER in f.read()
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        if not has_header:
            f.write(f"\n\n{TEACH_HEADER}")
        f.write(f"\n\nUser: {question}\nBabyBata: {to_slots(answer)}")


print(f"Talking to {name} (type 'quit' to exit)")
if teach:
    print("TEACH MODE: after each answer ->  Enter = good, save it  |  type a better answer = save that"
          "  |  s = skip, don't save")
print()

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() == "quit":
        break

    answer = fill(generate(question))
    print(f"{name}: {answer}")

    if teach:
        feedback = input("   (Enter = good / type the correct answer / s = skip) > ").strip()
        if feedback.lower() == "s":
            print("   skipped\n")
            continue
        save_example(question, feedback or answer)
        print(f"   saved ({count_examples()} examples). Retrain with python train.py to learn it.\n")
    else:
        print()
