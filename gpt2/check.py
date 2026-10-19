# Fact-check BabyBata: does it get the important things right?
# Run:  python check.py                      <- babybata-model/
#       python check.py models/archive/babybata-model-v1    <- any other saved version
#
# The loss number never warned us the email was wrong. This does.
# Answers are generated greedily (no randomness), so results are repeatable.

import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.utils import logging

from facts import fill
from train import OUT_DIR, prompt_for

logging.set_verbosity_error()

# (question, text the answer MUST contain). Phrased differently from the training data on purpose.
TESTS = [
    ("who are you", "BabyBata"),
    ("whats your name", "BabyBata"),
    ("how do i contact batbold", "{EMAIL}"),
    ("what's his email address?", "{EMAIL}"),
    ("where is batbold's github", "{GITHUB}"),
    ("where does batbold go to university", "{UNIVERSITY}"),
    ("what's batbold's gpa", "{GPA}"),
    ("what projects has batbold made", "Express"),
    ("tell me about smart campus", "university"),
    ("what's the salon app", "salon"),
    ("is batbold looking for work", "internship"),
    ("where is batbold from", "Mongolia"),
    ("what does batbold do for fun", "basketball"),
    ("what's the weather", "only know about Batbold"),
    ("can you do my homework", "only know about Batbold"),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    tokenizer = GPT2TokenizerFast.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).eval()

    passed = 0
    for question, must_contain in TESTS:
        ids = tokenizer.encode(prompt_for(question), return_tensors="pt")
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=60,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).split("\nUser:")[0].strip()
        ok = must_contain.lower() in raw.lower()
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {question}\n      -> {fill(raw)}")
    print(f"\n{path}: {passed}/{len(TESTS)} facts correct")


if __name__ == "__main__":
    main()
