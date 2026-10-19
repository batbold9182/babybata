# Talk to BabyBata 2.0. It remembers the last few turns, so follow-ups work
# ("what projects has he built?" -> "tell me more about the first one").
# Run:  python chat.py           <- babybata2-model/
#       python chat.py --teach   <- correct its answers; saved to data/handwritten.txt for the next training
#       python chat.py --base    <- plain Qwen + profile, no fine-tuning (to compare)
# Commands while chatting:  reset = forget the conversation,  quit = exit

import os
import sys

from transformers.utils import logging

from bot import BASE_MODEL, MODEL_DIR, TEACH_FILE, answer, load

logging.set_verbosity_error()
MEMORY = 3  # how many earlier turns it sees

path = BASE_MODEL if "--base" in sys.argv else MODEL_DIR
if path == MODEL_DIR and not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    sys.exit(f"No model in {MODEL_DIR}. Train it on Colab and download it here.")
teach = "--teach" in sys.argv

print("loading...")
tokenizer, model, device = load(path)


def save_example(question, reply):
    new_file = not os.path.exists(TEACH_FILE)
    with open(TEACH_FILE, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Hand-written examples + corrections from: python chat.py --teach\n")
        f.write(f"\nUser: {question}\nBabyBata: {reply}\n")


print(f"Talking to {'plain Qwen (no fine-tuning)' if path == BASE_MODEL else 'BabyBata 2.0'}. "
      "Commands: reset, quit")
if teach:
    print("TEACH MODE: after each answer ->  Enter = good, save it  |  type a better answer = save that  |  s = skip")
print()

history = []
while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() == "quit":
        break
    if question.lower() == "reset":
        history = []
        print("(conversation forgotten)\n")
        continue

    reply = answer(tokenizer, model, device, question, history[-MEMORY:])
    print(f"BabyBata: {reply}")

    if teach:
        feedback = input("   (Enter = good / type the correct answer / s = skip) > ").strip()
        if feedback.lower() != "s":
            reply = feedback or reply
            save_example(question, reply)
            print("   saved to data/handwritten.txt")
    history.append((question, reply))
    print()
