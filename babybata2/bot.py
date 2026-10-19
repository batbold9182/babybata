# Shared pieces for BabyBata 2.0: the prompt, the data format, loading and answering.

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(HERE, "profile.txt")
DATA_FILES = [os.path.join(HERE, "data", "generated.txt"),     # made by make_data.py
              os.path.join(HERE, "data", "handwritten.txt")]   # yours + teach mode
TEACH_FILE = DATA_FILES[1]
MODEL_DIR = os.path.join(HERE, "babybata2-model")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

RULES = """You are BabyBata, a small AI assistant on Batbold Samdan's portfolio website. Batbold fine-tuned you himself.
Answer visitors' questions about Batbold using ONLY the profile below.
- Keep answers short: 1 to 3 sentences.
- Refer to Batbold as "he" / "his".
- If the answer is not in the profile, say you don't know and suggest emailing him. Never invent facts.
- Politely decline anything that is not about Batbold (homework, general knowledge, coding help)."""


def system_prompt():
    """The instructions + the fact sheet. Edit profile.txt to change facts: no retraining needed."""
    with open(PROFILE_FILE, encoding="utf-8") as f:
        return RULES + "\n\nPROFILE:\n" + f.read().strip()


def load_conversations(path):
    """Blocks separated by blank lines; each block alternates 'User:' / 'BabyBata:' lines.
    Returns a list of conversations, each a list of (question, answer) turns."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
    conversations = []
    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l and not l.startswith("#")]
        turns = []
        for user, bot in zip(lines[0::2], lines[1::2]):
            if user.startswith("User:") and bot.startswith("BabyBata:"):
                turns.append((user[5:].strip(), bot[9:].strip()))
        if turns and len(turns) * 2 == len(lines):
            conversations.append(turns)
    return conversations


def load_all_conversations():
    return [c for path in DATA_FILES for c in load_conversations(path)]


def to_messages(turns, question=None):
    """(question, answer) turns -> chat messages, starting with the system prompt."""
    messages = [{"role": "system", "content": system_prompt()}]
    for q, a in turns:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    if question is not None:
        messages.append({"role": "user", "content": question})
    return messages


def load(path=MODEL_DIR):
    """Returns (tokenizer, model, device). float16 on a GPU, float32 on a CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    return tokenizer, model, device


def answer(tokenizer, model, device, question, history=(), sample=False):
    """Answer `question`, given earlier (question, answer) turns for follow-ups."""
    import torch

    text = tokenizer.apply_chat_template(to_messages(list(history), question),
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=sample,
                             temperature=0.3 if sample else None, top_p=0.9 if sample else None,
                             top_k=None, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
