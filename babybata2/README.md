# BabyBata 2.0 🍼

The smarter BabyBata: **Qwen2.5-0.5B-Instruct**, fine-tuned with **LoRA**, answering from a **fact sheet** (`profile.txt`) instead of memorizing facts.

| | BabyBata v3 (GPT-2, `../`) | BabyBata 2.0 (this folder) |
|---|---|---|
| Base model | GPT-2 (2019, text continuation) | Qwen2.5-0.5B-Instruct (2024, follows instructions) |
| Where facts live | Memorized in the weights (+ `{SLOTS}`) | **Read from `profile.txt`** in every prompt |
| Change a fact | Edit data + retrain | **Edit `profile.txt`, no retraining** |
| Unknown questions | Guesses a memorized answer | Trained to say "I don't know" |
| Follow-ups ("tell me more about it") | ❌ | ✅ remembers the last 3 turns |
| Training | Full fine-tune | LoRA (~1% of weights), Colab T4 |

## Quick reference

| I want to… | Do this |
|---|---|
| Change or add a fact | Edit `profile.txt`. That's it |
| Change how it answers (tone, refusals…) | Edit `make_data.py` → `python make_data.py` → retrain |
| Correct answers by chatting | `python chat.py --teach` → retrain |
| Retrain on Colab | Bump `VERSION` in `make_notebook.py` → `python make_notebook.py` → open `babybata2.ipynb` → Run All |
| Check quality | `python check.py` (27 tests) |
| Compare with un-trained Qwen | `python check.py --base` / `python chat.py --base` |
| Chat | `python chat.py` (commands: `reset`, `quit`) |

## Files

| File | What it is |
|---|---|
| `profile.txt` | **The facts.** Everything BabyBata knows about Batbold |
| `bot.py` | Shared: the rules + profile prompt, data format, loading, answering |
| `make_data.py` | Generates `data/generated.txt`: topics, "does he know X?", "I don't know" cases, off-topic refusals, follow-up dialogues |
| `data/generated.txt` | Generated. Don't edit by hand |
| `data/handwritten.txt` | Your own examples + teach-mode corrections |
| `train.py` | LoRA fine-tuning (GPU only) → merges into `babybata2-model/` |
| `check.py` | 27-question test (never in the training data), incl. honesty tests |
| `chat.py` | Chat with memory of recent turns, `--teach`, `--base` |
| `make_notebook.py` | Packs everything into `babybata2.ipynb` for Colab |

## How it works
1. Every question is sent with **rules + profile** as the system prompt (~1,000 tokens).
2. Fine-tuning teaches **behavior**: short answers, he/his, read the facts from the profile, say "I don't know" when they're missing, decline off-topic.
3. **LoRA**: Qwen's 494M weights stay frozen; small add-on matrices (~1%) are trained, then merged in.
4. Only BabyBata's replies are graded (loss masking), including across follow-up turns.

## After training on Colab
1. Download `MyDrive/babybata2/babybata2-model-<VERSION>` from Drive.
2. Extract it here and rename it to `babybata2-model` (so `config.json` is directly inside it).
3. `python check.py`, then `python chat.py`.

Needs locally: `torch`, `transformers` (already installed). On your laptop each answer takes a few seconds.

## Gotchas
- **Training needs a GPU.** On a CPU it would take many hours.
- **Keep `profile.txt` accurate.** BabyBata repeats what's in it, and everything in it is public.
- **Always regenerate the notebook** (`python make_notebook.py`) after changing any file: Colab can't see this PC.

## Score history
| Version | Fact check (27 tests) |
|---|---|
| plain Qwen + profile (no training) | 18/27 (invents facts, rambles, talks as "I") |
| 2.0 v1 | 26/27 (4 test questions had leaked into training dialogues, so slightly flattering). Found later: loses track in follow-ups ("in which code?"), claimed Smart Campus has a demo |
| 2.0 v2 | ? / 32 (+15 follow-up dialogues, demo yes/no, more privacy questions, follow-up tests, leak check) |
