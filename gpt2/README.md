# BabyBata 🍼

A small chatbot for Batbold's portfolio: **GPT-2 fine-tuned** to answer questions about Batbold's projects, skills and contact info.

---

## Quick reference

| I want to… | Do this |
|---|---|
| Change a fact (email, GitHub, GPA…) | Edit `facts.py`. **No retraining needed** |
| Add or change a topic / answer | Edit `augment.py` → `python augment.py` → retrain |
| Correct answers by chatting | `python chat.py --teach` → retrain |
| Retrain on Colab (fast) | `python make_notebook.py` → open `babybata.ipynb` → Run All |
| Retrain on the laptop (slow) | `python augment.py` → `python train.py` |
| Check how good it is | `python check.py` |
| Chat with it | `python chat.py` |

---

## Project files

| File | What it is |
|---|---|
| `facts.py` | Exact facts (email, GitHub, LeetCode, full name, university, GPA). The model writes `{EMAIL}`; this file fills in the real value |
| `augment.py` | **Main place to edit knowledge.** Topics, each with ONE answer and many ways to ask. Generates `data/generated.txt` |
| `data/babybata.txt` | Hand-written examples + everything saved by teach mode |
| `data/generated.txt` | Made by `augment.py`. **Don't edit by hand**: it gets overwritten |
| `train.py` | Fine-tunes GPT-2 on both data files → `babybata-model/` |
| `check.py` | Fact test: 15 key questions, prints PASS/FAIL and a score |
| `chat.py` | Talk to BabyBata (`--teach`, `--small`, `--base` modes) |
| `compress.py` | Experiment: 8/4-bit quantization + removing 6 of 12 layers, with a report card |
| `make_notebook.py` | Packs the data and code into `babybata.ipynb` for Colab |
| `babybata.ipynb` | Generated Colab notebook. **Don't edit by hand**: regenerate it instead |

## Model folders

| Folder | What it is |
|---|---|
| `babybata-model/` | The **active** model (what `chat.py` and `check.py` use) |
| `babybata-model-small/` | 6-layer compressed version (`chat.py --small`) |
| `babybata-model-v1/` | Old v1 backup (wrong email, 4/15 on the fact check) |
| `babybata-model-small-v1/` | Old compressed v1 |

On Google Drive: `MyDrive/babybata/`

---

## Workflows

### Change a fact (email, GitHub, GPA…)
1. Edit the value in `facts.py`.
2. That's it. `chat.py` fills in the new value immediately, no retraining.

### Add or change what BabyBata knows
1. Open `augment.py` and find the topic in `TOPICS` (or add a new one):
   ```python
   "hobbies": (
       "Batbold plays basketball and Dota 2, and loves discovering new music.",   # ONE answer
       ["What are Batbold's hobbies?", "Hobbies?", "What do you do for fun?"],   # MANY questions
   ),
   ```
2. Regenerate the data:
   ```
   python augment.py
   ```
3. Retrain (see below).

Tips:
- **One answer per topic.** Different answers to similar questions confuse the model.
- **Many ways to ask:** formal, casual, short ("GPA?"), and indirect ("is Batbold a good student?").
- **Exact facts go in `facts.py`** as a `{SLOT}`, never written out in answers.
- **Keep answers short.** One fact per answer is learned more reliably.
- Questions tested by `check.py` are left out automatically, so the test stays fair.

### Teach by chatting
```
python chat.py --teach
```
After each answer: **Enter** = good, save it · **type a better answer** = save that · **s** = skip.
Corrections are appended to `data/babybata.txt` (real emails etc. are turned back into `{SLOTS}` automatically). Chatting alone never changes the model: **retrain** to learn the corrections.

### Retrain on Colab (recommended, ~5 min)
0. Bump `VERSION` at the top of `make_notebook.py` (`"v3"` → `"v4"`…) so Drive keeps every version.
1. Pack the latest data and code into the notebook:
   ```
   python make_notebook.py
   ```
2. Open `babybata.ipynb` in VS Code → **Select Kernel → Colab → New Colab Server → GPU → T4**.
3. **Run All** (cells 6–7, the compression, are optional).
4. Check cell **3b** (the fact check score).
5. Cell 5 saves to Drive as `MyDrive/babybata/babybata-model-<VERSION>`.
6. Download that folder, extract it into this project and rename it to `babybata-model` (rename the old one first if you want to keep it).

⚠️ **Always run `python make_notebook.py` after changing any data or code.** The notebook carries a copy, and Colab can't see files on this PC.

### Retrain on the laptop (~45+ min)
```
python augment.py
python train.py
```
Reads the files directly (no notebook step). The first run downloads GPT-2 (~550 MB).

### Check quality
```
python check.py                      # active model
python check.py babybata-model-v1    # any other version, to compare
```

---

## Settings (top of `train.py`)

| Setting | Current | Meaning |
|---|---|---|
| `BASE_MODEL` | `"gpt2"` | Where training starts. Point to a folder to continue from a saved version |
| `EPOCHS` | `6` | Passes over the data. More = learns facts better, but can memorize |
| `LEARNING_RATE` | `5e-5` | Kept tiny so GPT-2's English isn't overwritten |
| `FREEZE_EMBEDDINGS` | `True` | Don't train the 38.6M-weight vocabulary table (faster, less memory) |

Generation settings (in `chat.py`): `temperature=0.3` (low = fewer invented facts), `repetition_penalty=1.05` (kept tiny so names and emails can repeat exactly).

---

## Score history

| Version | Data | Fact check |
|---|---|---|
| v1 | 35 → 133 hand-written examples, literal email | 4/15 |
| v2 | 133 examples with `{SLOTS}` | 8/15 |
| v3 | + 486 generated by `augment.py` + teach mode (630 total), best at epoch 4 of 12 | **14/15** |
| v4 | + more education phrasings (fixes "go to university"), 6 epochs | ? |

---

## Gotchas
- **Colab deletes everything** when the session ends. Save to Drive (cell 5) before closing.
- **`HF_TOKEN` warnings** on Colab are harmless. Ignore them; no token needed.
- **"ipykernel required" popup** = the notebook is on local Python. Switch the kernel to **Colab**.
- **Unzipped a Drive download?** Make sure `config.json` sits directly inside the model folder, not in a nested one.
- **Everything BabyBata knows is public.** Only put info in the data you're happy for any visitor to see.
