# BabyBata 🍼

A small AI assistant for my portfolio site that answers visitors' questions about my projects, skills and how to reach me.

This is an **AI/ML learning project**. I first wrote neural networks from scratch in NumPy to understand the fundamentals, then fine-tuned two language models, learning from every failure along the way.

🤗 **Model on Hugging Face:** [BatboldSamdan/babybata2](https://huggingface.co/BatboldSamdan/babybata2)

```
You:      what projects has he built?
BabyBata: His main projects are Express (a music review app), Smart Campus System,
          Beauty Salon Finder and Python Game Hub. He also built a Book Catalogue app
          and AI projects like me!
You:      tell me more about the second one
BabyBata: Smart Campus System is a university management app for iOS, Android and web,
          with dashboards for admins, faculty and students and real-time chat.
You:      does he know kotlin?
BabyBata: Kotlin isn't listed in his profile, so I don't know. You can ask him directly
          at batbold918273645@gmail.com.
```
*(real output from BabyBata 2.0)*

---

## Two versions

| | [`gpt2/`](gpt2/): BabyBata 1 | [`babybata2/`](babybata2/): BabyBata 2.0 |
|---|---|---|
| Base model | GPT-2 small (124M, 2019) | Qwen2.5-0.5B-Instruct (494M, 2024) |
| Training | Full fine-tuning (embeddings frozen) | **LoRA**: 1.7% of the weights trained |
| Where facts live | Memorized; exact facts injected via `{SLOTS}` | **Read from [`profile.txt`](babybata2/profile.txt)** in the prompt (grounding) |
| Change a fact | Edit data, retrain | Edit `profile.txt`, no retraining |
| Unknown questions | Guesses a memorized answer | Says "I don't know" |
| Follow-up questions | ❌ | ✅ remembers recent turns |
| Download | not published | [🤗 BatboldSamdan/babybata2](https://huggingface.co/BatboldSamdan/babybata2) |

## Results

Each version is scored by a fact-check script (`check.py`) on questions that are **not** in the training data.

**BabyBata 1 (GPT-2), 15-question test**
| Version | What changed | Score |
|---|---|---|
| v1 | 133 hand-written examples | 4/15 |
| v2 | Exact facts as `{EMAIL}`-style slots (fixed a blended, wrong email) | 8/15 |
| v3 | +486 augmented examples, one consistent answer per topic | 14/15 |
| v4 | More phrasings, fewer epochs (validation loss showed overfitting) | **15/15** |

On 12 harder, unseen questions, v3 only got ~8/12. GPT-2 memorizes well but doesn't really understand, which is why I built 2.0.

**BabyBata 2.0 (Qwen + LoRA), 27-question test incl. honesty checks**
| Version | Score | Notes |
|---|---|---|
| Plain Qwen + profile, no training | 18/27 | Invented facts ("Batbold is 4 years old"), rambled, spoke as if it were me |
| v1 (LoRA, 3 epochs) | 26/27 | 4 test questions had leaked into training dialogues, so slightly optimistic; leak check added since |
| v2 | in progress | +follow-up dialogues, demo yes/no answers, 32-question test incl. multi-turn |

## How BabyBata 2.0 works

```
 visitor question ──► [ rules + profile.txt + last 3 turns + question ] ──► Qwen + LoRA ──► short answer
```

1. **Grounding:** every prompt carries the rules and the fact sheet, so the model *reads* facts instead of memorizing them.
2. **Fine-tuning teaches behavior:** short answers, "he/his", "I don't know" when a fact is missing, polite refusal of off-topic requests, follow-ups.
3. **LoRA:** the base weights stay frozen; small add-on matrices are trained and then merged into one normal model.
4. **Loss masking:** only BabyBata's replies are graded, never the prompt.
5. **Memory-friendly loss:** the vocabulary has 151,936 words, so scoring every one of ~1,100 prompt positions ran a 15 GB GPU out of memory. The loss is computed only at reply positions instead: identical value, ~25× less memory for that step.


## Project structure

```
babybata/
├── gpt2/                    BabyBata 1 (GPT-2)
│   ├── augment.py           generates training data: one answer + many phrasings per topic
│   ├── facts.py             exact facts, filled into {SLOTS}
│   ├── train.py / check.py / chat.py / compress.py
│   ├── make_notebook.py     packs everything into babybata.ipynb for Colab
│   └── data/
└── babybata2/               BabyBata 2.0 (Qwen + LoRA)
    ├── profile.txt          the fact sheet
    ├── make_data.py         generates conversations: topics, "I don't know", follow-ups, refusals
    ├── bot.py               prompt, data format, loading, answering
    ├── train.py / check.py / chat.py
    ├── make_notebook.py     packs everything into babybata2.ipynb for Colab
    └── data/
```
Each folder has its own README with all the commands.

## Running it

Trained models (0.5–1 GB each) aren't in this git repo.

**Use the published model** (easiest):
```
pip install torch transformers
cd babybata2
hf download BatboldSamdan/babybata2 --local-dir babybata2-model
python chat.py
```

**Or train your own** on a free Google Colab GPU:
1. `cd babybata2` and run `python make_notebook.py`
2. Open `babybata2.ipynb` in Colab (or VS Code with the Colab extension), choose a **T4 GPU**, then **Run All** (~10 min)
3. Download the trained model into `babybata2/babybata2-model/`, then `python chat.py`

## Tech
Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · NumPy · Google Colab (T4)

## Next
- Train and evaluate 2.0 v2
- Add BabyBata to my [portfolio site](https://portfolio-batbold.vercel.app)

---
Made by [Batbold Samdan](https://github.com/batbold9182)
