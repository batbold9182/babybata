---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
- lora
- peft
- chatbot
- portfolio
---

# BabyBata 2.0 🍼

A small assistant that answers questions about [Batbold Samdan](https://github.com/batbold9182)'s portfolio: his projects, skills, education and contact info. An AI/ML **learning project**.

- **Base model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Method:** LoRA fine-tuning (r=16, all attention + MLP projections, 1.7% of weights trained), merged into the base model, saved in float16
- **Data:** ~630 generated conversations: topic Q&A, "does he know X?", "I don't know" cases, off-topic refusals, multi-turn follow-ups
- **Training:** 3 epochs on a free Colab T4 (~10 min), loss on assistant replies only
- **Code:** [github.com/batbold9182/babybata](https://github.com/batbold9182/babybata)

## Important: it needs the profile
BabyBata is trained to answer **from a fact sheet in the system prompt** (grounding), not from memory. Use it with the same system prompt it was trained with: the rules + `profile.txt` from the repo (`bot.py` builds it).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "BatboldSamdan/babybata2"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

system = open("babybata2/profile.txt").read()   # in practice: bot.system_prompt() from the repo
messages = [{"role": "system", "content": system},
            {"role": "user", "content": "what projects has he built?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Evaluation
Custom fact check on questions not in the training data (incl. honesty checks like "does he know Rust?" → must say it doesn't know):

| Model | Score |
|---|---|
| Qwen2.5-0.5B-Instruct + profile, no fine-tuning | 18/27 |
| BabyBata 2.0 v1 | 26/27 (4 test questions had leaked into training dialogues, so slightly optimistic) |

## Limitations
- Only knows what's in the profile; it's a small model and can still be wrong. Verify important details on the portfolio site.
- Known v1 issues: sometimes loses track in follow-up questions; once claimed a project had a live demo when it doesn't.
- All training data is public information from the portfolio site.
