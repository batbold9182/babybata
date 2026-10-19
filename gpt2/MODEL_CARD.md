---
license: mit
language:
- en
base_model: openai-community/gpt2
library_name: transformers
pipeline_tag: text-generation
tags:
- chatbot
- portfolio
---

# BabyBata 1 (GPT-2) 🍼

The first version of a small assistant for [Batbold Samdan](https://github.com/batbold9182)'s portfolio. GPT-2 small (124M), fine-tuned to answer questions about his projects, skills and contact info. An AI/ML **learning project**, superseded by [BabyBata 2.0](https://huggingface.co/BatboldSamdan/babybata2).

- **Base model:** GPT-2 small
- **Method:** full fine-tuning with the token/position embeddings frozen
- **Data:** ~640 question/answer pairs: hand-written + augmented (one consistent answer per topic, many phrasings, casual variants)
- **Exact facts use slots:** the model writes `{EMAIL}`, `{GITHUB}`, `{GPA}`… and code replaces them (`facts.py` in the repo). Without this, GPT-2 blended the email and GitHub handle into a wrong address.
- **Code:** [github.com/batbold9182/babybata](https://github.com/batbold9182/babybata)

## Prompt format
```
User: <question>
BabyBata:
```
The model continues after `BabyBata:` and ends with `<|endoftext|>`. Fill the `{SLOTS}` in the output with `facts.fill()` from the repo.

## Evaluation (15 questions not in the training data)
| Version | Score |
|---|---|
| v1: 133 hand-written examples | 4/15 |
| v2: + `{SLOTS}` for exact facts | 8/15 |
| v3: + 486 augmented examples | 14/15 |
| v4: more phrasings, fewer epochs | 15/15 |

## Limitations
GPT-2 memorizes rather than understands: on harder unseen questions v3 scored only ~8/12 and sometimes invented details. BabyBata 2.0 (Qwen + LoRA + grounding) fixes most of this.
