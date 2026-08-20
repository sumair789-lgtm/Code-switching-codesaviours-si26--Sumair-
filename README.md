# Code Switching NLP | Code Saviours SI-26 | Sumair Parveiz


Identifies which words in a mixed Roman Urdu–English sentence are Urdu, English, or a blend of both — word by word.

## Why This Matters
Roman Urdu-English code-switching is how most Pakistanis actually write online, but almost no NLP tooling handles it well — models trained on "clean" English or Urdu break down on real mixed-language text. This project builds a labeled dataset for exactly this pattern and trains a model to tag each word by language, a building block for downstream tasks like sentiment analysis or search over code-switched text.

## Live Demo
🔗 [Model on HuggingFace Hub](https://huggingface.co/Sumair-Parveiz/code-switching-codesaviours-si26-sumair)

## How It Works
A lexicon-based pipeline generated a labeled dataset of naturalistic Roman Urdu-English sentences, tagging every word as URD, ENG, or MIX. XLM-RoBERTa (`xlm-roberta-base`) was fine-tuned for token classification, treating language identification as a per-word tagging task similar to part-of-speech tagging. The best checkpoint was kept by validation loss rather than the final epoch, to avoid overfitting on a small dataset.

## Results
| Label | F1 Score |
|-------|----------|
| URD | 0.99 |
| ENG | 0.98 |
| MIX | 0.00* |

**Weighted average F1: 0.98**

*MIX made up under 0.15% of all labeled tokens — too few examples for the model to learn this class reliably. Reported honestly rather than hidden; the URD/ENG split, which covers 99.85% of real tokens, performs at near-ceiling accuracy.

## How to Run Locally
```bash
pip install transformers torch sentencepiece
```

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Sumair-Parveiz/code-switching-codesaviours-si26-sumair")
model = AutoModelForTokenClassification.from_pretrained("Sumair-Parveiz/code-switching-codesaviours-si26-sumair")

words = "Mujhe kal office jana hai but weather bad hai".split()
inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
preds = outputs.logits.argmax(-1)[0]
print(preds)
```

---
**Built by:** Sumair Parveiz | Code Saviours SI-26 | 2026
