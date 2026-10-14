---
license: apache-2.0
language:
- en
base_model:
- desklib/ai-text-detector-v1.01
pipeline_tag: text-classification
tags:
- ai-detector
- ai-detection
- ai-text-detector
- ai-content-detection
- chatgpt-detector
- gpt-detector
- llm-detection
- generated-text-detection
- ai-generated-text
- humanized-ai-detection
- deberta-v3
library_name: transformers
---

# tropa-mini — open-weights AI Text Detector by wasitaigenerated

**Detect AI-generated text from ChatGPT, GPT-5, Claude, Gemini, Llama and other LLMs — free, open weights, runs on CPU.**

**tropa-mini** is the open-weights AI detector by [wasitaigenerated.com](https://wasitaigenerated.com), the small sibling of the **tropa-2** model behind the hosted API. In our benchmark of every notable open-source AI text detector on public datasets, tropa-mini comes out as the **strongest open-weights AI text detector available** — best ROC-AUC and the highest detection rate at a fixed 0.5 % false-positive rate across raw, humanized and frontier-model text ([full results below](#benchmarks-vs-other-open-weights-ai-detectors)).

- ✅ **93 % of raw AI text** caught at a 0.5 % false-positive rate (next-best open model: 84 %)
- ✅ The only open detector with a **humanizer class** — it flags AI text laundered through paraphrasing / "humanizer" tools
- ✅ DeBERTa-v3-large backbone, ~1.7 GB, **CPU-friendly** — no GPU required
- ✅ Apache-2.0, commercial use allowed

Unlike most detectors it has a **4-class head** — it doesn't just say *AI or human*:

| class | meaning |
|---|---|
| `human` | written by a person |
| `ai` | raw LLM output (ChatGPT, Claude, Gemini, …) |
| `ai_edited` | human text lightly rewritten or polished by an LLM |
| `humanized` | AI text passed through a "humanizer" / paraphrasing tool |

`ai_score = 1 − P(human)` is the headline number in [0, 1].

## How to detect AI-generated text in Python

```python
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

class AIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    _tied_weights_keys = []          # transformers>=5 compatibility

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        n = getattr(config, "detector_num_labels", 1)
        self.classifier = nn.Linear(config.hidden_size, n)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids, attention_mask=attention_mask)
        h = out[0]                                    # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1)      # mean pooling over real tokens
        return self.classifier(pooled)

repo = "wasitaigeneratedcom/ai-text-detector-small"
tok = AutoTokenizer.from_pretrained(repo)
model = AIDetectionModel.from_pretrained(repo).eval()

text = "Your text here..."
enc = tok(text, truncation=True, max_length=768, return_tensors="pt")
with torch.inference_mode():
    probs = torch.softmax(model(**enc), dim=-1)[0]

labels = ["human", "ai", "ai_edited", "humanized"]
ai_score = 1.0 - probs[0].item()
print({l: round(p.item(), 4) for l, p in zip(labels, probs)}, "| ai_score:", round(ai_score, 4))
```

For long documents, split into ≤768-token chunks (sentence-aligned) and average chunk scores weighted by length. A practical decision threshold at a 0.5 % false-positive operating point is `ai_score ≥ 0.976` (see `serving_head.json`).

Prefer an API call over self-hosting? The [wasitaigenerated AI Detector API](https://www.wasitaigenerated.com/ai-detector-api) runs the substantially stronger **tropa-2** model (98.5 % vs 93.2 % on raw AI, 76 % vs 42 % on humanized text — [comparison below](#tropa-mini-vs-tropa-2-the-hosted-api)), is one POST request, and comes with 1,000 free credits.

## Benchmarks (vs. other open-weights AI detectors)

All numbers are measured on **public datasets**, so anyone can reproduce them:

- [Jabarian & Imas (2025)](https://github.com/brianjabarian/DetectionAI) — 1,930 human passages, 7,683 raw generations (GPT-4.1, Claude Opus 4, Claude Sonnet 4, Gemini 2.0 Flash), 7,867 StealthGPT-humanized versions
- [Liang et al. (2023)](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias) — TOEFL essays by non-native writers
- A 1,060-text **frontier set**: GPT-5.x, Claude Opus 5, Gemini 3.x, Grok, DeepSeek V4, …
- 5,000 **pre-LLM (2018) FineWeb web pages** as a neutral human pool

Every model gets its decision threshold set to the same matched **0.5 % false-positive rate** on the same 6,930 human documents. Recall is then measured per group — a fair, like-for-like comparison.

| model | ROC-AUC | raw AI | humanized AI | frontier models |
|---|---|---|---|---|
| **tropa-mini (this model)** | **0.968** | **93.2 %** | **41.6 %** | **33.6 %** |
| desklib/ai-text-detector-v1.01 | 0.875 | 83.9 % | 4.0 % | 1.8 % |
| SuperAnnotate/ai-detector | 0.824 | 0.5 % | 1.4 % | 0.6 % |
| Hello-SimpleAI/chatgpt-detector-roberta | 0.571 | 0.8 % | 0.4 % | 0.2 % |
| yaful/MAGE | 0.507 | —\* | —\* | —\* |
| roberta-large-openai-detector | 0.313 | 0.0 % | 0.1 % | 0.0 % |

\* MAGE cannot reach a 0.5 % FPR at any threshold (it flags 26 % of ordinary human web text with score > 0.9999).

One honest caveat: on the Liang non-native TOEFL essays tropa-mini flags 15.6 % at that operating point — more than desklib's 3.3 % (which, however, detects almost nothing at the same FPR). Use conservative thresholds for learner writing.

## tropa-mini vs. tropa-2 (the hosted API)

The hosted detector at wasitaigenerated.com runs **tropa-2**, a larger, continuously retrained system; tropa-mini is its fast, CPU-friendly sibling. API numbers below were measured **through the public API endpoint** (verdict ≥ 90) on the same public datasets — reproducible with any API key.

| dataset | tropa-mini (open) | tropa-2 (hosted API) |
|---|---|---|
| Jabarian human passages, falsely flagged | 0.5 % | **0.1 %** |
| Jabarian raw AI (4 frontier 2025 models) | 93.2 % | **98.5 %** |
| StealthGPT-humanized | 41.6 % | **76.0 %** (90 % at verdict ≥ 70) |
| 2026 frontier set (GPT-5.x, Opus 5, …) | 33.6 % | **82.8 %** (88 % at verdict ≥ 70) |
| Non-English text | English-first | supported (measured FPR 0.00–0.04 % across DE/FR/ES/IT/PT/NL) |

## FAQ

**Which AI models does it detect?**
Text from ChatGPT (GPT-4, GPT-4o, GPT-5.x), Claude, Gemini, Llama, Mistral, DeepSeek, Grok and other large language models. Detection is strongest for the model generations it was trained against (see the frontier column above); the hosted tropa-2 API is retrained continuously as new models appear.

**Can it detect humanized or paraphrased AI text?**
Yes — tropa-mini is the only open-weights detector with a dedicated `humanized` class, and it catches 10× more humanizer output than the next-best open model (41.6 % vs 4.0 % at the same false-positive rate). The hosted API catches 76–90 %.

**How accurate is it on human writing?**
At the recommended threshold it falsely flags about 1 in 200 human documents (0.5 % FPR), measured on 6,930 human texts. The hosted API operates at ~1 in 1,000 and below.

**Does it work for essays and academic writing?**
Yes, with a caution: like all AI detectors it flags short, simple learner prose (e.g. non-native TOEFL essays) more often than average. For essay and thesis checking, treat scores as evidence and keep a human in the loop.

**Is it free?**
Yes — Apache-2.0, free for commercial use. The hosted API has a free tier (1,000 credits).

**What languages are supported?**
tropa-mini is English-first. The hosted detector supports multiple languages with a measured false-positive rate of 0.00–0.04 % across German, French, Spanish, Italian, Portuguese and Dutch.

## Intended use

- Optimized for **English** prose of **50+ words**.
- Open-weights releases are snapshots; the hosted API is retrained continuously as new generator models appear.
- A score is **evidence, not proof**. Don't use it as the sole basis for accusations or academic-integrity decisions — combine it with process evidence and human judgment, as with every AI detector.

## Attribution

Fine-tuned from [desklib/ai-text-detector-v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01) (MIT), which builds on [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) (MIT). Both upstream licenses permit this derivative; upstream notices are preserved in `NOTICE`.

*Built by [wasitaigenerated](https://wasitaigenerated.com) — AI content detection for text and images: [AI detector](https://wasitaigenerated.com) · [ChatGPT detector](https://www.wasitaigenerated.com/detectors/chatgpt-detector) · [AI essay detector](https://www.wasitaigenerated.com/detectors/ai-essay-detector) · [deepfake detector](https://www.wasitaigenerated.com/detectors/deepfake-detector) · [AI detection API](https://www.wasitaigenerated.com/ai-detector-api)*
