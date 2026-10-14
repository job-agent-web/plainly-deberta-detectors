# Public browser detector model distribution

Open-source model components used by Plainly. This site distributes only model files and attribution. It collects no user text, credentials or payments and provides no hosted inference service.

Weights are pinned, converted with ONNX Runtime 1.23.2, and checked against original model outputs before release. Weight-only int8 retains float activation precision. This is a conversion parity check, not an accuracy evaluation. Estimates are not proof of authorship.

- desklib: https://huggingface.co/desklib/ai-text-detector-v1.01
  - Licence: MIT. See upstream source and accompanying notices.
  - SHA256: d3ddabe24abb6733b81333baf9a920265fb4e1e984b9b2e91cb4fe0092b6bacb
- tropa: https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small
  - Licence: Apache-2.0. See upstream source and accompanying notices.
  - SHA256: 7058ff1b96f78a11f83d2ff2e748ac89786035ec0c2e3d3926e12a4ac2f10d3c

SLOP's FP16 operations were promoted to FP32 for CPU support; its released q4 weights are retained. SLOP requires 17 readability features. Plainly uses standard English formulas and a syllable heuristic because the author's feature implementation is unavailable. SmolLM's output named probability_human is documented as inverted; its raw output is interpreted as the AI estimate.

First-time downloads are large. Browser caches may be evicted; model inference needs a desktop browser with sufficient memory.
