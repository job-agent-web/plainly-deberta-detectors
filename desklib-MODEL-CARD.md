---
license: mit
datasets:
- liamdugan/raid
language:
- en
base_model:
- microsoft/deberta-v3-large
pipeline_tag: text-classification
library_name: transformers
---
# desklib/ai-text-detector-v1.01

## Overview

This AI-generated text detection model, developed by Desklib, is designed to classify English text as either human-written or AI-generated. It currently leads the [RAID Benchmark for AI Detection](https://raid-bench.xyz/leaderboard?domain=all&decoding=all&repetition=all&attack=all).  This model is a fine-tuned version of **microsoft/deberta-v3-large**, leveraging a transformer-based architecture to achieve high accuracy. It is robust and handles various adversarial attacks across different domains remarkably well. This model is particularly useful for applications in content moderation, academic integrity, journalism, and anywhere the authenticity of text is paramount.

**Desklib** provides AI-based tools for personalized learning and study help. This model is one of the many tools offered by Desklib for students, educators and universities.

**Try the model online!**: [Desklib AI Detector](https://desklib.com/ai-content-detector/)

**Github Repo**: [https://github.com/desklib/ai-text-detector](https://github.com/desklib/ai-text-detector)

## Performance

This model achieves top performance on the RAID benchmark at the time of submission: [Visit RAID Leaderboard](https://raid-bench.xyz/leaderboard?domain=all&decoding=all&repetition=all&attack=all)

## Model Architecture

The model is built upon a fine-tuned **microsoft/deberta-v3-large** transformer architecture. The core components include:

*   **Transformer Base:**  The pre-trained `microsoft/deberta-v3-large` model serves as the foundation. This model utilizes DeBERTa (Decoding-enhanced BERT with disentangled attention), an improved version of BERT and RoBERTa, which incorporates disentangled attention and enhanced mask decoder for better performance.
*   **Mean Pooling:**  A mean pooling layer aggregates the hidden states from the transformer, creating a fixed-size representation of the input text. This method averages the token embeddings, weighted by the attention mask, to capture the overall semantic meaning.
*   **Classifier Head:** A linear layer acts as a classifier, taking the pooled representation and outputting a single logit. This logit represents the model's confidence that the input text is AI-generated. Sigmoid activation is applied to the logit to produce a probability.

## Usage

Here's how to use the model with the Hugging Face `transformers` library:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label

def main():
    # --- Model and Tokenizer Directory ---
    model_directory = "desklib/ai-text-detector-v1.01"

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = DesklibAIDetectionModel.from_pretrained(model_directory)

    # --- Set up device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Example Input text ---
    text_ai = "AI detection refers to the process of identifying whether a given piece of content, such as text, images, or audio, has been generated by artificial intelligence. This is achieved using various machine learning techniques, including perplexity analysis, entropy measurements, linguistic pattern recognition, and neural network classifiers trained on human and AI-generated data. Advanced AI detection tools assess writing style, coherence, and statistical properties to determine the likelihood of AI involvement. These tools are widely used in academia, journalism, and content moderation to ensure originality, prevent misinformation, and maintain ethical standards. As AI-generated content becomes increasingly sophisticated, AI detection methods continue to evolve, integrating deep learning models and ensemble techniques for improved accuracy."
    text_human = "It is estimated that a major part of the content in the internet will be generated by AI / LLMs by 2025. This leads to a lot of misinformation and credibility related issues. That is why if is important to have accurate tools to identify if a content is AI generated or human written"

    # --- Run prediction ---
    probability, predicted_label = predict_single_text(text_ai, model, tokenizer, device)
    print(f"Probability of being AI generated: {probability:.4f}")
    print(f"Predicted label: {'AI Generated' if predicted_label == 1 else 'Not AI Generated'}")

    probability, predicted_label = predict_single_text(text_human, model, tokenizer, device)
    print(f"Probability of being AI generated: {probability:.4f}")
    print(f"Predicted label: {'AI Generated' if predicted_label == 1 else 'Not AI Generated'}")

if __name__ == "__main__":
    main()
