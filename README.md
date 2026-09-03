Multi-Task NLP Transformer Implementation (GPT-2, 124M)
Author

Aakash Kumar Tiwari Roll No.: 25MA60R25 M.Tech in Computer Science and Data Processing Indian Institute of Technology Kharagpur

Project Supervisor

Prof. Adrijit Goswami

Project Overview

This project presents a from-scratch implementation of a 124M-parameter GPT-2-style decoder-only Transformer in PyTorch.

The main objective is to understand and implement the internal working of a modern Transformer-based language model rather than directly relying on the GPT2Model implementation provided by the Hugging Face Transformers library.

The project implements the complete core Transformer architecture, including:

Multi-head causal self-attention
Learned positional embeddings
Pre-Layer Normalization
GELU activation
Feed-forward neural networks
Residual connections
Weight tying between token embeddings and output projection
GPT-2 BPE tokenization
Pretrained GPT-2 weight loading
Temperature-based decoding
Top-k sampling
Downstream SMS spam classification
Partial Transformer fine-tuning
Automated LLM-based evaluation using LLaMA-3 through Ollama

The project therefore covers the complete workflow from Transformer architecture implementation → pretrained weight transfer → text generation → downstream fine-tuning → classification → automated LLM evaluation.

Project Motivation

Large Language Models such as GPT-2 are based on the Transformer architecture. Although libraries such as Hugging Face provide ready-to-use implementations, directly using these models hides many of the internal operations involved in building and executing a Transformer.

The motivation of this project is to implement the major components of GPT-2 manually and understand:

How tokenized text is converted into embeddings.
How positional information is incorporated.
How causal self-attention works.
How multiple attention heads process contextual information.
How Transformer blocks transform representations.
How pretrained model weights can be transferred into a custom implementation.
How a pretrained language model can be adapted for a downstream NLP classification task.
How generated model outputs can be automatically evaluated using another LLM.
System Architecture
                    ┌──────────────────────────┐
                    │       Input Text          │
                    │   "Every effort moves"    │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      BPE Tokenizer        │
                    │        tiktoken           │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Token Embeddings      │
                    │ + Positional Embeddings   │
                    └────────────┬──────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │        GPT-2 Transformer             │
              │                                       │
              │  ┌───────────────────────────────┐    │
              │  │ Multi-Head Causal Attention    │    │
              │  └───────────────┬─────────────────┘   │
              │                  │                     │
              │            Residual Connection          │
              │                  │                     │
              │  ┌───────────────▼─────────────────┐   │
              │  │      Layer Normalization         │   │
              │  └───────────────┬─────────────────┘    │
              │                  │                     │
              │  ┌───────────────▼─────────────────┐   │
              │  │     GELU Feed-Forward NN         │   │
              │  └───────────────┬─────────────────┘    │
              │                  │                     │
              │            Residual Connection          │
              │                  │                     │
              │              × 12 Blocks                │
              └──────────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Final LayerNorm      │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Weight-Tied Output      │
                    │      Projection Head      │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Next Token Prediction   │
                    └──────────────────────────┘

The same Transformer backbone can additionally be adapted for downstream classification:

                  SMS Message
                       │
                       ▼
                 BPE Tokenizer
                       │
                       ▼
              GPT-2 Transformer
                       │
                       ▼
             Final Representation
                       │
                       ▼
              Classification Head
                       │
                 ┌─────┴─────┐
                 ▼           ▼
               HAM          SPAM
GPT-2 Model Configuration

The main implementation follows the GPT-2 124M configuration.

Parameter	Value
Vocabulary Size	50,257
Context Length	1,024
Embedding Dimension	768
Transformer Layers	12
Attention Heads	12
Parameters	~124M
Activation	GELU
Architecture	Decoder-only Transformer
Normalization	LayerNorm
Attention Type	Causal Self-Attention

The implemented model contains approximately 124,439,808 parameters, verified directly during model initialization.

Core Transformer Architecture
1. Token Embeddings

The input text is first converted into integer token IDs using GPT-2's BPE tokenizer. Each token ID is mapped to a 768-dimensional embedding vector.

Token ID
   ↓
Embedding Lookup
   ↓
768-dimensional representation
2. Learned Positional Embeddings

Because self-attention itself does not inherently provide sequence order, learned positional embeddings are added to the token embeddings.

Token Representation = Token Embedding + Positional Embedding
3. Multi-Head Causal Self-Attention

The central component of the Transformer is multi-head self-attention. For every token, the model creates a Query (Q), Key (K), and Value (V).

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

A causal mask is applied so that a token cannot attend to future tokens:

Token 1 → can see Token 1
Token 2 → can see Token 1, Token 2
Token 3 → can see Token 1, Token 2, Token 3

This maintains the autoregressive nature of GPT-style language modeling.

4. Multi-Head Attention

Instead of a single attention mechanism, the model divides the representation into 12 attention heads, each learning different contextual relationships. Outputs from all heads are concatenated and projected back into the model embedding dimension.

5. Feed-Forward Network

Each Transformer block contains a position-wise feed-forward network:

Linear → GELU → Linear
6. Layer Normalization

The implementation follows a pre-LayerNorm Transformer block — normalization is applied before the main attention and feed-forward transformations.

7. Residual Connections

Residual connections are used around the attention and feed-forward components:

Input
  │
  ├───────────────┐
  │               │
  ▼               │
Attention          │
  │               │
  └───────► Add ◄──┘
              │
              ▼
           LayerNorm
              │
              ├───────────────┐
              │               │
              ▼               │
             FFN               │
              │               │
              └──────► Add ◄───┘

Residual connections help preserve information and allow deeper Transformer networks to be trained effectively.

8. Weight-Tied Output Head

The token embedding matrix is reused by the output projection layer (weight tying between Input Token Embedding and Output Projection), reducing the number of independent parameters.

Pretrained GPT-2 Weight Loading

A major component of the project is loading the official pretrained GPT-2 124M weights into the custom Transformer implementation. The Hugging Face transformers library is used only for obtaining the pretrained weight tensors — the forward pass itself is performed by the custom implementation in src/model.py.

Weight loading is implemented in src/weight_loader.py:

Official GPT-2 Checkpoint
          │
          ▼
Hugging Face Transformers
          │
          ▼
Extract pretrained tensors
          │
          ▼
Map parameter names
          │
          ▼
Convert tensor dimensions
          │
          ▼
Handle fused QKV weights
          │
          ▼
Copy weights into custom model
          │
          ▼
Custom 124M GPT-2 Model

The implementation specifically handles:

GPT-2 parameter mapping
Fused QKV → separate Query/Key/Value parameters
Conv1D weight transposition
Parameter shape conversion
Copying pretrained tensors into custom PyTorch modules

This allows the custom Transformer to reproduce pretrained GPT-2 behavior while keeping the model implementation independent of GPT2Model.

BPE Tokenization

The project uses GPT-2's Byte Pair Encoding tokenizer through tiktoken.

Raw Text → BPE Tokenization → Token IDs → GPT-2 Embedding Layer

The same tokenizer is used during text generation.

Text Generation

The project supports multiple decoding strategies:

Greedy Decoding — the highest-probability token is selected at every step:

Next Token = argmax(P(token))

Temperature Scaling — modifies the probability distribution before sampling. Lower temperature → more deterministic; higher temperature → more diverse.

Top-k Sampling — only the top-k highest-probability tokens are considered, and can be combined with temperature scaling.

Example:

bash
python scripts/run_generation.py \
    --prompt "Every effort moves you" \
    --use_pretrained \
    --top_k 50 \
    --temperature 0.8 \
    --max_new_tokens 40

The pretrained generation pipeline was successfully tested using the custom 124M model.

Downstream SMS Spam Classification

The project demonstrates how the pretrained GPT-2 Transformer can be adapted for a downstream NLP classification problem, using the UCI SMS Spam Collection dataset.

Dataset Download → Data Preparation → Class Balancing →
Train/Validation/Test Split → Tokenization →
Pretrained GPT-2 Backbone → Classification Head →
Partial Fine-Tuning → Evaluation

The dataset is balanced between ham and spam messages. Final split:

Split	Samples
Training	1045
Validation	149
Testing	300

A binary classification head is attached on top of the Transformer backbone. Fine-tuning keeps most of the pretrained backbone fixed while allowing the Classification Head + Last Transformer Block + Final LayerNorm to be updated — a parameter-efficient adaptation approach.

SMS Spam Classification Results

The model was trained for 10 epochs using the pretrained 124M GPT-2 model.

Epoch	Training Accuracy	Validation Accuracy
1	77.50%	80.00%
2	90.00%	85.00%
3	77.50%	82.50%
4	67.50%	80.00%
5	75.00%	85.00%
6	75.00%	82.50%
7	75.00%	87.50%
8	92.50%	97.50%
9	97.50%	97.50%
10	100.00%	97.50%

Final Test Accuracy: 98.67%

The trained classifier was saved as checkpoints/spam_classifier.pt.

Automated LLM-Based Evaluation

The project also contains an automated evaluation pipeline using LLaMA-3 + Ollama, implemented in src/evaluate_ollama.py.

Instruction / Input → Custom GPT-2 Model → Generated Response →
LLaMA-3 Judge → Score: 0–100 → Evaluation Results

The LLaMA-3 model is served locally using Ollama and provides a numerical score for generated responses. The evaluation process uses:

Temperature = 0
Fixed Seed
Strict Integer Output Format

This aims to make the scoring process reproducible under the same execution conditions. The final evaluation output contains individual response scores along with summary statistics (mean, minimum, maximum).

Ollama Evaluation Setup

Install and run Ollama locally and pull the LLaMA-3 model:

bash
ollama pull llama3
ollama serve

Then run:

bash
python scripts/run_ollama_eval.py \
    --alpaca_json data/alpaca_sample.json \
    --use_pretrained \
    --judge_model llama3

Evaluation results are written to eval_results.json.

Previously generated outputs can also be evaluated directly:

bash
python -m src.evaluate_ollama \
    --input my_outputs.json \
    --output eval_results.json

Expected input format:

json
[
  {
    "instruction": "Example instruction",
    "input": "Example input",
    "model_response": "Example response"
  }
]
Complete Project Workflow
                         ┌─────────────────────┐
                         │       Raw Text        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    BPE Tokenizer      │
                         │      tiktoken         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Token + Positional    │
                         │     Embeddings        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │          GPT-2 Transformer             │
                 │                                        │
                 │  12 × Transformer Blocks                │
                 │                                        │
                 │  ┌────────────────────────────────┐     │
                 │  │ Multi-Head Causal Self-Attn    │     │
                 │  ├────────────────────────────────┤     │
                 │  │ Residual + LayerNorm           │     │
                 │  ├────────────────────────────────┤     │
                 │  │ GELU Feed-Forward Network      │     │
                 │  └────────────────────────────────┘     │
                 └──────────────────┬─────────────────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │                             │
                      ▼                             ▼
             ┌──────────────────┐         ┌──────────────────┐
             │  Text Generation   │         │  Classification   │
             └────────┬──────────┘         └────────┬─────────┘
                      │                             │
                      ▼                             ▼
             Top-k / Temperature             SMS Spam Task
                      │                             │
                      ▼                             ▼
              Generated Text                  Fine-Tuning
                                                     │
                                                     ▼
                                              Test Prediction
                                                     │
                                                     ▼
                                              98.67% Accuracy


             Generated Text
                    │
                    ▼
             LLaMA-3 Judge
                    │
                    ▼
             Ollama Evaluation
                    │
                    ▼
             0–100 Quality Score
Project Components
File	Description
src/config.py	GPT-2 model configuration — vocabulary size, context length, embedding dimension, number of layers/heads, dropout config, and a lightweight test configuration for CPU experiments.
src/model.py	Core custom Transformer implementation: GPTModel, GPTClassifier, multi-head causal self-attention, Transformer blocks, feed-forward network, layer normalization, positional embeddings, residual connections, output projection, weight tying.
src/tokenizer.py	GPT-2 BPE tokenizer interface using tiktoken.
src/weight_loader.py	Loads pretrained GPT-2 weights from Hugging Face and maps them into the custom PyTorch implementation, handling parameter mapping, fused QKV conversion, Conv1D transpose conversion, and tensor shape compatibility.
src/generate.py	Text generation implementation — greedy decoding, temperature scaling, top-k sampling.
src/spam_dataset.py	UCI SMS Spam Collection download, data loading, class balancing, and train/validation/test splitting.
src/train_utils.py	Reusable training and evaluation utilities used by the fine-tuning pipeline.
src/evaluate_ollama.py	LLaMA-3/Ollama-based automated evaluation system — accepts generated outputs and produces numerical quality scores.
Project Structure
gpt2-multitask-nlp/
│
├── src/
│   ├── config.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── weight_loader.py
│   ├── generate.py
│   ├── spam_dataset.py
│   ├── train_utils.py
│   └── evaluate_ollama.py
│
├── scripts/
│   ├── run_generation.py
│   ├── finetune_spam.py
│   └── run_ollama_eval.py
│
├── data/
│   └── alpaca_sample.json
│
├── checkpoints/
│   └── spam_classifier.pt
│
├── requirements.txt
│
└── README.md
Technologies Used
Category	Tools
Programming Language	Python
Deep Learning	PyTorch
NLP / Language Models	GPT-2
Transformer Architecture	BPE Tokenization, tiktoken, Hugging Face Transformers
Model Evaluation	LLaMA-3, Ollama, LLM-as-a-Judge
Dataset	UCI SMS Spam Collection
Development Tools	Git, GitHub, VS Code, Jupyter / Google Colab
Installation

Create a virtual environment:

bash
python -m venv venv

Activate the environment:

bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

Install dependencies:

bash
pip install -r requirements.txt
Usage
1. Test Text Generation
bash
python scripts/run_generation.py \
    --prompt "Every effort moves you" \
    --use_pretrained \
    --top_k 50 \
    --temperature 0.8 \
    --max_new_tokens 40

The --use_pretrained argument loads the pretrained GPT-2 weights. Without this argument, the model can be initialized randomly for architecture and implementation testing.

2. Fine-Tune for SMS Spam Classification
bash
python scripts/finetune_spam.py \
    --use_pretrained \
    --epochs 10

The pipeline:

Downloads the UCI SMS Spam Collection dataset.
Balances the ham/spam classes.
Creates train/validation/test splits.
Loads pretrained GPT-2 weights.
Attaches a classification head.
Fine-tunes the selected Transformer components.
Evaluates training and validation performance.
Evaluates the final model on the test set.
Saves the fine-tuned classifier.

The experimentally obtained test accuracy for the 10-epoch run was 98.67%.

3. Run LLaMA-3 Automated Evaluation
bash
ollama pull llama3
ollama serve
bash
python scripts/run_ollama_eval.py \
    --alpaca_json data/alpaca_sample.json \
    --use_pretrained \
    --judge_model llama3
Hardware Considerations

The complete 124M-parameter model is computationally expensive for CPU-based fine-tuning. A GPU is recommended for downstream fine-tuning.

The architecture also contains a smaller test configuration, GPT2_CONFIG_TEST, which can be used for fast CPU-based smoke testing.

Recommended workflow:

Small Configuration → CPU Smoke Test → Verify Code Paths →
Full 124M Configuration → GPU → Generation / Fine-Tuning
Key Learning Outcomes

This project provides practical understanding of:

Transformer architecture
Decoder-only language models
GPT-2 architecture
Self-attention
Causal masking
Multi-head attention
Positional embeddings
Layer normalization
Residual connections
GELU activation
Weight tying
BPE tokenization
Pretrained model weight transfer
Transfer learning
NLP fine-tuning
Text generation
Sampling strategies
LLM-based evaluation
Local LLM inference using Ollama
Important Implementation Note

The Transformer forward pass is implemented using custom PyTorch modules in src/model.py. The Hugging Face transformers package is used for obtaining the pretrained GPT-2 weight tensors, not for replacing the custom Transformer forward implementation.

The project therefore demonstrates the internal implementation of a GPT-2-style Transformer while still making it possible to initialize the model with pretrained GPT-2 parameters.

The data/alpaca_sample.json file contains a small Alpaca-style instruction sample used by the automated evaluation pipeline. It is used for generating and evaluating responses and should not be interpreted as a full Alpaca training dataset.

Final Project Summary

This project implements a complete 124M-parameter GPT-2-style Transformer from scratch in PyTorch, transfers pretrained GPT-2 parameters into the custom architecture, performs autoregressive text generation using configurable decoding strategies, and adapts the pretrained Transformer for SMS spam classification.

The project further demonstrates automated response evaluation using a locally hosted LLaMA-3 model through Ollama, providing an end-to-end workflow covering Transformer implementation, pretrained weight transfer, NLP generation, downstream fine-tuning, classification, and LLM-based evaluation.

Final experimentally measured SMS spam classification performance: 98.67% Test Accuracy