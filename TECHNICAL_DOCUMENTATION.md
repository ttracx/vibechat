# vibechat Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Training Pipeline](#training-pipeline)
5. [Model Architecture](#model-architecture)
6. [Tokenization](#tokenization)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Development Guide](#development-guide)

## Project Overview

**vibechat** is a full-stack implementation of a ChatGPT-like LLM system designed to run on a single 8xH100 node. The project includes:

- Custom BPE tokenizer training (Rust + Python)
- Model pretraining (base model)
- Midtraining (conversation format adaptation)
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning (RL, optional)
- Web-based chat interface
- Comprehensive evaluation suite

**Key Features:**
- Clean, minimal, hackable codebase (~8K lines across 45 files)
- Dependency-lite (see pyproject.toml)
- Complete training pipeline in ~4 hours on 8xH100 ($100 tier)
- Model sizes: d20 (default), d26 (GPT-2 grade), customizable

## Architecture

### High-Level Pipeline

```
1. Tokenizer Training (rustbpe)
   ├── Download dataset shards
   ├── Train BPE tokenizer
   └── Export to tiktoken format

2. Base Model Pretraining
   ├── Initialize GPT model
   ├── Train on FineWeb-Edu dataset
   ├── Evaluate CORE metric
   └── Save checkpoint

3. Midtraining
   ├── Load base checkpoint
   ├── Train on conversation data
   ├── Adapt to special tokens
   └── Save checkpoint

4. Supervised Fine-Tuning
   ├── Load mid checkpoint
   ├── Fine-tune on task datasets
   ├── Evaluate on benchmarks
   └── Save checkpoint

5. (Optional) Reinforcement Learning
   ├── Load SFT checkpoint
   ├── RL training on GSM8K
   └── Save checkpoint

6. Inference & Deployment
   ├── Load trained checkpoint
   ├── Serve via FastAPI
   └── Web UI for chatting
```

### Directory Structure

```
vibechat/
├── vibechat/          # Core library modules
│   ├── gpt.py        # GPT model architecture
│   ├── tokenizer.py  # Tokenizer wrappers
│   ├── engine.py     # Inference engine with KV cache
│   ├── dataloader.py # Streaming data loader
│   ├── muon.py       # Muon optimizer
│   ├── adamw.py      # Distributed AdamW optimizer
│   ├── common.py     # Shared utilities
│   ├── dataset.py    # Dataset management
│   ├── checkpoint_manager.py  # Model saving/loading
│   ├── core_eval.py  # CORE metric evaluation
│   ├── loss_eval.py  # Bits-per-byte evaluation
│   ├── report.py     # Training report generation
│   ├── execution.py  # Sandboxed code execution
│   └── configurator.py  # CLI argument parsing
│
├── scripts/          # Training and evaluation scripts
│   ├── base_train.py # Base model training
│   ├── base_eval.py  # Base model evaluation
│   ├── base_loss.py  # Loss evaluation
│   ├── mid_train.py  # Midtraining
│   ├── chat_sft.py   # Supervised fine-tuning
│   ├── chat_rl.py    # Reinforcement learning
│   ├── chat_eval.py  # Chat model evaluation
│   ├── chat_web.py   # Web server
│   ├── chat_cli.py   # CLI chat interface
│   ├── tok_train.py  # Tokenizer training
│   └── tok_eval.py   # Tokenizer evaluation
│
├── tasks/            # Evaluation task implementations
│   ├── common.py     # Task base class
│   ├── arc.py        # ARC benchmark
│   ├── mmlu.py       # MMLU benchmark
│   ├── gsm8k.py      # GSM8K math problems
│   ├── humaneval.py  # HumanEval code generation
│   └── smoltalk.py   # SmolTalk conversations
│
├── rustbpe/          # Rust BPE tokenizer
│   ├── src/lib.rs    # Core BPE implementation
│   └── Cargo.toml    # Rust dependencies
│
├── tests/            # Test suite
│   └── test_rustbpe.py
│
├── speedrun.sh       # Full training pipeline script
├── pyproject.toml    # Python dependencies
└── README.md         # User documentation
```

## Core Components

### 1. GPT Model (`vibechat/gpt.py`)

**Architecture Features:**
- Rotary position embeddings (RoPE) - no learned positional embeddings
- QK normalization for stable training
- Untied embedding/unembedding weights
- ReLU² activation in MLP layers
- RMSNorm (no learnable parameters)
- No bias in linear layers
- Multi-Query Attention (MQA) support

**Configuration:**
```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024    # Max context length
    vocab_size: int = 50304     # Vocabulary size
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 6             # Number of query heads
    n_kv_head: int = 6          # Number of KV heads (for MQA)
    n_embd: int = 768           # Model dimension
```

**Key Methods:**
- `forward()` - Forward pass with optional loss computation
- `generate()` - Naive autoregressive generation
- `setup_optimizers()` - Create Muon + AdamW optimizers
- `estimate_flops()` - Calculate FLOPs per token

### 2. Tokenizer (`vibechat/tokenizer.py`)

Two implementations available:

**RustBPETokenizer (Recommended):**
- Fast training via Rust
- Efficient inference via tiktoken
- GPT-4 style regex splitting
- ~65K vocabulary (default)

**HuggingFaceTokenizer (Alternative):**
- Pure Python training and inference
- More features but slower

**Special Tokens:**
```python
SPECIAL_TOKENS = [
    "<|bos|>",              # Beginning of sequence
    "<|user_start|>",       # User message start
    "<|user_end|>",         # User message end
    "<|assistant_start|>",  # Assistant message start
    "<|assistant_end|>",    # Assistant message end
    "<|python_start|>",     # Python tool call start
    "<|python_end|>",       # Python tool call end
    "<|output_start|>",     # Tool output start
    "<|output_end|>",       # Tool output end
]
```

### 3. Inference Engine (`vibechat/engine.py`)

**Features:**
- KV cache management for efficient generation
- Batch generation with multiple samples
- Tool use (calculator integration)
- Streaming support
- Temperature and top-k sampling

**KV Cache:**
- Dynamically sized
- Supports prefill + decode pattern
- Batch expansion for multiple samples

### 4. Optimizers

**Muon (`vibechat/muon.py`):**
- Used for linear layer parameters
- Newton-Schulz orthogonalization
- SGD with momentum
- Aspect-ratio scaled learning rate

**DistAdamW (`vibechat/adamw.py`):**
- Used for embedding and unembedding layers
- ZeRO-2 style sharding
- Gradient reduction across ranks
- 1/√d_model learning rate scaling

### 5. Data Pipeline (`vibechat/dataloader.py`)

**Streaming Data Loader:**
- Reads from Parquet files
- Tokenizes on-the-fly
- Distributed across ranks
- Memory-efficient buffering
- No disk caching needed

**Dataset (`vibechat/dataset.py`):**
- FineWeb-Edu 100B dataset
- On-demand downloading
- 1823 shards × ~250M chars each
- Parquet format for efficiency

## Training Pipeline

### 1. Tokenizer Training

```bash
python -m scripts.tok_train \
    --vocab_size=65536 \
    --max_chars=2000000000
```

**Process:**
1. Stream text from dataset
2. Split by GPT-4 regex pattern
3. Train BPE merges (Rust)
4. Export to tiktoken format
5. Compute token byte lengths for evaluation

### 2. Base Model Pretraining

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=32
```

**Hyperparameters (d20 model):**
- Model: 561M parameters
- Tokens: 11.2B (20× params, Chinchilla optimal)
- Batch size: 524K tokens
- Learning rate: 0.02 (Muon), 0.2 (embedding), 0.004 (unembedding)
- Sequence length: 2048
- Training time: ~2 hours on 8xH100

**Evaluation:**
- Validation bits-per-byte (bpb)
- CORE metric every 2000 steps
- Sampling every 2000 steps

### 3. Midtraining

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.mid_train
```

**Purpose:**
- Adapt to conversation format
- Learn special tokens
- Train on multi-turn dialogues
- Introduce tool use patterns

**Dataset:** SmolTalk + synthetic data

### 4. Supervised Fine-Tuning

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft
```

**Datasets:**
- SmolTalk (conversations)
- GSM8K (math problems)
- HumanEval (code generation)
- ARC (reasoning)
- MMLU (knowledge)

**Training:**
- Mask loss on user messages
- Compute loss only on assistant responses
- Handle multi-part messages (text + tool calls)

### 5. Reinforcement Learning (Optional)

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl
```

**Algorithm:** Policy gradient (REINFORCE)
- Task: GSM8K math problems
- Reward: Binary (correct/incorrect)
- Baseline: Moving average
- Sample K completions per prompt

## Model Architecture

### Transformer Block

```
Input
  ↓
RMSNorm
  ↓
Multi-Query Attention (MQA)
  ├── Q, K, V projections
  ├── RoPE embeddings
  ├── QK normalization
  └── Attention + output projection
  ↓
Residual connection (+)
  ↓
RMSNorm
  ↓
MLP
  ├── Linear (expand 4x)
  ├── ReLU²
  └── Linear (project back)
  ↓
Residual connection (+)
  ↓
Output
```

### Attention Details

**Multi-Query Attention:**
- Q heads: `n_head` (e.g., 6)
- KV heads: `n_kv_head` (e.g., 6, or 1 for true MQA)
- Head dim: `n_embd / n_head`

**RoPE Embeddings:**
- Base frequency: 10,000
- Applied to Q and K
- Relative positional encoding
- Cached and reused

**QK Normalization:**
- Apply RMSNorm to queries and keys
- Improves training stability
- No learnable parameters

### FLOPs Estimation

FLOPs per token ≈ `6N + 12Lhqt`

Where:
- N = total parameters (excluding embeddings)
- L = number of layers
- h = number of heads
- q = head dimension
- t = sequence length

## Tokenization

### Training Process

1. **Text Splitting:**
   - Regex pattern: `GPT4_PATTERN`
   - Splits on word boundaries, numbers, special chars
   - Preserves whitespace patterns

2. **BPE Merges:**
   - Start with 256 byte tokens
   - Iteratively merge most frequent pairs
   - Uses heap for efficient pair selection
   - Parallel processing in Rust

3. **Export:**
   - Convert to tiktoken format
   - Add special tokens
   - Save as pickle file

### Inference

```python
tokenizer = get_tokenizer()
tokens = tokenizer.encode("Hello, world!", prepend="<|bos|>")
text = tokenizer.decode(tokens)
```

### Conversation Rendering

```python
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
}
ids, mask = tokenizer.render_conversation(conversation)
# mask=1 where assistant should predict, 0 elsewhere
```

## Evaluation

### CORE Metric

From the DCLM paper, evaluates on:
- Multiple choice (MC)
- Schema completion
- Language modeling (LM)

**Tasks:**
- ARC-Easy, ARC-Challenge
- HellaSwag
- MMLU
- PIQA
- Winogrande
- And more...

**Implementation:** `vibechat/core_eval.py`

### Bits Per Byte (BPB)

- Tokenization-invariant metric
- Normalizes loss by token byte length
- Used for validation loss
- Lower is better

### Chat Evaluation

**Generative Tasks:**
- GSM8K: Math problems with calculator
- HumanEval: Code generation

**Metrics:**
- Exact match accuracy
- Pass@1 for code

## Deployment

### Web Interface

```bash
python -m scripts.chat_web \
    --source sft \
    --port 8000 \
    --host 0.0.0.0
```

**Features:**
- FastAPI backend
- Server-Sent Events (SSE) for streaming
- Single-file HTML/CSS/JS UI
- ChatGPT-like interface

**API Endpoints:**
- `GET /` - Serve UI
- `POST /chat/completions` - Chat completion
- `GET /health` - Health check
- `GET /logo.svg` - Logo

### CLI Interface

```bash
python -m scripts.chat_cli \
    --prompt "Why is the sky blue?"
```

Interactive mode:
```bash
python -m scripts.chat_cli
```

## Development Guide

### Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Running Tests

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

### Training Custom Models

**Small model (d12):**
```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=12 \
    --device_batch_size=64
```

**Larger model (d26, GPT-2 grade):**
```bash
# Download more data
python -m vibechat.dataset -n 450

# Train with smaller batch size to fit in memory
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=26 \
    --device_batch_size=16
```

### Memory Management

If OOM:
1. Reduce `--device_batch_size`
2. Code auto-adjusts gradient accumulation
3. Trading parallel compute for sequential

### Logging with WandB

```bash
# Login
wandb login

# Run with logging
WANDB_RUN=my-run bash speedrun.sh
```

### Configuration

All scripts support CLI overrides:
```bash
python script.py --arg1=value1 --arg2=value2
```

See `vibechat/configurator.py` for details.

### Distributed Training

**Single GPU:**
```bash
python -m scripts.base_train
```

**Multi-GPU (torchrun):**
```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train
```

**Environment variables:**
- `RANK` - Global rank
- `LOCAL_RANK` - Local rank on node
- `WORLD_SIZE` - Total number of processes

## Performance Optimization

### Compilation

All models use `torch.compile()` for ~2x speedup.

### Precision

- Embeddings: BF16
- Activations: BF16 (via autocast)
- Matmuls: TF32 (via `torch.set_float32_matmul_precision("high")`)

### Memory

- Gradient checkpointing: Not used (models fit in memory)
- Mixed precision: BF16 for forward/backward
- KV cache: Dynamic sizing

### Data Loading

- Streaming from disk
- Tokenization on-the-fly
- Background prefetching
- No data caching

## Troubleshooting

### Common Issues

**1. OOM during training:**
- Reduce `--device_batch_size`
- Use smaller model (lower `--depth`)

**2. Slow data loading:**
- Increase `--tokenizer_threads`
- Check disk I/O

**3. Compilation errors:**
- Disable with `--compile=False`
- Check PyTorch version

**4. Distributed hanging:**
- Check NCCL_DEBUG=INFO
- Verify network connectivity

### Debug Mode

```bash
WANDB_RUN=dummy python -m scripts.base_train --num_iterations=10
```

## Best Practices

1. **Always use `torchrun` for multi-GPU**
2. **Monitor MFU (Model FLOPs Utilization)** - should be >40%
3. **Check validation loss regularly**
4. **Save checkpoints frequently**
5. **Use `screen` or `tmux` for long runs**
6. **Enable wandb logging for production runs**

## References

- [LLM101n Course](https://github.com/karpathy/LLM101n) - Course this is designed for
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)
- [DCLM Paper](https://arxiv.org/abs/2406.11794) - CORE metric
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Chinchilla Paper](https://arxiv.org/abs/2203.15556) - Scaling laws

## License

See LICENSE file in repository.

## Contributing

This is primarily an educational project for LLM101n. See main README for contribution guidelines.
