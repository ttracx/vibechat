# vibechat API Reference

## Module Index

- [vibechat.gpt](#vibechatgpt) - GPT model implementation
- [vibechat.tokenizer](#vibechattokenizer) - Tokenizer wrappers
- [vibechat.engine](#vibechatengine) - Inference engine
- [vibechat.dataloader](#vibechatdataloader) - Data loading utilities
- [vibechat.checkpoint_manager](#vibechatcheckpoint_manager) - Model checkpointing
- [vibechat.common](#vibechatcommon) - Common utilities
- [vibechat.muon](#vibechatmuon) - Muon optimizer
- [vibechat.adamw](#vibechatadamw) - Distributed AdamW
- [vibechat.core_eval](#vibechatcore_eval) - CORE metric evaluation
- [vibechat.loss_eval](#vibechatloss_eval) - Loss evaluation
- [vibechat.execution](#vibechatexecution) - Code execution sandbox
- [vibechat.report](#vibechatreport) - Report generation

---

## vibechat.gpt

### Classes

#### `GPTConfig`

Configuration dataclass for GPT model.

**Attributes:**
- `sequence_len: int` - Maximum sequence length (default: 1024)
- `vocab_size: int` - Vocabulary size (default: 50304)
- `n_layer: int` - Number of transformer layers (default: 12)
- `n_head: int` - Number of query attention heads (default: 6)
- `n_kv_head: int` - Number of key/value heads for MQA (default: 6)
- `n_embd: int` - Model embedding dimension (default: 768)

#### `GPT`

Main GPT model class.

**Constructor:**
```python
model = GPT(config: GPTConfig)
```

**Methods:**

##### `forward(idx, targets=None, kv_cache=None, loss_reduction='mean')`

Forward pass through the model.

**Parameters:**
- `idx: torch.Tensor` - Input token IDs of shape (B, T)
- `targets: torch.Tensor | None` - Target token IDs for loss computation
- `kv_cache: KVCache | None` - KV cache for efficient inference
- `loss_reduction: str` - Loss reduction method ('mean', 'none')

**Returns:**
- If `targets` is None: logits of shape (B, T, vocab_size)
- If `targets` is provided: scalar loss value

##### `generate(tokens, max_tokens, temperature=1.0, top_k=None, seed=42)`

Generate tokens autoregressively (naive implementation).

**Parameters:**
- `tokens: list[int]` - Initial token sequence
- `max_tokens: int` - Maximum tokens to generate
- `temperature: float` - Sampling temperature (0 = greedy)
- `top_k: int | None` - Top-k sampling parameter
- `seed: int` - Random seed

**Yields:**
- `int` - Next generated token

##### `init_weights()`

Initialize model weights using custom initialization scheme.

##### `setup_optimizers(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0)`

Create optimizers for different parameter groups.

**Parameters:**
- `unembedding_lr: float` - Learning rate for lm_head
- `embedding_lr: float` - Learning rate for token embeddings
- `matrix_lr: float` - Learning rate for linear layers (Muon)
- `weight_decay: float` - Weight decay for Adam parameters

**Returns:**
- `list[Optimizer]` - List containing [adamw_optimizer, muon_optimizer]

##### `estimate_flops()`

Estimate FLOPs per token for the model.

**Returns:**
- `int` - Estimated FLOPs per forward pass per token

##### `get_device()`

Get the device the model is on.

**Returns:**
- `torch.device` - Device of the model

---

## vibechat.tokenizer

### Classes

#### `RustBPETokenizer`

Tokenizer using Rust for training and tiktoken for inference.

**Class Methods:**

##### `train_from_iterator(text_iterator, vocab_size)`

Train tokenizer from text iterator.

**Parameters:**
- `text_iterator: Iterator[str]` - Iterator yielding text strings
- `vocab_size: int` - Target vocabulary size

**Returns:**
- `RustBPETokenizer` - Trained tokenizer instance

##### `from_directory(tokenizer_dir)`

Load tokenizer from directory.

**Parameters:**
- `tokenizer_dir: str` - Directory containing tokenizer.pkl

**Returns:**
- `RustBPETokenizer` - Loaded tokenizer

##### `from_pretrained(tiktoken_name)`

Load pretrained tiktoken tokenizer (e.g., "gpt2").

**Parameters:**
- `tiktoken_name: str` - Name of tiktoken encoding

**Returns:**
- `RustBPETokenizer` - Loaded tokenizer

**Instance Methods:**

##### `encode(text, prepend=None, append=None, num_threads=8)`

Encode text to token IDs.

**Parameters:**
- `text: str | list[str]` - Text or list of texts to encode
- `prepend: int | str | None` - Token to prepend
- `append: int | str | None` - Token to append
- `num_threads: int` - Number of threads for batch encoding

**Returns:**
- `list[int] | list[list[int]]` - Token IDs

##### `decode(ids)`

Decode token IDs to text.

**Parameters:**
- `ids: list[int]` - Token IDs to decode

**Returns:**
- `str` - Decoded text

##### `render_conversation(conversation, max_tokens=2048)`

Tokenize a conversation with special tokens.

**Parameters:**
- `conversation: dict` - Conversation dict with "messages" list
- `max_tokens: int` - Maximum sequence length

**Returns:**
- `tuple[list[int], list[int]]` - (token_ids, mask)
  - mask=1 for assistant tokens to train on, 0 otherwise

##### `render_for_completion(conversation)`

Render conversation priming assistant for completion (RL).

**Parameters:**
- `conversation: dict` - Conversation dict

**Returns:**
- `list[int]` - Token IDs ending with assistant_start token

##### `get_vocab_size()`

Get vocabulary size.

**Returns:**
- `int` - Vocabulary size

##### `get_bos_token_id()`

Get BOS token ID.

**Returns:**
- `int` - BOS token ID

##### `encode_special(text)`

Encode a special token by exact match.

**Parameters:**
- `text: str` - Special token string

**Returns:**
- `int` - Token ID

##### `save(tokenizer_dir)`

Save tokenizer to directory.

**Parameters:**
- `tokenizer_dir: str` - Directory to save to

#### `HuggingFaceTokenizer`

Alternative tokenizer using HuggingFace tokenizers library.

Similar API to `RustBPETokenizer` with slightly different implementation.

### Functions

##### `get_tokenizer()`

Get the tokenizer from the default location.

**Returns:**
- `RustBPETokenizer` - Loaded tokenizer

##### `get_token_bytes(device="cpu")`

Load token byte lengths tensor.

**Parameters:**
- `device: str` - Device to load tensor on

**Returns:**
- `torch.Tensor` - Tensor of shape (vocab_size,) with byte lengths

---

## vibechat.engine

### Classes

#### `KVCache`

Key-Value cache for efficient transformer inference.

**Constructor:**
```python
cache = KVCache(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_layers: int
)
```

**Methods:**

##### `insert_kv(layer_idx, k, v)`

Insert keys and values for a layer.

**Parameters:**
- `layer_idx: int` - Layer index
- `k: torch.Tensor` - Keys of shape (B, H, T, D)
- `v: torch.Tensor` - Values of shape (B, H, T, D)

**Returns:**
- `tuple[torch.Tensor, torch.Tensor]` - Full cached (keys, values)

##### `reset()`

Reset cache position to 0.

##### `get_pos()`

Get current cache position.

**Returns:**
- `int` - Current position

##### `prefill(other)`

Prefill cache from another cache.

**Parameters:**
- `other: KVCache` - Source cache to copy from

#### `Engine`

Inference engine with advanced features.

**Constructor:**
```python
engine = Engine(model: GPT, tokenizer: RustBPETokenizer)
```

**Methods:**

##### `generate(tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42)`

Generate tokens with KV cache and tool use.

**Parameters:**
- `tokens: list[int]` - Initial token sequence
- `num_samples: int` - Number of parallel samples
- `max_tokens: int | None` - Maximum tokens to generate
- `temperature: float` - Sampling temperature
- `top_k: int | None` - Top-k sampling
- `seed: int` - Random seed

**Yields:**
- `tuple[list[int], list[int]]` - (token_column, token_masks)
  - token_column: Next token for each sample
  - token_masks: 1=sampled, 0=forced (tool output)

##### `generate_batch(tokens, num_samples=1, **kwargs)`

Non-streaming batch generation.

**Parameters:**
- Same as `generate()`

**Returns:**
- `tuple[list[list[int]], list[list[int]]]` - (results, masks)
  - results: Token sequences for each sample
  - masks: Mask sequences (1=sampled, 0=forced)

### Functions

##### `sample_next_token(logits, rng, temperature=1.0, top_k=None)`

Sample next token from logits.

**Parameters:**
- `logits: torch.Tensor` - Logits of shape (B, vocab_size)
- `rng: torch.Generator` - Random number generator
- `temperature: float` - Sampling temperature
- `top_k: int | None` - Top-k filtering

**Returns:**
- `torch.Tensor` - Sampled token IDs of shape (B, 1)

---

## vibechat.dataloader

### Functions

##### `tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128)`

Create streaming data loader with on-the-fly tokenization.

**Parameters:**
- `B: int` - Batch size
- `T: int` - Sequence length
- `split: str` - "train" or "val"
- `tokenizer_threads: int` - Number of tokenizer threads
- `tokenizer_batch_size: int` - Batch size for tokenization

**Yields:**
- `tuple[torch.Tensor, torch.Tensor]` - (inputs, targets)
  - inputs: Token IDs of shape (B, T)
  - targets: Target token IDs of shape (B, T)

---

## vibechat.checkpoint_manager

### Functions

##### `save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data)`

Save model checkpoint.

**Parameters:**
- `checkpoint_dir: str` - Directory to save checkpoint
- `step: int` - Training step number
- `model_data: dict` - Model state dict
- `optimizer_data: dict | None` - Optimizer state dict
- `meta_data: dict` - Metadata (config, hyperparameters, etc.)

##### `load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)`

Load model checkpoint.

**Parameters:**
- `checkpoint_dir: str` - Directory containing checkpoint
- `step: int` - Step number to load
- `device: torch.device` - Device to load to
- `load_optimizer: bool` - Whether to load optimizer state

**Returns:**
- `tuple[dict, dict | None, dict]` - (model_data, optimizer_data, meta_data)

##### `load_model(source, device, phase, model_tag=None, step=None)`

Load model from standard vibechat directory structure.

**Parameters:**
- `source: str` - Model source: "base", "mid", "sft", or "rl"
- `device: torch.device` - Device to load to
- `phase: str` - "train" or "eval"
- `model_tag: str | None` - Model tag (e.g., "d20")
- `step: int | None` - Step to load (None = latest)

**Returns:**
- `tuple[GPT, RustBPETokenizer, dict]` - (model, tokenizer, meta_data)

##### `find_largest_model(checkpoint_dir)`

Find largest model in checkpoint directory.

**Parameters:**
- `checkpoint_dir: str` - Checkpoint directory

**Returns:**
- `str` - Model tag of largest model

##### `find_last_step(checkpoint_dir)`

Find latest checkpoint step in directory.

**Parameters:**
- `checkpoint_dir: str` - Checkpoint directory

**Returns:**
- `int` - Latest step number

---

## vibechat.common

### Functions

##### `compute_init()`

Initialize distributed training environment.

**Returns:**
- `tuple[bool, int, int, int, torch.device]` - (ddp, rank, local_rank, world_size, device)

##### `compute_cleanup()`

Clean up distributed training environment.

##### `get_dist_info()`

Get distributed training info.

**Returns:**
- `tuple[bool, int, int, int]` - (ddp, rank, local_rank, world_size)

##### `get_base_dir()`

Get base directory for vibechat artifacts.

**Returns:**
- `str` - Base directory path (default: ~/.cache/vibechat)

##### `print0(*args, **kwargs)`

Print only on rank 0.

##### `print_banner()`

Print vibechat ASCII banner.

### Classes

#### `DummyWandb`

Dummy wandb object for non-logging mode.

---

## vibechat.muon

### Classes

#### `Muon`

Muon optimizer (SGD + momentum + Newton-Schulz orthogonalization).

**Constructor:**
```python
optimizer = Muon(
    params: Iterable[Tensor],
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5
)
```

**Methods:**

##### `step()`

Perform single optimization step.

#### `DistMuon`

Distributed version of Muon with gradient sharding.

**Constructor:**
```python
optimizer = DistMuon(
    params: Iterable[Tensor],
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5
)
```

---

## vibechat.adamw

### Classes

#### `DistAdamW`

Distributed AdamW optimizer with ZeRO-2 style sharding.

**Constructor:**
```python
optimizer = DistAdamW(
    param_groups: list[dict],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01
)
```

**Methods:**

##### `step()`

Perform single optimization step with gradient reduction.

---

## vibechat.core_eval

### Functions

##### `evaluate_task(model, tokenizer, data, device, task_meta)`

Evaluate model on a CORE task.

**Parameters:**
- `model: GPT` - Model to evaluate
- `tokenizer: RustBPETokenizer` - Tokenizer
- `data: list[dict]` - Task dataset
- `device: torch.device` - Device
- `task_meta: dict` - Task metadata:
  - `task_type: str` - "multiple_choice", "schema", or "language_modeling"
  - `num_fewshot: int` - Number of few-shot examples
  - `continuation_delimiter: str` - Delimiter between context and continuation

**Returns:**
- `float` - Accuracy (0-1)

##### `evaluate_example(idx, model, tokenizer, data, device, task_meta)`

Evaluate single example.

**Parameters:**
- `idx: int` - Example index
- Others same as `evaluate_task`

**Returns:**
- `bool` - True if correct, False otherwise

---

## vibechat.loss_eval

### Functions

##### `evaluate_bpb(model, batches, steps, token_bytes)`

Evaluate bits-per-byte metric.

**Parameters:**
- `model: GPT` - Model to evaluate
- `batches: Iterator` - Data batch iterator
- `steps: int` - Number of evaluation steps
- `token_bytes: torch.Tensor` - Token byte length tensor

**Returns:**
- `float` - Bits per byte

---

## vibechat.execution

### Classes

#### `ExecutionResult`

Result of code execution.

**Attributes:**
- `success: bool` - Whether execution succeeded
- `stdout: str` - Captured stdout
- `stderr: str` - Captured stderr
- `error: str | None` - Error message if failed
- `timeout: bool` - Whether execution timed out
- `memory_exceeded: bool` - Whether memory limit was exceeded

### Functions

##### `execute_code(code, timeout=5.0, maximum_memory_bytes=256*1024*1024)`

Execute Python code in sandbox.

**Parameters:**
- `code: str` - Python code to execute
- `timeout: float` - Execution timeout in seconds
- `maximum_memory_bytes: int | None` - Memory limit in bytes

**Returns:**
- `ExecutionResult` - Execution result

---

## vibechat.report

### Classes

#### `Report`

Training report generator.

**Constructor:**
```python
report = Report(report_dir: str)
```

**Methods:**

##### `log(section, data)`

Log data to report section.

**Parameters:**
- `section: str` - Section name
- `data: list` - List of dicts or strings to log

**Returns:**
- `str` - Path to written file

##### `generate()`

Generate final report from all sections.

**Returns:**
- `str` - Path to generated report

##### `reset()`

Reset report and write header.

### Functions

##### `get_report()`

Get report instance (rank 0 only, others get dummy).

**Returns:**
- `Report | DummyReport` - Report instance

---

## Task API

### Base Class

#### `Task`

Base class for evaluation tasks.

**Methods:**

##### `num_examples()`

Get number of examples in task.

**Returns:**
- `int` - Number of examples

##### `get_example(index)`

Get example at index.

**Parameters:**
- `index: int` - Example index

**Returns:**
- `dict` - Conversation dict

##### `evaluate(conversation, assistant_response)`

Evaluate assistant response.

**Parameters:**
- `conversation: dict` - Full conversation
- `assistant_response: str` - Model's generated response

**Returns:**
- `int` - 1 if correct, 0 if incorrect

##### `reward(conversation, assistant_response)`

Compute reward for RL.

**Parameters:**
- Same as `evaluate()`

**Returns:**
- `float` - Reward value

**Properties:**

##### `eval_type`

Get evaluation type.

**Returns:**
- `str` - "multiple_choice" or "generative"

---

## Examples

### Training a Model

```python
from vibechat.gpt import GPT, GPTConfig
from vibechat.dataloader import tokenizing_distributed_data_loader
from vibechat.common import compute_init

# Initialize
ddp, rank, local_rank, world_size, device = compute_init()

# Create model
config = GPTConfig(n_layer=12, n_embd=768)
model = GPT(config).to(device)

# Create optimizer
optimizers = model.setup_optimizers()

# Create data loader
train_loader = tokenizing_distributed_data_loader(32, 1024, "train")

# Training loop
for x, y in train_loader:
    loss = model(x, y)
    loss.backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad()
```

### Inference

```python
from vibechat.checkpoint_manager import load_model
from vibechat.engine import Engine
from vibechat.common import compute_init

# Load model
_, _, _, _, device = compute_init()
model, tokenizer, _ = load_model("sft", device, "eval")

# Create engine
engine = Engine(model, tokenizer)

# Generate
tokens = tokenizer.encode("Hello, world!", prepend="<|bos|>")
results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=50)
print(tokenizer.decode(results[0]))
```

### Evaluation

```python
from vibechat.checkpoint_manager import load_model
from tasks.gsm8k import GSM8K

# Load model and task
model, tokenizer, _ = load_model("sft", device, "eval")
task = GSM8K("main", "test")

# Evaluate
correct = 0
for i in range(task.num_examples()):
    conversation = task.get_example(i)
    # ... generate response ...
    if task.evaluate(conversation, response):
        correct += 1

accuracy = correct / task.num_examples()
```
