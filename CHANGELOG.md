# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-11-15

### Fixed
- **Project Name Consistency**: Fixed all references from old "nanochat"/"nanoChat"/"NanoChat" naming to consistent "vibechat" branding throughout the codebase
  - Updated environment variable references from `NANOCHAT_BASE_DIR` to `VIBECHAT_BASE_DIR` in:
    - `vibechat/common.py`
    - `vibechat/report.py`
  - Updated ASCII banner in `vibechat/common.py` from "nanochat" to "vibechat"
  - Updated comments and docstrings in:
    - `vibechat/tokenizer.py`
    - `scripts/base_eval.py`
  - Updated UI text and titles in:
    - `scripts/chat_web.py` (Web Server description)
    - `scripts/chat_cli.py` (Interactive Mode header)
    - `vibechat/ui.html` (HTML page title)

- **Rust Edition Fix**: Corrected `rustbpe/Cargo.toml` edition from invalid "2024" to valid "2021"

### Notes
- The `uv.lock` file contains a legacy package name reference ("nanochat" at line 754) which will be automatically updated on the next `uv sync` or `uv lock` command execution

## Project Information

**vibechat** is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. This project is designed to be the capstone of the LLM101n course being developed by Eureka Labs.

For more information, see [README.md](README.md).
