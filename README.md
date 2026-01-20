# Perplexity Calculator

A Python tool to calculate the perplexity of text using ONNX-based Language Models. This project provides multiple methods to compute perplexity, allowing for performance comparisons between [`onnxruntime-genai`](https://github.com/microsoft/onnxruntime-genai) and standard [`onnxruntime`](https://github.com/microsoft/onnxruntime) approaches.

## Project Structure

- `main.py`: Entry point for the CLI.
- `perplexity.py`: Core logic for perplexity calculation methods.
- `model.py`: Model loading and context management.
- `benchmark_*.py`: Scripts for performance profiling.

## Features

- **Backend Comparison**: Tools to evaluate scoring performance across different ONNX runtimes.
- **Flexible Input**: Accepts text via command-line arguments or standard input (stdin).
- **Custom Models**: Support for specifying different Hugging Face model IDs (must be ONNX compatible).

### Calculation Backends

| Method | Description |
| :--- | :--- |
| `onnxruntime_optimized` | Single-pass calculation using ONNX Runtime (fastest for scoring). |
| `onnxruntime_genai` | (Default) Uses the `onnxruntime-genai` API to process tokens sequentially. |
| `onnxruntime_baseline` | Sequential execution with manual KV caching (for reference/debugging). |

### Recommended Model

For a lightweight and tested starting point, we recommend using [`SmolLM2-135M-ONNX`](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX).

## Installation

This project is managed with [`uv`](https://github.com/astral-sh/uv).

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd perplexity-calculator
    ```

2. **Install dependencies:**

    ```bash
    uv sync
    ```

## Usage

Calculate perplexity for a simple string:

```bash
uv run main.py "The quick brown fox jumps over the lazy dog"
```

Or pipe text via stdin:

```bash
echo "Hello world" | uv run main.py --method onnxruntime_optimized
```

### Advanced Options

```bash
usage: main.py [-h] [--model MODEL] [--method METHOD] [text]

options:
  -h, --help       show this help message and exit
  --model MODEL    Hugging Face model ID (default: onnx-community/SmolLM2-135M-ONNX)
  --method METHOD  Calculation method: 'onnxruntime_genai', 'onnxruntime_optimized',
                   or 'onnxruntime_baseline' (default: onnxruntime_genai)
```

## Benchmarking

### Time Benchmark

Uses `pytest-benchmark` to compare calculation methods:

```bash
uv run pytest benchmark_time.py
```

### Memory Benchmark

Uses `memray` to profile memory usage:

```bash
uv run memray run benchmark_memory.py
uv run memray summary memray-benchmark_memory.py.*.bin
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
