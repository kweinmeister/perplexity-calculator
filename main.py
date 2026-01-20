import argparse
import os
import sys
from typing import Callable

import onnxruntime_genai as og
import yaml
from huggingface_hub import snapshot_download

import model  # Import new model module
import perplexity


def load_model(
    model_id: str = "onnx-community/SmolLM2-135M-ONNX",
) -> model.ModelContext:
    """
    Downloads the model-artifacts from Hugging Face and initializes the
    ModelContext with onnxruntime-genai Model and Tokenizer.

    Args:
        model_id (str): The Hugging Face repository ID.

    Returns:
        perplexity.ModelContext: The initialized model context.
    """
    print(f"Loading model from {model_id}...", file=sys.stderr)

    # Download the model artifacts
    model_path = snapshot_download(repo_id=model_id)
    print(f"Model artifacts available at: {model_path}", file=sys.stderr)

    # Determine the correct model directory
    # 1. Check if 'genai_config.json' exists in 'onnx' subdirectory (PREFERRED)
    if os.path.exists(os.path.join(model_path, "onnx", "genai_config.json")):
        print(
            f"Found 'genai_config.json' in 'onnx' subdir. Using {model_path}/onnx",
            file=sys.stderr,
        )
        model_path = os.path.join(model_path, "onnx")
    # 2. Check if genai_config.json exists in the root
    elif os.path.exists(os.path.join(model_path, "genai_config.json")):
        print(f"Found 'genai_config.json' in root. Using {model_path}", file=sys.stderr)
    # 3. Fallback: check if 'onnx' directory exists (legacy/simple structure?)
    elif os.path.isdir(os.path.join(model_path, "onnx")):
        print(
            f"Found 'onnx' subdirectory, using it as model path: {os.path.join(model_path, 'onnx')}",
            file=sys.stderr,
        )
        model_path = os.path.join(model_path, "onnx")

    # Load the model
    print("Initializing ONNX GenAI Model...", file=sys.stderr)
    genai_model = None
    tokenizer = None

    try:
        genai_model = og.Model(model_path)
    except Exception as e:
        print(f"Warning: Failed to load ONNX GenAI Model: {e}", file=sys.stderr)
        # Identify if we can proceed without og.Model (e.g. using tokenizers + raw ORT)
        pass

    # Initialize tokenizer
    print("Initializing Tokenizer...", file=sys.stderr)
    if genai_model:
        tokenizer = og.Tokenizer(genai_model)
    else:
        # Fallback to 'tokenizers' library
        try:
            from tokenizers import Tokenizer

            tokenizer_json = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                print(
                    f"Falling back to 'tokenizers' library using {tokenizer_json}",
                    file=sys.stderr,
                )
                raw_tokenizer = Tokenizer.from_file(tokenizer_json)

                # Create adapter to match OG interface
                class TokenizerAdapter:
                    def __init__(self, raw) -> None:
                        self.raw = raw

                    def encode(self, text):
                        import numpy as np

                        encoded = self.raw.encode(text)
                        return np.array(encoded.ids, dtype=np.int32)

                tokenizer = TokenizerAdapter(raw_tokenizer)
            else:
                print("Error: No 'tokenizer.json' found for fallback.", file=sys.stderr)
        except ImportError:
            print(
                "Error: 'tokenizers' library not installed and og.Model failed.",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Error loading fallback tokenizer: {e}", file=sys.stderr)

    if tokenizer is None:
        print("Failed to initialize any tokenizer. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Model and Tokenizer loaded successfully.", file=sys.stderr)
    return model.ModelContext(model_path, tokenizer, genai_model)


def load_test_data(filepath: str = "test_data.yaml") -> list[dict]:
    """
    Loads test data from a YAML file.
    """
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_test_texts(data: list[dict]) -> list[str]:
    """
    Extracts texts from test data.
    """
    return [item["text"] for item in data]


def run_batch_perplexity(
    context: model.ModelContext,
    texts: list[str],
    calc_fn: Callable[
        [model.ModelContext, str], float
    ] = perplexity.calculate_perplexity_onnxruntime_genai,
) -> list[float]:
    """
    Runs perplexity calculation for a batch of texts.
    Returns a list of calculated perplexity scores.
    """
    results: list[float] = []
    for text in texts:
        score = calc_fn(context, text)
        results.append(score)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate perplexity of a text using SmolLM2-135M-ONNX"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Input text to evaluate. If not provided, reads from stdin.",
    )
    parser.add_argument(
        "--model",
        default="onnx-community/SmolLM2-135M-ONNX",
        help="Hugging Face model ID",
    )

    parser.add_argument(
        "--method",
        default="onnxruntime_genai",
        help="Calculation method (suffix for calculate_perplexity_*)",
    )

    args = parser.parse_args()

    # Determine input text
    if args.text:
        text = args.text
    else:
        print("Reading text from stdin (Ctrl+D to finish)...", file=sys.stderr)
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("Error: Empty input text.", file=sys.stderr)
        sys.exit(1)

    # Resolve calculation method
    method_name = f"calculate_perplexity_{args.method}"
    calc_fn = getattr(perplexity, method_name, None)

    if calc_fn is None:
        print(
            f"Error: Method '{method_name}' not found in perplexity module.",
            file=sys.stderr,
        )
        print(
            f"Available methods: {[m for m in dir(perplexity) if m.startswith('calculate_perplexity_')]}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Load Model
        # This part separates model management from calculation
        context = load_model(args.model)

        # Calculate Perplexity
        print(
            f"Calculating perplexity for input length: {len(text)} chars using method: {args.method}...",
            file=sys.stderr,
        )
        ppl = calc_fn(context, text)

        print(f"\nPerplexity: {ppl:.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # For deeper debugging, you might want to log the traceback here
        sys.exit(1)


if __name__ == "__main__":
    main()
