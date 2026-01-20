import argparse
import logging
import os
import sys
from collections.abc import Callable
from typing import cast

import numpy as np  # Moved from TokenizerAdapter.encode
import onnxruntime_genai as og
import yaml
from huggingface_hub import snapshot_download

import model  # Import new model module
import perplexity

# Configure logging to write to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def load_model(
    model_id: str = "onnx-community/SmolLM2-135M-ONNX",
) -> model.ModelContext:
    """Download the model-artifacts from Hugging Face and initializes the
    ModelContext with onnxruntime-genai Model and Tokenizer.

    Args:
        model_id (str): The Hugging Face repository ID.

    Returns:
        model.ModelContext: The initialized model context.

    """
    logger.info("Loading model from %s...", model_id)

    # Download the model artifacts
    model_path = snapshot_download(repo_id=model_id)
    logger.info("Model artifacts available at: %s", model_path)

    # Determine the correct model directory
    # 1. Check if 'genai_config.json' exists in 'onnx' subdirectory (PREFERRED)
    if os.path.exists(os.path.join(model_path, "onnx", "genai_config.json")):
        logger.info(
            "Found 'genai_config.json' in 'onnx' subdir. Using %s/onnx",
            model_path,
        )
        model_path = os.path.join(model_path, "onnx")
    # 2. Check if genai_config.json exists in the root
    elif os.path.exists(os.path.join(model_path, "genai_config.json")):
        logger.info("Found 'genai_config.json' in root. Using %s", model_path)
    # 3. Fallback: check if 'onnx' directory exists (legacy/simple structure?)
    elif os.path.isdir(os.path.join(model_path, "onnx")):
        logger.info(
            "Found 'onnx' subdirectory, using it as model path: %s",
            os.path.join(model_path, "onnx"),
        )
        model_path = os.path.join(model_path, "onnx")

    # Load the model
    logger.info("Initializing ONNX GenAI Model...")
    genai_model = None
    tokenizer = None

    try:
        genai_model = og.Model(model_path)
    except Exception as e:
        logger.warning("Failed to load ONNX GenAI Model: %s", e)
        # Identify if we can proceed without og.Model (e.g. using tokenizers + raw ORT)

    # Initialize tokenizer
    logger.info("Initializing Tokenizer...")
    if genai_model:
        tokenizer = og.Tokenizer(genai_model)
    else:
        # Fallback to 'tokenizers' library
        try:
            from tokenizers import Tokenizer

            tokenizer_json = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                logger.info(
                    "Falling back to 'tokenizers' library using %s",
                    tokenizer_json,
                )
                raw_tokenizer = Tokenizer.from_file(tokenizer_json)

                # Create adapter to match OG interface
                class TokenizerAdapter:
                    def __init__(self, raw) -> None:
                        self.raw = raw

                    def encode(self, text):
                        encoded = self.raw.encode(text)
                        return np.array(encoded.ids, dtype=np.int32)

                tokenizer = TokenizerAdapter(raw_tokenizer)
            else:
                logger.error("No 'tokenizer.json' found for fallback.")
        except ImportError:
            logger.exception("'tokenizers' library not installed and og.Model failed.")
        except Exception as e:
            logger.exception("Error loading fallback tokenizer: %s", e)

    if tokenizer is None:
        logger.error("Failed to initialize any tokenizer. Exiting.")
        sys.exit(1)

    logger.info("Model and Tokenizer loaded successfully.")
    return model.ModelContext(model_path, tokenizer, genai_model)


def load_test_data(filepath: str = "test_data.yaml") -> list[dict]:
    """Load test data from a YAML file."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def get_test_texts(data: list[dict]) -> list[str]:
    """Extract texts from test data."""
    return [item["text"] for item in data]


def run_batch_perplexity(
    context: model.ModelContext,
    texts: list[str],
    calc_fn: Callable[
        [model.ModelContext, str],
        float,
    ] = perplexity.calculate_perplexity_onnxruntime_genai,
) -> list[float]:
    """Run perplexity calculation for a batch of texts."""
    results: list[float] = []
    for text in texts:
        score = calc_fn(context, text)
        results.append(score)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate perplexity of a text using SmolLM2-135M-ONNX",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Input text to interpret. If not provided, reads from stdin.",
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
        logger.info("Reading text from stdin (Ctrl+D to finish)...")
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        logger.error("Empty input text.")
        sys.exit(1)

    # Resolve calculation method
    method_name = f"calculate_perplexity_{args.method}"
    raw_calc_fn = getattr(perplexity, method_name, None)

    if raw_calc_fn is None:
        logger.error("Method '%s' not found in perplexity module.", method_name)
        available = [
            m for m in dir(perplexity) if m.startswith("calculate_perplexity_")
        ]
        logger.info("Available methods: %s", available)
        sys.exit(1)

    calc_fn = cast("Callable[[model.ModelContext, str], float]", raw_calc_fn)
    try:
        # Load Model
        # This part separates model management from calculation
        context = load_model(args.model)

        # Calculate Perplexity
        logger.info(
            "Calculating perplexity for %d chars using method: %s...",
            len(text),
            args.method,
        )
        calc_fn(context, text)

        # Final result stays on stdout for CLI piping

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Error: %s", e)
        # For deeper debugging, you might want to log the traceback here
        sys.exit(1)


if __name__ == "__main__":
    main()
