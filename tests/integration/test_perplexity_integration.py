import logging
import os
from collections.abc import Callable

import pytest

import model
import perplexity

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "calc_fn",
    [
        perplexity.calculate_perplexity_onnxruntime_genai,
        perplexity.calculate_perplexity_onnxruntime_baseline,
        perplexity.calculate_perplexity_onnxruntime_optimized,
    ],
)
def test_perplexity_functions(
    loaded_model: tuple[model.ModelContext, str],
    calc_fn: Callable[[model.ModelContext, str], float],
    test_case: dict,
) -> None:
    """Test that calculates perplexity and checks against expectations (if any).
    This creates a matrix of (calculation functions) x (test examples).
    """
    context, model_id = loaded_model
    description = test_case.get("description", "Unknown")
    text = test_case.get("text")
    assert text is not None, (
        f"Test case '{description}' is missing required 'text' field"
    )
    expected_dict = test_case.get("perplexity")

    # Perform calculation
    actual = calc_fn(context, text)
    method_name = getattr(calc_fn, "__name__", "Unknown")
    logger.info("Method: %s", method_name)
    logger.info("Example: %s", description)
    logger.info("Perplexity: %.4f", actual)

    # Basic assertion
    assert isinstance(actual, float)

    # Consistency check (if model-specific expectation exists)
    if expected_dict and model_id in expected_dict:
        entry = expected_dict[model_id]

        expected = None
        if isinstance(entry, dict):
            # Check which file from the expected list exists in the model path

            found_file = False
            for filename, val in entry.items():
                # We need to guess where the file is (root or onnx subdir)
                # context.model_path point to the dir containing config/model
                candidate = os.path.join(context.model_path, filename)
                if os.path.exists(candidate):
                    expected = val
                    logger.info("Matched expectation for file: %s", filename)
                    found_file = True
                    break

            if not found_file:
                pytest.fail(
                    f"Could not find a matching model file for model_id '{model_id}' "
                    f"in {context.model_path} with expected files {list(entry.keys())}",
                )
        else:
            expected = entry

        if expected is not None:
            logger.info(
                "Consistency Check (%s): Actual=%.4f, Expected=%.4f",
                model_id,
                actual,
                expected,
            )
            assert actual == pytest.approx(expected, rel=5e-2)
        else:
            logger.warning(
                "Skipping consistency check: Could not match active model file to expected keys %s",
                list(entry.keys()),
            )
    else:
        logger.info(
            "Basic Validity Check: No model-specific expectation found for %s",
            model_id,
        )
        assert actual >= 1.0
