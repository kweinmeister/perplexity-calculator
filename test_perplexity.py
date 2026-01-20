from typing import Callable

import pytest

import model
import perplexity


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
    """
    Test that calculates perplexity and checks against expectations (if any).
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
    print(f"\nMethod: {method_name}")
    print(f"Example: {description}")
    print(f"Perplexity: {actual:.4f}")

    # Basic assertion
    assert isinstance(actual, float)

    # Consistency check (if model-specific expectation exists)
    if expected_dict and model_id in expected_dict:
        entry = expected_dict[model_id]

        expected = None
        if isinstance(entry, dict):
            # Check which file from the expected list exists in the model path
            import os

            found_file = False
            for filename, val in entry.items():
                # We need to guess where the file is (root or onnx subdir)
                # context.model_path point to the dir containing config/model
                candidate = os.path.join(context.model_path, filename)
                if os.path.exists(candidate):
                    expected = val
                    print(f"Matched expectation for file: {filename}")
                    found_file = True
                    break

            if not found_file:
                # Fallback: if we simply have a float value in the dict (backward compat but shouldn't happen with new schema)
                # Or just print warning
                print(
                    f"Warning: No expected file found directly in {context.model_path}. Checking keys..."
                )
                # Blindly take the first one? No, unsafe.
                # Let's assume standard names
                pass
        else:
            expected = entry

        if expected is not None:
            print(
                f"Consistency Check ({model_id}): Actual={actual:.4f}, Expected={expected:.4f}"
            )
            assert actual == pytest.approx(expected, rel=5e-2)
        else:
            print(
                f"Skipping consistency check: Could not match active model file to expected keys {list(entry.keys())}"
            )
    else:
        print(
            f"Basic Validity Check: No model-specific expectation found for {model_id}"
        )
        assert actual >= 1.0
