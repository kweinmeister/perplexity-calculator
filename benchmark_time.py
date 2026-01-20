from collections.abc import Callable
from typing import Any

import pytest

import main
import model
import perplexity


@pytest.fixture
def test_data() -> list[str]:
    """Fixture that provides the list of all test texts."""
    data = main.load_test_data()
    return main.get_test_texts(data)


@pytest.mark.parametrize(
    "calc_fn",
    [
        perplexity.calculate_perplexity_onnxruntime_genai,
        perplexity.calculate_perplexity_onnxruntime_baseline,
        perplexity.calculate_perplexity_onnxruntime_optimized,
    ],
)
def test_benchmark_perplexity(
    benchmark: Any,
    loaded_model: tuple[model.ModelContext, str],
    test_data: list[str],
    calc_fn: Callable[[model.ModelContext, str], float],
) -> None:
    """Benchmark the perplexity calculation on a batch of test examples."""
    context, _ = loaded_model

    def run_batch() -> None:
        main.run_batch_perplexity(context, test_data, calc_fn)

    benchmark(run_batch)
