import pytest

import main
import perplexity


@pytest.fixture(scope="session")
def loaded_model() -> tuple[perplexity.ModelContext, str]:
    """
    Fixture to load the model and tokenizer once for the entire test session.
    """
    model_id = "onnx-community/Qwen3-0.6B-DQ-ONNX"
    context = main.load_model(model_id)
    return context, model_id


@pytest.fixture(
    params=main.load_test_data("test_data.yaml"),
    ids=lambda x: x.get("description", "Unknown"),
)
def test_case(request):
    """
    Fixture that provides one test case at a time from test_data.yaml.
    """
    return request.param
