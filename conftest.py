import pytest

import main
import model


@pytest.fixture(scope="session")
def loaded_model() -> tuple[model.ModelContext, str]:
    """Fixture to load the model and tokenizer once for the entire test session."""
    model_id = "onnx-community/SmolLM2-135M-ONNX"
    context = main.load_model(model_id)
    return context, model_id


@pytest.fixture(
    params=main.load_test_data("test_data.yaml"),
    ids=lambda x: x.get("description", "Unknown"),
)
def test_case(request: pytest.FixtureRequest) -> dict:
    """Fixture that provides one test case at a time from test_data.yaml."""
    return request.param
