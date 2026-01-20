import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock onnxruntime_genai in sys.modules to avoid import errors in CI
module_mock = MagicMock()
sys.modules["onnxruntime_genai"] = module_mock

import perplexity  # noqa: E402
from model import ModelContext  # noqa: E402


class TestProcessLogitsChunk:
    def test_basic_calculation(self) -> None:
        logits1 = np.array([0.0, 10.0, 0.0])  # predicts index 1
        logits2 = np.array([0.0, 0.0, 10.0])  # predicts index 2

        logits_list = [logits1, logits2]
        targets_list = [1, 2]

        nll_sum, count = perplexity._process_logits_chunk(logits_list, targets_list)

        assert count == 2
        assert nll_sum < 0.1

    def test_high_perplexity(self) -> None:
        # Predicts index 0, but target is 1
        logits1 = np.array([10.0, 0.0, 0.0])
        logits_list = [logits1]
        targets_list = [1]

        nll_sum, count = perplexity._process_logits_chunk(logits_list, targets_list)

        assert count == 1
        assert nll_sum > 9.0


class TestCalculatePerplexityGenAI:
    @pytest.fixture
    def mock_context(self) -> Mock:
        context = Mock(spec=ModelContext)
        context.tokenizer = Mock()
        context.og_model = Mock()
        return context

    def test_short_input(self, mock_context) -> None:
        mock_context.tokenizer.encode.return_value = np.array([123])  # Just one token
        result = perplexity.calculate_perplexity_onnxruntime_genai(mock_context, "hi")
        assert result == float("inf")

    @patch("perplexity.og.GeneratorParams")
    @patch("perplexity.og.Generator")
    def test_calculation_flow(
        self, mock_generator_cls, mock_generator_params_cls, mock_context,
    ) -> None:
        # Setup input
        text = "prediction text"
        input_ids = np.array([10, 20, 30], dtype=np.int32)
        mock_context.tokenizer.encode.return_value = input_ids

        # Setup generator mock
        mock_gen_instance = mock_generator_cls.return_value

        vocab_size = 50
        fake_logits_1 = np.zeros((1, 1, vocab_size), dtype=np.float32)
        fake_logits_1[0, 0, 20] = 10.0  # Correct prediction for next token 20

        fake_logits_2 = np.zeros((1, 1, vocab_size), dtype=np.float32)
        fake_logits_2[0, 0, 30] = 10.0  # Correct prediction for next token 30

        mock_gen_instance.get_logits.side_effect = [fake_logits_1, fake_logits_2]

        score = perplexity.calculate_perplexity_onnxruntime_genai(mock_context, text)

        # Expectations
        assert mock_context.tokenizer.encode.called
        assert mock_generator_params_cls.called
        assert mock_generator_cls.called
        assert mock_gen_instance.append_tokens.call_count == 2
        assert 1.0 <= score < 1.1


class TestCalculatePerplexityOptimized:
    @pytest.fixture
    def mock_context(self) -> Mock:
        context = Mock(spec=ModelContext)
        context.tokenizer = Mock()
        context.ort_session = Mock()
        context.empty_past_key_values = {}
        return context

    def test_calculation_flow(self, mock_context) -> None:
        input_ids = np.array([10, 20, 30], dtype=np.int64)
        mock_context.tokenizer.encode.return_value = input_ids

        vocab_size = 50
        logits = np.zeros((1, 3, vocab_size), dtype=np.float32)

        # At index 0 (token 10), we want to predict token 20
        logits[0, 0, 20] = 10.0

        # At index 1 (token 20), we want to predict token 30
        logits[0, 1, 30] = 10.0

        mock_context.ort_session.run.return_value = [logits]

        score = perplexity.calculate_perplexity_onnxruntime_optimized(
            mock_context, "foo",
        )

        assert score is not None
        assert 1.0 <= score < 1.1
