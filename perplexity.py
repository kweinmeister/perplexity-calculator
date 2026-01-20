from typing import cast

import numpy as np
import onnxruntime_genai as og

from model import ModelContext


def _process_logits_chunk(
    logits_list: list[np.ndarray],
    targets_list: list[int],
) -> tuple[float, int]:
    """Calculates NLL sum and count for a chunk of logits and targets."""
    chunk_logits = np.stack(logits_list)
    chunk_targets = np.array(targets_list)

    c = np.max(chunk_logits, axis=1, keepdims=True)
    lse = c + np.log(np.sum(np.exp(chunk_logits - c), axis=1, keepdims=True))

    row_indices = np.arange(len(chunk_targets))
    target_logits = chunk_logits[row_indices, chunk_targets]

    # Sum of NLL for this chunk
    # NLL = lse - target_logits
    chunk_nll_sum = np.sum(lse.squeeze() - target_logits)

    return float(chunk_nll_sum), len(chunk_targets)


def calculate_perplexity_onnxruntime_genai(context: ModelContext, text: str) -> float:
    """Calculates perplexity using optimized NumPy operations."""
    tokenizer = context.tokenizer
    model = context.og_model

    if model is None:
        return float("inf")

    input_ids = tokenizer.encode(text)

    if len(input_ids) < 2:
        return float("inf")

    generator = og.Generator(model, og.GeneratorParams(model))

    input_ids_int32 = (
        input_ids.astype(np.int32) if input_ids.dtype != np.int32 else input_ids
    )

    logits_list = []
    targets_list = []
    append_tokens = generator.append_tokens
    get_logits = generator.get_logits

    nll_sum = 0.0
    count = 0
    chunk_size = 128

    # Iterate over input tokens to predict the next token
    # i goes from 0 to len-2.
    # input: input_ids_int32[i]
    # target: input_ids_int32[i+1]
    for i in range(len(input_ids_int32) - 1):
        append_tokens(input_ids_int32[i : i + 1])
        logits_list.append(get_logits()[0, 0, :])
        targets_list.append(int(input_ids_int32[i + 1]))

        if len(logits_list) >= chunk_size:
            # Process chunk
            chunk_nll_sum, chunk_count = _process_logits_chunk(
                logits_list,
                targets_list,
            )
            nll_sum += chunk_nll_sum
            count += chunk_count

            logits_list = []
            targets_list = []

    # Process remaining logits
    if logits_list:
        chunk_nll_sum, chunk_count = _process_logits_chunk(logits_list, targets_list)
        nll_sum += chunk_nll_sum
        count += chunk_count

    if count == 0:
        return 0.0

    return float(np.exp(nll_sum / count))


def calculate_perplexity_onnxruntime_optimized(
    context: ModelContext,
    text: str,
) -> float:
    """Calculates perplexity using single-pass ONNX Runtime inference."""
    tokenizer = context.tokenizer
    session = context.ort_session

    input_ids = tokenizer.encode(text)
    if len(input_ids) < 2:
        return float("inf")

    input_ids_int64 = input_ids.astype(np.int64)
    # Ensure 2D [1, Seq]
    if len(input_ids_int64.shape) == 1:
        input_ids_int64 = np.expand_dims(input_ids_int64, axis=0)

    inputs = {
        "input_ids": input_ids_int64,
        "attention_mask": np.ones(input_ids_int64.shape, dtype=np.int64),
    }
    inputs.update(context.empty_past_key_values)

    outputs = session.run(None, inputs)
    logits = cast("np.ndarray", outputs[0])  # [1, SeqLen, Vocab]

    # We want logits for t=0..N-2 to predict t=1..N-1
    # logits shape: [1, SeqLen, Vocab]
    shift_logits = logits[0, :-1, :]  # [SeqLen-1, Vocab]

    # Targets are input_ids 1..N-1
    targets = input_ids_int64[0, 1:]  # [SeqLen-1]

    # Compute Cross Entropy / NLL efficiently
    c = np.max(shift_logits, axis=1, keepdims=True)
    lse = c + np.log(np.sum(np.exp(shift_logits - c), axis=1, keepdims=True))

    row_indices = np.arange(len(targets))
    target_logits = shift_logits[row_indices, targets]

    # Mean NLL
    mean_nll = np.mean(lse.squeeze() - target_logits)

    return float(np.exp(mean_nll))


def calculate_perplexity_onnxruntime_baseline(
    context: ModelContext,
    text: str,
) -> float:
    """Calculates perplexity using sequential execution with manual KV caching."""
    tokenizer = context.tokenizer
    session = context.ort_session

    input_ids = tokenizer.encode(text)
    if len(input_ids) < 2:
        return float("inf")

    input_ids_int64 = input_ids.astype(np.int64)

    # Initialize KV cache with empty tensors
    past_key_values = dict(context.empty_past_key_values)

    # Map present_* outputs back to past_key_values_* inputs for next step
    # We assume standard HF naming (present.X.key -> past_key_values.X.key)

    output_names = [o.name for o in session.get_outputs()]
    # Expected output[0] is logits

    nll_sum = 0.0
    count = 0

    seq_len = len(input_ids_int64)

    # Loop through sequence, predicting token[i+1] given token[0...i]

    for i in range(seq_len - 1):
        current_token = input_ids_int64[i]
        target_token = input_ids_int64[i + 1]

        current_input_ids = np.array([[current_token]], dtype=np.int64)
        current_attention_mask = np.ones((1, i + 1), dtype=np.int64)

        inputs = {
            "input_ids": current_input_ids,
            "attention_mask": current_attention_mask,
        }
        inputs.update(past_key_values)

        # Run inference
        outputs = session.run(None, inputs)

        logits = cast("np.ndarray", outputs[0])  # (1, 1, vocab_size)

        # Calculate loss (logits shape: 1, 1, vocab_size)
        next_token_logits = logits[0, 0, :]

        # Stable Softmax / NLL
        c = np.max(next_token_logits)
        lse = c + np.log(np.sum(np.exp(next_token_logits - c)))
        target_logit = next_token_logits[target_token]
        nll = lse - target_logit

        nll_sum += nll
        count += 1

        # Update KV Cache by mapping outputs to inputs

        present_values = outputs[1:]
        present_names = output_names[1:]

        for name, val in zip(present_names, present_values, strict=False):
            # name is like 'present.0.key'
            # we want key to be 'past_key_values.0.key'
            new_key = name.replace("present", "past_key_values")
            past_key_values[new_key] = cast("np.ndarray", val)

    return float(np.exp(nll_sum / count))
