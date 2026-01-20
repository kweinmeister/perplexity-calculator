import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og

logger = logging.getLogger(__name__)


@dataclass
class ModelContext:
    model_path: str
    tokenizer: Any  # Should comply with encode(str) -> np.ndarray
    og_model: og.Model | None = None
    _ort_session: ort.InferenceSession | None = field(default=None, init=False)
    _empty_past_key_values: dict[str, np.ndarray] | None = field(
        default=None,
        init=False,
    )

    @property
    def ort_session(self) -> ort.InferenceSession:
        if self._ort_session is None:
            # Determine ONNX filename from genai_config.json if possible
            onnx_filename = "model.onnx"
            config_path = os.path.join(self.model_path, "genai_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    if "model" in config and "decoder" in config["model"]:
                        filename = config["model"]["decoder"].get("filename")
                        if filename:
                            onnx_filename = filename
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse genai_config.json: %s", e)
                except Exception as e:
                    logger.exception(
                        "Unexpected error reading genai_config.json: %s",
                        e,
                    )

            onnx_file = os.path.join(self.model_path, onnx_filename)
            if not os.path.exists(onnx_file):
                # Fallback: check 'onnx' subdirectory
                onnx_file_subdir = os.path.join(self.model_path, "onnx", onnx_filename)
                if os.path.exists(onnx_file_subdir):
                    onnx_file = onnx_file_subdir

            logger.info("Initializing ONNX Runtime session from %s", onnx_file)
            self._ort_session = ort.InferenceSession(onnx_file)
        return self._ort_session

    @property
    def empty_past_key_values(self) -> dict[str, np.ndarray]:
        """Returns initialized but empty past_key_values tensors for low-level ORT."""
        if self._empty_past_key_values is None:
            logger.info("Initializing empty KV cache tensors for low-level session")
            self._empty_past_key_values = {}
            session = self.ort_session
            for input_meta in session.get_inputs():
                if input_meta.name.startswith("past_key_values"):
                    # Shape comes as a list of ints or strings (for symbolic dims)
                    # Example: ['batch_size', 3, 'past_sequence_length', 64]
                    shape = []
                    for dim in input_meta.shape:
                        if isinstance(dim, str):
                            # Handle symbolic dimensions
                            if "batch" in dim.lower():
                                shape.append(1)
                            elif "seq" in dim.lower():
                                shape.append(0)
                            else:
                                logger.warning(
                                    "Unknown symbolic dimension '%s' in KV cache shape. Assuming 1. "
                                    "This may cause issues with some models.",
                                    dim,
                                )
                                shape.append(1)
                        else:
                            shape.append(dim)

                    # Determine dtype
                    # input_meta.type is string like 'tensor(float)' or 'tensor(float16)'
                    dtype = np.float32
                    if "float16" in input_meta.type:
                        dtype = np.float16
                    elif "double" in input_meta.type:
                        dtype = np.float64

                    self._empty_past_key_values[input_meta.name] = np.zeros(
                        shape,
                        dtype=dtype,
                    )
        return self._empty_past_key_values
