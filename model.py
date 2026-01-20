import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og


@dataclass
class ModelContext:
    model_path: str
    tokenizer: Any  # Should comply with encode(str) -> np.ndarray
    og_model: Optional[og.Model] = None
    _ort_session: Optional[ort.InferenceSession] = field(default=None, init=False)
    _empty_past_key_values: Optional[dict[str, np.ndarray]] = field(
        default=None, init=False
    )

    @property
    def ort_session(self) -> ort.InferenceSession:
        if self._ort_session is None:
            # Determine ONNX filename from genai_config.json if possible
            onnx_filename = "model.onnx"
            config_path = os.path.join(self.model_path, "genai_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    if "model" in config and "decoder" in config["model"]:
                        filename = config["model"]["decoder"].get("filename")
                        if filename:
                            onnx_filename = filename
                except Exception as e:
                    print(f"Warning: Failed to parse genai_config.json: {e}")

            onnx_file = os.path.join(self.model_path, onnx_filename)
            if not os.path.exists(onnx_file):
                # Fallback: check 'onnx' subdirectory
                onnx_file_subdir = os.path.join(self.model_path, "onnx", onnx_filename)
                if os.path.exists(onnx_file_subdir):
                    onnx_file = onnx_file_subdir

            print(f"Loading ORT session from: {onnx_file}")
            self._ort_session = ort.InferenceSession(onnx_file)
        return self._ort_session

    @property
    def empty_past_key_values(self) -> dict[str, np.ndarray]:
        if self._empty_past_key_values is None:
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
                        shape, dtype=dtype
                    )
        return self._empty_past_key_values
