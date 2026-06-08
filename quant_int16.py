#!/usr/bin/env python3
"""
Script to perform static dummy quantization on an ONNX model (e.g., ResNet50)
for performance evaluation purposes.
"""

import os
import sys

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class DummyDataReader(CalibrationDataReader):
    """
    Generates dummy image data for model calibration during quantization.
    """

    def __init__(self, input_name: str, num_samples: int = 100):
        self.input_name = input_name
        self.num_samples = num_samples
        self.count = 0

    def get_next(self) -> dict | None:
        """Yields the next dummy input sample or None if exhausted."""
        if self.count < self.num_samples:
            self.count += 1
            # Random uniform data is sufficient for performance benchmarking
            dummy_image = np.random.uniform(
                0, 127, (1, 3, 224, 224)
            ).astype(np.float32)
            return {self.input_name: dummy_image}
        return None


def calibrate_model(
    input_onnx: str, output_onnx: str, quant_type: QuantType
) -> None:
    """
    Calibrates and quantizes the ONNX model statically.
    """
    print(f"\n[+] Starting calibration for '{output_onnx}'...")

    model = onnx.load(input_onnx)
    input_name = model.graph.input[0].name

    data_reader = DummyDataReader(input_name=input_name, num_samples=100)

    quantize_static(
        model_input=input_onnx,
        model_output=output_onnx,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=quant_type,
        activation_type=quant_type,
    )
    
    print(f"Successfully saved: '{output_onnx}'")


def main() -> None:
    model_path = "models/resnet50_Opset17.onnx"

    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Generate the model calibrated with 16-bit scales
    calibrate_model(
        model_path, 
        "resnet50_int16_calibrated.onnx", 
        QuantType.QInt16
    )

    print("\nCalibration completed. You can now generate the C headers.")


if __name__ == "__main__":
    main()