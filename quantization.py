#!/usr/bin/env python3
import os
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np

class DummyDataReader(CalibrationDataReader):
    def __init__(self, input_name, num_samples=100):
        self.input_name = input_name
        self.num_samples = num_samples
        self.count = 0

    def get_next(self):
        if self.count < self.num_samples:
            self.count += 1
            # Gera dados aleatórios imitando imagens normalizadas em float32 (1, 3, 224, 224)
            # Isso é suficiente para o ONNX Runtime medir as escalas (S_in e S_out) e evitar o overflow.
            dummy_image = np.random.uniform(0, 127, (1, 3, 224, 224)).astype(np.float32)
            return {self.input_name: dummy_image}
        return None

def calibrate_model(input_onnx, output_onnx, quant_type):
    print(f"\n[+] Iniciando calibração para {output_onnx}...")
    
    # Descobre o nome da camada de entrada automaticamente (geralmente 'data' ou 'input')
    model = onnx.load(input_onnx)
    input_name = model.graph.input[0].name
    
    # Instancia o leitor de dados simulados
    data_reader = DummyDataReader(input_name=input_name, num_samples=100)
    
    # Executa a calibração estática
    quantize_static(
        model_input=input_onnx,
        model_output=output_onnx,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ, # Formato Quantize-Dequantize esperado pelo seu gerador
        per_channel=False,            # Gemmini usa quantização por tensor, não por canal
        weight_type=quant_type,
        activation_type=quant_type
    )
    print(f"[✓] Salvo com sucesso: {output_onnx}")

if __name__ == "__main__":
    modelo_original = "resnet50_Opset17.onnx"
    
    if not os.path.exists(modelo_original):
        print(f"Erro: O arquivo {modelo_original} não foi encontrado no diretório atual.")
        exit(1)

    # 1. Gera o modelo calibrado com escalas para 8 bits
    calibrate_model(modelo_original, "resnet50_int8_calibrado.onnx", QuantType.QInt8)
    
    # 2. Gera o modelo calibrado com escalas para 16 bits
    calibrate_model(modelo_original, "resnet50_int16_calibrado.onnx", QuantType.QInt16)
    
    print("\nCalibração concluída. Agora você pode gerar os headers em C.")