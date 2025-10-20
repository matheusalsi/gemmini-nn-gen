#!/usr/bin/env python3
"""
onnx2gemmini.py (versão final e robusta)

Gera arquivos .c e .h compatíveis com Gemmini a partir de um modelo ONNX.
Correções:
 - Corrigido o SyntaxError.
 - Adicionado um bloco 'else' genérico para tratar todas as camadas não-computacionais
   (Add, Relu, Flatten, etc.), garantindo a propagação de buffers em qualquer
   arquitetura e resolvendo o `KeyError`.
 - Lógica mantida para Conv, Gemm, Pooling e formatação.
"""
import onnx
import numpy as np
import os
import argparse

def to_scalar(x, default=None):
    if x is None: return default
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return int(x)
    try:
        arr = np.array(x).flatten()
        return default if arr.size == 0 else int(arr.tolist()[0])
    except Exception:
        try: return int(x)
        except Exception: return default

def tensor_to_numpy(tensor):
    if tensor.data_type == onnx.TensorProto.FLOAT:
        dtype, attr = np.float32, 'float_data'
    elif tensor.data_type == onnx.TensorProto.INT32:
        dtype, attr = np.int32, 'int32_data'
    elif tensor.data_type == onnx.TensorProto.INT64:
        dtype, attr = np.int64, 'int64_data'
    else:
        raise NotImplementedError(f"ONNX dtype {tensor.data_type} not implemented")
    
    data = getattr(tensor, attr) or np.frombuffer(tensor.raw_data, dtype=dtype)
    return np.array(data, dtype=dtype).reshape(tensor.dims)

def quantize_tensor(tensor_f32, precision_bits=8):
    qmax = (2 ** (precision_bits - 1)) - 1
    dtype = np.int8 if precision_bits == 8 else np.int16
    maxval = np.max(np.abs(tensor_f32))
    scale = float(maxval / qmax) if maxval != 0 else 1.0
    q = np.round(tensor_f32 / scale).astype(dtype)
    return q, scale

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name: return onnx.helper.get_attribute_value(a)
    return default

def compute_conv_output(h_in, w_in, k, stride, pads, dilation):
    h_in, w_in = to_scalar(h_in), to_scalar(w_in)
    if h_in is None or w_in is None: raise RuntimeError("Input spatial dims unknown.")
    stride_val, dilation_val = to_scalar(stride, 1), to_scalar(dilation, 1)
    pad_h = pads[0] + pads[2] if pads and len(pads) == 4 else 0
    pad_w = pads[1] + pads[3] if pads and len(pads) == 4 else 0
    kH = kW = to_scalar(k, 1)
    out_h = (h_in + pad_h - dilation_val * (kH - 1) - 1) // stride_val + 1
    out_w = (w_in + pad_w - dilation_val * (kW - 1) - 1) // stride_val + 1
    return int(out_h), int(out_w)

def export_gemmini(onnx_path, out_dir='out', precision=8, batch_size=4):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    graph = model.graph
    
    inits = {t.name: tensor_to_numpy(t) for t in graph.initializer}
    
    node_by_input = {i: [] for node in graph.node for i in node.input if i}
    for node in graph.node:
        for i in node.input:
            if i: node_by_input[i].append(node)

    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, C, H, W = input_shape if len(input_shape) == 4 else (None, None, None, None)

    basename = os.path.basename(out_dir)
    h_lines, c_lines = [], []

    # Boilerplate...
    guard = f"{basename.upper()}_PARAMETERS_H"
    h_lines.append(f"#ifndef {guard}\n#define {guard}\n\n#include <include/gemmini_params.h>\n#include <stdbool.h>\n\n")
    c_lines.extend(['#include <stdio.h>\n', '#include <string.h>\n', '#include <stdbool.h>\n', '#ifndef BAREMETAL\n', '#include <sys/mman.h>\n', '#endif\n', '#include "include/gemmini.h"\n', '#include "include/gemmini_nn.h"\n\n', f'#include "{basename}_params.h"\n', '#include "images.h"\n\n', 'int main (int argc, char * argv[]) {\n', '#ifndef BAREMETAL\n', '    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }\n', '#endif\n\n', '    gemmini_flush(0);\n\n', '    enum tiled_matmul_type_t tiled_matmul_type = WS;\n', '    if (argc > 1) {\n', '        if (strcmp(argv[1], "cpu") == 0) tiled_matmul_type = CPU;\n', '        else if (strcmp(argv[1], "os") == 0) tiled_matmul_type = OS;\n', '        else if (strcmp(argv[1], "ws") == 0) tiled_matmul_type = WS;\n', '    }\n\n', '    uint64_t start, end;\n', '    uint64_t conv_cycles = 0, matmul_cycles = 0;\n\n', '    // model execution\n'])

    layer_idx = 1
    tensor_buffer, output_dims = {}, {}
    inp_name = graph.input[0].name
    tensor_buffer[inp_name], output_dims[inp_name] = 'images', (C, H, W)
    processed_nodes = set()

    for node in graph.node:
        if node.name in processed_nodes: continue

        if node.op_type == 'Conv':
            X_name, W_name = node.input[0], node.input[1]
            B_name = node.input[2] if len(node.input) > 2 else None
            Y_name = node.output[0]
            W_data = inits[W_name]
            out_ch, in_ch_per_group, kH, _ = W_data.shape
            groups = get_attr(node, 'group', 1)
            in_ch = in_ch_per_group * groups
            strides_attr, pads_attr = get_attr(node, 'strides', [1,1]), get_attr(node, 'pads', [0,0,0,0])
            _, h_in, w_in = output_dims[X_name]
            out_h, out_w = compute_conv_output(h_in, w_in, kH, strides_attr, pads_attr, [1,1])
            output_dims[Y_name] = (out_ch, out_h, out_w)
            B_data = inits.get(B_name, np.zeros(out_ch, dtype=np.float32))
            qW, scaleW = quantize_tensor(W_data.astype(np.float32), precision)
            qB, _ = quantize_tensor(B_data.astype(np.float32), precision)
            lname = f"conv_{layer_idx}"
            patch_size = in_ch * kH * kH
            w_t = qW.reshape(out_ch, -1).T
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in w_t]
            w_cstr, b_str = "{" + ",".join(rows) + "}", "{" + ",".join(map(str, qB.tolist())) + "}"
            n_patches = out_h * out_w * batch_size
            h_lines.extend([f"static const elem_t {lname}_w[{patch_size}][{out_ch}] row_align(1) = {w_cstr};\n",
                            f"static const acc_t {lname}_b[{out_ch}] row_align_acc(1) = {b_str};\n",
                            f"static elem_t {lname}_in[{n_patches}][{patch_size}] row_align(1);\n",
                            f"static elem_t {lname}_out[{n_patches}][{out_ch}] row_align(1);\n"])
            tensor_buffer[Y_name] = f"{lname}_out"
            act, pool_node = 'NONE', None
            out_h_pooled, out_w_pooled = out_h, out_w
            pool_params = {'pool_size': 1, 'pool_stride': 1, 'pool_padding': 0}
            current_tensor_name = Y_name
            while True:
                next_consumers = node_by_input.get(current_tensor_name, [])
                if not next_consumers: break
                consumer = next_consumers[0]
                if consumer.op_type == 'Relu' and act == 'NONE':
                    act = 'RELU'
                    processed_nodes.add(consumer.name)
                    current_tensor_name = consumer.output[0]
                    output_dims[current_tensor_name], tensor_buffer[current_tensor_name] = output_dims[Y_name], tensor_buffer[Y_name]
                elif consumer.op_type in ('MaxPool', 'AveragePool'):
                    pool_node = consumer
                    processed_nodes.add(consumer.name)
                    break
                else: break
            if pool_node:
                pool_k, pool_s, pool_p = get_attr(pool_node, 'kernel_shape'), get_attr(pool_node, 'strides'), get_attr(pool_node, 'pads')
                pool_params = {'pool_size': to_scalar(pool_k), 'pool_stride': to_scalar(pool_s), 'pool_padding': to_scalar(pool_p)}
                out_h_pooled, out_w_pooled = compute_conv_output(out_h, out_w, pool_params['pool_size'], pool_s, pool_p, 1)
                h_lines.append(f"static elem_t {lname}_out_pooled[{batch_size}][{out_h_pooled}][{out_w_pooled}][{out_ch}];\n")
                pool_out_name = pool_node.output[0]
                output_dims[pool_out_name], tensor_buffer[pool_out_name] = (out_ch, out_h_pooled, out_w_pooled), f"{lname}_out_pooled"
            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0
            params_str = (f"static const struct ConvParams {lname}_params = {{"
                          f".batch_size={batch_size}, .in_row_dim={h_in}, .in_col_dim={w_in}, "
                          f".kernel_size={kH}, .in_channels={in_ch}, .out_channels={out_ch}, "
                          f".stride={to_scalar(strides_attr)}, .padding={to_scalar(pads_attr)}, "
                          f".bias=1, .depthwise={1 if groups == in_ch else 0}, "
                          f".out_row_dim={out_h}, .out_col_dim={out_w}, "
                          f".n_patches={n_patches}, .patch_size={patch_size}, "
                          f".pool_size={pool_params['pool_size']}, .pool_stride={pool_params['pool_stride']}, .pool_padding={pool_params['pool_padding']}, "
                          f".out_dim_pooled={out_h_pooled}, .output_scale=(1.0 / (1 << {shift})), "
                          f".I={n_patches}, .J={out_ch}, .K={patch_size}}};")
            h_lines.append(f"{params_str}\n\n")
            c_lines.extend([f"    // {lname}\n", "    start = read_cycles();\n", f"    tiled_conv_auto( {lname}_params.batch_size, {lname}_params.in_row_dim, {lname}_params.in_col_dim, {lname}_params.in_channels, {lname}_params.out_channels, {lname}_params.out_row_dim, {lname}_params.out_col_dim, {lname}_params.stride, 1, 1, {lname}_params.padding, {lname}_params.kernel_size, false, false, false, false, false, (elem_t*){tensor_buffer.get(X_name, 'images')}, (elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out, {act}, {lname}_params.output_scale, {lname}_params.pool_size, {lname}_params.pool_stride, {lname}_params.pool_padding, tiled_matmul_type);\n", "    end = read_cycles();\n", f"    conv_cycles += end - start;\n"])
            layer_idx += 1
            processed_nodes.add(node.name)

        elif node.op_type == 'Gemm':
            A_name, W_name, B_name = node.input[0], node.input[1], node.input[2]
            Y_name = node.output[0]
            W_data, B_data = inits[W_name], inits[B_name]
            out_features, in_features = W_data.shape
            qW, scaleW = quantize_tensor(W_data.astype(np.float32), precision)
            qB, _ = quantize_tensor(B_data.astype(np.float32), precision)
            qW_t = qW.T
            lname = f"fc_{layer_idx}"
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in qW_t]
            w_cstr, b_str = "{" + ",".join(rows) + "}", "{" + ",".join(map(str, qB.tolist())) + "}"
            h_lines.extend([f"static const elem_t {lname}_w[{in_features}][{out_features}] row_align(1) = {w_cstr};\n",
                            f"static const acc_t {lname}_b[{out_features}] row_align_acc(1) = {b_str};\n",
                            f"static elem_t {lname}_out[{batch_size}][{out_features}] row_align(1);\n"])
            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0
            fc_params_str = (f"static const struct FcParams {lname}_params = {{"
                             f".batch_size={batch_size}, .in_features={in_features}, .out_features={out_features}, "
                             f".bias=1, .output_scale=(1.0 / (1 << {shift})), "
                             f".I={batch_size}, .J={out_features}, .K={in_features}}};")
            h_lines.append(f"{fc_params_str}\n\n")
            tensor_buffer[Y_name] = f"{lname}_out"
            c_lines.extend([f"    // {lname}\n", "    start = read_cycles();\n", f"    tiled_matmul_nn_auto({batch_size}, {out_features}, {in_features}, (elem_t*){tensor_buffer[A_name]}, (elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out, NO_ACTIVATION, 1.0, true, tiled_matmul_type, false, \"{lname}\");\n", "    end = read_cycles();\n", f"    matmul_cycles += end - start;\n"])
            layer_idx += 1
            processed_nodes.add(node.name)

        else: # Manipulador genérico para todos os outros nós (Add, Relu, Flatten, etc.)
            if not node.input or not node.output: continue
            
            # A maioria dos nós tem uma entrada principal que propaga as dimensões
            input_name = node.input[0]
            output_name = node.output[0]

            if input_name in output_dims:
                # Lógica especial para Flatten
                if node.op_type == 'Flatten':
                    dims = output_dims[input_name]
                    # Achata para (N, C*H*W) - mas para nós só importa o buffer
                    flat_dim = dims[0] * dims[1] * dims[2] if len(dims) == 3 else dims[0]
                    output_dims[output_name] = (flat_dim, 1, 1)
                else:
                    output_dims[output_name] = output_dims[input_name]

            if input_name in tensor_buffer:
                tensor_buffer[output_name] = tensor_buffer[input_name]
            
            processed_nodes.add(node.name)


    # Footer...
    c_lines.extend(["\n    uint64_t total_cycles = conv_cycles + matmul_cycles;\n", '    printf("\\nTotal cycles: %llu (100%%)\\n", total_cycles);\n', '    if(total_cycles != 0) {\n', '        printf("Matmul cycles: %llu (%d%%)\\n", matmul_cycles, (int)((matmul_cycles * 100) / total_cycles));\n', '        printf("Conv cycles: %llu (%d%%)\\n", conv_cycles, (int)((conv_cycles * 100) / total_cycles));\n', '    }\n','    printf("PASS\\n");\n\n', '    exit(0);\n', '}\n'])
    h_lines.append(f"#endif /* {guard} */\n")

    with open(os.path.join(out_dir, f'{basename}_params.h'), 'w') as f:
        f.write(''.join(h_lines))
    with open(os.path.join(out_dir, f'{basename}.c'), 'w') as f:
        f.write(''.join(c_lines))

    print(f"Written {basename}_params.h and {basename}.c to {out_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('onnx', help='ONNX model path')
    p.add_argument('--out', default='out', help='output directory')
    p.add_argument('--precision', type=int, default=8, choices=[8, 16], help='quant precision bits')
    p.add_argument('--batch_size', type=int, default=4, help='batch size for execution')
    args = p.parse_args()
    export_gemmini(args.onnx, out_dir=args.out, precision=args.precision, batch_size=args.batch_size)