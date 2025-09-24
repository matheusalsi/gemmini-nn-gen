#!/usr/bin/env python3
"""
onnx2gemmini.py (corrigido)

Gera arquivos .c e .h compatíveis com Gemmini a partir de um modelo ONNX.
Correções:
 - usa as dimensões de entrada na ordem correta (C,H,W)
 - recupera dimensões por nome do tensor (não por nome do buffer)
 - gera conv weights como [patch_size][out_channels] (compatível com exemplos)
 - declara buffers de saída como [n_patches][out_channels]
 - output_scale escrito como float
 - parsing robusto de pads, strides, dilations
"""
import onnx
import numpy as np
import os
import argparse

def to_scalar(x, default=None):
    """
    Converte x para um int escalar de maneira robusta.
    Aceita int, float, np.integer, lists, tuples, np.ndarray, onnx lists, etc.
    Retorna default se não for possível.
    """
    if x is None:
        return default
    # já é escalar numpy / python
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    # tenta converter sequências/arrays
    try:
        arr = np.array(x).flatten()
        if arr.size == 0:
            return default
        # pegamos o primeiro elemento e transformamos em int
        first = arr.tolist()[0]
        return int(first)
    except Exception:
        # fallback final
        try:
            return int(x)
        except Exception:
            return default

def tensor_to_numpy(tensor):
    if tensor.data_type == onnx.TensorProto.FLOAT:
        dtype = np.float32
        if tensor.raw_data:
            arr = np.frombuffer(tensor.raw_data, dtype=dtype)
        else:
            arr = np.array(tensor.float_data, dtype=dtype)
    elif tensor.data_type == onnx.TensorProto.INT32:
        dtype = np.int32
        if tensor.raw_data:
            arr = np.frombuffer(tensor.raw_data, dtype=dtype)
        else:
            arr = np.array(tensor.int32_data, dtype=dtype)
    elif tensor.data_type == onnx.TensorProto.INT64:
        dtype = np.int64
        if tensor.raw_data:
            arr = np.frombuffer(tensor.raw_data, dtype=dtype)
        else:
            arr = np.array(tensor.int64_data, dtype=dtype)
    else:
        raise NotImplementedError(f"ONNX tensor dtype {tensor.data_type} not implemented")
    return arr.reshape(tensor.dims)

def quantize_tensor(tensor_f32, precision_bits=8):
    assert precision_bits in (8, 16), "Only 8 and 16-bit quantization supported"
    if precision_bits == 8:
        qmax = 127
        dtype = np.int8
    else:
        qmax = 2 ** 15 - 1
        dtype = np.int16
    maxval = np.max(np.abs(tensor_f32))
    if maxval == 0:
        scale = 1.0
    else:
        scale = float(maxval) / float(qmax)
    q = np.round(tensor_f32 / scale).astype(dtype)
    return q, float(scale)

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default

def compute_conv_output(h_in, w_in, k, stride, pads, dilation):
    """
    Retorna (out_h, out_w) como inteiros, aceitando inputs que sejam arrays/tuplas/listas.
    """
    h_in = to_scalar(h_in)
    w_in = to_scalar(w_in)
    if h_in is None or w_in is None:
        raise RuntimeError("Input spatial dims unknown. For conv nodes we need H and W.")

    # extrai stride e dilation de forma robusta
    stride_val = to_scalar(stride, default=1)
    dilation_val = to_scalar(dilation, default=1)

    # pads pode ser [top, left, bottom, right] ou [hpad, wpad] ou escalar
    pads_arr = []
    if pads is not None:
        try:
            pads_arr = np.array(pads).flatten().astype(int).tolist()
        except Exception:
            # se não for iterável, tenta converter direto
            try:
                pads_arr = [int(pads)]
            except Exception:
                pads_arr = []

    if len(pads_arr) >= 4:
        # onnx pads: [pad_top, pad_left, pad_bottom, pad_right]
        pad_h = pads_arr[0]
        pad_w = pads_arr[1]
    elif len(pads_arr) == 2:
        pad_h, pad_w = pads_arr
    elif len(pads_arr) == 1:
        pad_h = pad_w = pads_arr[0]
    else:
        pad_h = pad_w = 0

    k = int(k)
    out_h = (h_in + 2 * pad_h - dilation_val * (k - 1) - 1) // stride_val + 1
    out_w = (w_in + 2 * pad_w - dilation_val * (k - 1) - 1) // stride_val + 1
    return int(out_h), int(out_w)

def sanitize_name(name):
    return name.replace('/', '_').replace('.', '_')

def export_gemmini(onnx_path, out_dir='out', precision=8, batch_size=4):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    graph = model.graph
    inits = {}
    for t in graph.initializer:
        inits[t.name] = tensor_to_numpy(t)

    batch = batch_size

    # descobrir shape de input (assumimos NCHW)
    input_shape = None
    if len(graph.input) > 0:
        vi = graph.input[0]
        try:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            input_shape = dims
        except Exception:
            input_shape = None

    if input_shape is None or len(input_shape) < 4:
        C = None
        H = None
        W = None
    else:
        # ordem: N, C, H, W
        _, C, H, W = input_shape

    basename = os.path.basename(out_dir)

    h_lines = []
    c_lines = []

    # header
    guard = f"{basename.upper()}_PARAMETERS_H"
    h_lines.append(f"#ifndef {guard}\n")
    h_lines.append(f"#define {guard}\n\n")
    h_lines.append("#include <include/gemmini_params.h>\n")
    h_lines.append("#include <stdbool.h>\n\n")

    # .c boilerplate
    c_lines.append('#include <stdio.h>\n')
    c_lines.append('#include <string.h>\n')
    c_lines.append('#include <stdbool.h>\n')
    c_lines.append('#ifndef BAREMETAL\n')
    c_lines.append('#include <sys/mman.h>\n')
    c_lines.append('#endif\n')
    c_lines.append('#include "include/gemmini.h"\n')
    c_lines.append('#include "include/gemmini_nn.h"\n\n')
    c_lines.append(f'#include "{basename}_params.h"\n')
    c_lines.append('#include "images.h"\n\n')
    c_lines.append('int main (int argc, char * argv[]) {\n')
    c_lines.append('#ifndef BAREMETAL\n')
    c_lines.append('    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {\n')
    c_lines.append('      perror("mlockall failed");\n')
    c_lines.append('      exit(1);\n')
    c_lines.append('    }\n')
    c_lines.append('#endif\n\n')
    c_lines.append('    gemmini_flush(0);\n\n')
    c_lines.append('    enum tiled_matmul_type_t tiled_matmul_type = WS;\n')
    c_lines.append('    if (argc < 2) {\n')
    c_lines.append('        tiled_matmul_type = WS;\n')
    c_lines.append('    } else if (strcmp(argv[1], "cpu") == 0) {\n')
    c_lines.append('        tiled_matmul_type = CPU;\n')
    c_lines.append('    } else if (strcmp(argv[1], "os") == 0) {\n')
    c_lines.append('        tiled_matmul_type = OS;\n')
    c_lines.append('    } else if (strcmp(argv[1], "ws") == 0) {\n')
    c_lines.append('        tiled_matmul_type = WS;\n')
    c_lines.append('    } else if (strcmp(argv[1], "-h") == 0) {\n')
    c_lines.append('        printf("usage: %s [-h] matmul_option [check]\\n  matmul_option may be \'os\', \'ws\', or cpu\'\\n", argv[0]);\n')
    c_lines.append('        exit(0);\n')
    c_lines.append('    } else {\n')
    c_lines.append('        printf("Unknown command-line argument\\n");\n')
    c_lines.append('        printf("usage: %s [-h] matmul_option [check]\\n  matmul_option may be \'os\', \'ws\', or cpu\'\\n", argv[0]);\n')
    c_lines.append('        exit(1);\n')
    c_lines.append('    }\n\n')
    c_lines.append('    bool check=false;\n')
    c_lines.append('    if (argc > 2) {\n')
    c_lines.append('        if (strcmp(argv[2], "check") == 0) {\n')
    c_lines.append('            check = true;\n')
    c_lines.append('        } else {\n')
    c_lines.append('            printf("Unknown command-line argument\\n");\n')
    c_lines.append('            printf("usage: %s [-h] matmul_option [check]\\n  matmul_option may be \'os\', \'ws\', or cpu\'\\n", argv[0]);\n')
    c_lines.append('            exit(1);\n')
    c_lines.append('        }\n')
    c_lines.append('    }\n\n')
    c_lines.append('    uint64_t start, end;\n')
    c_lines.append('    uint64_t conv_cycles = 0, matmul_cycles = 0;\n\n')
    c_lines.append('    // model execution\n')

    layer_idx = 0
    tensor_buffer = {}
    # map tensor name -> (out_channels, out_h, out_w)
    output_dims = {}

    if len(graph.input) > 0:
        inp_name = graph.input[0].name
        tensor_buffer[inp_name] = 'images'
        # store input dims keyed by tensor name when available
        if C is not None and H is not None and W is not None:
            output_dims[inp_name] = (C, H, W)

    for node in graph.node:
        if node.op_type == 'Conv':
            X = node.input[0]                # tensor name of input
            W_name = node.input[1]
            B_name = node.input[2] if len(node.input) > 2 else None

            if W_name not in inits:
                print(f"Warning: weights {W_name} not found in initializers, skipping conv")
                continue
            W = inits[W_name]  # shape: (out_ch, in_ch_per_group, kH, kW)
            out_ch, in_ch_per_group, kH, kW = W.shape
            groups = get_attr(node, 'group', 1)
            in_ch = in_ch_per_group * groups

            stride = get_attr(node, 'strides', [1, 1])
            if isinstance(stride, (list, tuple, np.ndarray)):
                stride_val = int(np.array(stride).flatten()[0])
            else:
                stride_val = int(stride)
            pads = get_attr(node, 'pads', [0, 0, 0, 0])
            dilation = get_attr(node, 'dilations', [1, 1])
            if isinstance(dilation, (list, tuple, np.ndarray)):
                dilation_val = int(np.array(dilation).flatten()[0])
            else:
                dilation_val = int(dilation)

            # Recupera dims da entrada (usar nome do tensor X)
            if X in output_dims:
                in_ch_dim, h_in, w_in = output_dims[X]
            else:
                in_ch_dim, h_in, w_in = C, H, W  # fallback para input model-wide

            # garantir que sejam escalares Python int
            in_ch_dim = to_scalar(in_ch_dim)
            h_in = to_scalar(h_in)
            w_in = to_scalar(w_in)


            if h_in is None or w_in is None:
                raise RuntimeError("Input spatial dims unknown. For conv nodes we need H and W.")

            out_h, out_w = compute_conv_output(h_in, w_in, kH, stride_val, pads, dilation_val)

            # armazenar dims com o nome do tensor de saída
            if node.output:
                output_dims[node.output[0]] = (out_ch, out_h, out_w)

            if B_name and B_name in inits:
                B = inits[B_name]
            else:
                B = np.zeros(out_ch, dtype=np.float32)

            qW, scaleW = quantize_tensor(W.astype(np.float32), precision)
            qB, scaleB = quantize_tensor(B.astype(np.float32), precision)

            lname = f"conv_{layer_idx}"

            # reorganizar pesos: queremos [patch_size][out_ch]
            patch_size = in_ch * kH * kW
            w_per_filter = qW.reshape(out_ch, -1)            # (out_ch, patch_size)
            # transpor para (patch_size, out_ch) para formatar parecido com exemplos
            w_t = w_per_filter.T   # shape (patch_size, out_ch)

            # Formata array 2D em C: {{a,b,...},{...},...}
            rows = []
            for r in range(w_t.shape[0]):
                row_str = "{" + ",".join(map(str, w_t[r].tolist())) + "}"
                rows.append(row_str)
            w_cstr = "{" + ",".join(rows) + "}"

            h_lines.append(f"static const elem_t {lname}_w[{patch_size}][{out_ch}] row_align(1) = {w_cstr};\n")

            # bias
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"
            h_lines.append(f"static const acc_t {lname}_b[{out_ch}] row_align_acc(1) = {b_str};\n")

            # params
            batch_sz = batch
            # extrai um pad para o campo .padding (valor simples)
            padding = None
            if pads is not None:
                try:
                    padding = to_scalar(pads[0]) if hasattr(pads, '__len__') else to_scalar(pads)
                except Exception:
                    padding = to_scalar(pads)
            if padding is None:
                padding = 0
            n_patches = out_h * out_w * batch_sz
            output_scale = (1.0 / scaleW) if scaleW != 0 else 1.0
            depthwise = 1 if groups == in_ch else 0

            h_lines.append(
                f"static const struct ConvParams {lname}_params = {{"
                f".batch_size={batch_sz}, .in_row_dim={h_in}, .in_col_dim={w_in}, .kernel_size={kH}, "
                f".in_channels={in_ch}, .out_channels={out_ch}, .stride={stride_val}, .padding={padding}, "
                f".bias=1, .depthwise={depthwise}, .out_row_dim={out_h}, .out_col_dim={out_w}, "
                f".n_patches={n_patches}, .patch_size={patch_size}, .pool_size=1, .pool_stride=1, .pool_padding=0, "
                f".out_dim_pooled={out_h}, .output_scale=({output_scale:.12f}), .I={n_patches}, .J={out_ch}, .K={patch_size}}};\n"
            )

            # out buffer as 2D: [n_patches][out_ch]
            h_lines.append(f"static elem_t {lname}_out[{n_patches}][{out_ch}] row_align(1);\n")

            # map tensor name -> buffer name
            if node.output:
                tensor_buffer[node.output[0]] = f"{lname}_out"

            # detect activation after this conv
            next_nodes = [n for n in graph.node if n.input and n.input[0] == node.output[0]]
            act = 'RELU' if any(n.op_type == 'Relu' for n in next_nodes) else 'NONE'

            # c_lines: chamada tiled_conv_auto (mantive o template que você tinha)
            c_lines.append(f"  // {lname}\n")
            c_lines.append("  start = read_cycles();\n")
            c_lines.append(
                f"  tiled_conv_auto(\n"
                f"    {lname}_params.batch_size, {lname}_params.in_row_dim, {lname}_params.in_col_dim,\n"
                f"    {lname}_params.in_channels,\n"
                f"    {lname}_params.out_channels, {lname}_params.out_row_dim, {lname}_params.out_col_dim,\n"
                f"    {lname}_params.stride, 1, 1, {lname}_params.padding, {lname}_params.kernel_size,\n"
                f"    false, false, false, false, false,\n\n"
                f"    (elem_t*){tensor_buffer.get(X, 'images')}, (elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out,\n\n"
                f"    {act}, {lname}_params.output_scale,\n"
                f"    {lname}_params.pool_size, {lname}_params.pool_stride, {lname}_params.pool_padding,\n\n"
                f"    tiled_matmul_type);\n"
            )
            c_lines.append("  end = read_cycles();\n")
            c_lines.append(f"  conv_cycles += end - start;\n")
            c_lines.append(f'  printf("{lname} cycles: %llu\\n", end - start);\n')

            layer_idx += 1

        elif node.op_type == 'Gemm' or node.op_type == 'MatMul':
            A = node.input[0]
            Bn = node.input[1]
            Cn = node.input[2] if len(node.input) > 2 else None

            if Bn not in inits:
                print(f"Warning: fc weights {Bn} not found in initializers, skipping")
                continue
            B = inits[Bn]  # shape (out_features, in_features) or (K,N) depending
            # try to make sure we interpret shape as (out, in)
            if B.ndim != 2:
                print(f"Warning: unexpected weight shape for {Bn}: {B.shape}, skipping")
                continue

            if Cn and Cn in inits:
                C = inits[Cn]
            else:
                C = np.zeros(B.shape[0], dtype=np.float32)

            qW, scaleW = quantize_tensor(B.astype(np.float32), precision)
            qB, scaleB = quantize_tensor(C.astype(np.float32), precision)

            lname = f"fc_{layer_idx}"
            # flatten weights as [out_features][in_features] (row-major)
            out_dim, in_dim = qW.shape
            # format C 2D as rows
            rows = []
            for r in range(out_dim):
                rows.append("{" + ",".join(map(str, qW[r].tolist())) + "}")
            w_cstr = "{" + ",".join(rows) + "}"
            h_lines.append(f"static const elem_t {lname}_w[{out_dim}][{in_dim}] row_align(1) = {w_cstr};\n")
            h_lines.append(f"static const acc_t {lname}_b[{out_dim}] row_align_acc(1) = {{" + ",".join(map(str, qB.tolist())) + "}};\n")

            # out buffer
            h_lines.append(f"static elem_t {lname}_out[{out_dim}] row_align(1);\n")

            if node.output:
                tensor_buffer[node.output[0]] = f"{lname}_out"

            # activation detection
            next_nodes = [n for n in graph.node if n.input and node.output and n.input[0] == node.output[0]]
            act = 'RELU' if any(n.op_type == 'Relu' for n in next_nodes) else 'NONE'

            # matmul call (uso do template simplificado; ajustar se a assinatura da tua versão do gemmini for diferente)
            c_lines.append(f"  // {lname}\n")
            c_lines.append("  start = read_cycles();\n")
            c_lines.append(
                f"  tiled_matmul_nn_auto({out_dim}, 1, {in_dim}, (elem_t*){tensor_buffer.get(A,'images')}, (elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out, {act}, ({1.0/scaleW:.12f}), true, tiled_matmul_type, check, \"{lname}\");\n"
            )
            c_lines.append("  end = read_cycles();\n")
            c_lines.append(f"  matmul_cycles += end - start;\n")
            c_lines.append(f'  printf("{lname} cycles: %llu\\n", end - start);\n')

            layer_idx += 1

        else:
            # passthrough for activation/pool/etc: apenas propaga o buffer (mantém o mesmo nome)
            if node.output and node.input:
                in_buf_name = tensor_buffer.get(node.input[0])
                if in_buf_name:
                    tensor_buffer[node.output[0]] = in_buf_name

    # footer c
    c_lines.append("\n    uint64_t total_cycles = conv_cycles + matmul_cycles;\n\n")
    c_lines.append('    printf("\\nTotal cycles: %llu (100%%)\\n", total_cycles);\n')
    c_lines.append('    printf("Matmul cycles: %llu (%d%%)\\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);\n')
    c_lines.append('    printf("Conv cycles: %llu (%d%%)\\n", conv_cycles, (conv_cycles * 100) / total_cycles);\n')
    c_lines.append('    printf("PASS\\n");\n\n')
    c_lines.append('    exit(0);\n')
    c_lines.append('}\n')

    h_lines.append(f"#endif /* {guard} */\n")

    # grava arquivos
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
