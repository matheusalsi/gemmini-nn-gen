#!/usr/bin/env python3
"""
onnx2gemmini.py (Header Generator Only)

Suporta:
 - ONNX float32
 - ONNX já quantizado (INT4 / INT8 / UINT8 / INT16)
 - ONNX com QuantizeLinear (Brevitas / QONNX)
 - Extrai peso mesmo se passar por nós intermediários (Constant, QuantizeLinear, etc.)
 - Evita double quantization
 - Gera EXCLUSIVAMENTE o header de parâmetros (*_params.h)
"""

import onnx
import numpy as np
import os
import argparse
from typing import Optional, Tuple, Dict, Set

# ============================================================
# UTILS
# ============================================================

def to_scalar(x, default=None):
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    try:
        arr = np.array(x).flatten()
        return default if arr.size == 0 else int(arr.tolist()[0])
    except Exception:
        try:
            return int(x)
        except Exception:
            return default

def tensor_to_numpy(tensor):
    return onnx.numpy_helper.to_array(tensor)

def quantize_tensor_auto(tensor: np.ndarray, precision_bits=8, scale: Optional[float] = None):
    dtype = np.int8 if precision_bits in (4, 8) else np.int16
    
    if tensor.dtype in (np.int8, np.uint8, np.int16):
        q = tensor.astype(dtype)
        if scale is None:
            scale = 1.0
        return q, float(scale)

    qmax = (2 ** (precision_bits - 1)) - 1
    if scale is None:
        maxval = np.max(np.abs(tensor))
        scale = float(maxval / qmax) if maxval != 0 else 1.0
    
    q = np.round(tensor / scale).astype(dtype)
    
    if precision_bits == 4:
        q = np.clip(q, -8, 7)
        
    return q, float(scale)

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default

def compute_conv_output(h_in, w_in, k, stride, pads, dilation):
    h_in, w_in = to_scalar(h_in), to_scalar(w_in)
    if h_in is None or w_in is None:
        raise RuntimeError("Input spatial dims unknown.")
    stride_val, dilation_val = to_scalar(stride, 1), to_scalar(dilation, 1)
    pad_h = pads[0] + pads[2] if pads and len(pads) == 4 else 0
    pad_w = pads[1] + pads[3] if pads and len(pads) == 4 else 0
    kH = kW = to_scalar(k, 1)
    out_h = (h_in + pad_h - dilation_val * (kH - 1) - 1) // stride_val + 1
    out_w = (w_in + pad_w - dilation_val * (kW - 1) - 1) // stride_val + 1
    return int(out_h), int(out_w)

def find_maxpool_after(start_name, consumers, graph):
    """ Busca (BFS) o primeiro nó de MaxPool ignorando ReLUs e Quants """
    queue = [start_name]
    visited = set()
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        if curr in consumers:
            for node in consumers[curr]:
                if node.op_type == 'MaxPool':
                    return node
                elif node.op_type not in ('Conv', 'Gemm'):
                    queue.extend(node.output)
    return None

# ============================================================
# RESOLVER RECURSIVO DE PESOS
# ============================================================

def build_index(graph):
    producers_by_output: Dict[str, onnx.NodeProto] = {}
    constants_by_name: Dict[str, np.ndarray] = {}

    for node in graph.node:
        for out in node.output:
            producers_by_output[out] = node

        if node.op_type == "Constant":
            val = get_attr(node, "value", None)
            if isinstance(val, onnx.TensorProto):
                try:
                    constants_by_name[node.output[0]] = tensor_to_numpy(val)
                except Exception:
                    pass

    return producers_by_output, constants_by_name

def resolve_input_tensor(name: str, graph: onnx.GraphProto, inits: Dict[str, np.ndarray],
                         producers_by_output: Dict[str, onnx.NodeProto],
                         constants_by_name: Dict[str, np.ndarray],
                         visited: Optional[Set[str]] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if visited is None:
        visited = set()
    if name in visited:
        return None, None
    visited.add(name)

    if name in inits:
        return inits[name], None
    if name in constants_by_name:
        return constants_by_name[name], None

    node = producers_by_output.get(name, None)
    if node is None:
        return None, None

    if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
        real_name = node.input[0]
        scale_name = node.input[1] if len(node.input) > 1 else None
        W, _ = resolve_input_tensor(real_name, graph, inits, producers_by_output, constants_by_name, visited)
        
        scale_val = None
        if scale_name:
            if scale_name in inits:
                arr = inits[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            elif scale_name in constants_by_name:
                arr = constants_by_name[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            else:
                s, _ = resolve_input_tensor(scale_name, graph, inits, producers_by_output, constants_by_name, visited)
                if s is not None:
                    arr = np.array(s).reshape(-1)
                    scale_val = float(arr[0]) if arr.size > 0 else None
        return W, scale_val

    if node.op_type == "Constant":
        val = get_attr(node, "value", None)
        if isinstance(val, onnx.TensorProto):
            try:
                return tensor_to_numpy(val), None
            except Exception:
                pass

    for inp in node.input:
        if not inp:
            continue
        W, s = resolve_input_tensor(inp, graph, inits, producers_by_output, constants_by_name, visited)
        if W is not None:
            return W, s

    return None, None

def extract_quantized_weight(W_name: str, graph: onnx.GraphProto, inits: Dict[str, np.ndarray]):
    producers_by_output, constants_by_name = build_index(graph)

    if W_name in inits:
        return inits[W_name], 1.0

    W, scale = resolve_input_tensor(W_name, graph, inits, producers_by_output, constants_by_name, visited=set())

    if W is not None:
        return W, (float(scale) if scale is not None else 1.0)

    available_inits = list(inits.keys())
    similar = [n for n in available_inits if n in W_name or W_name in n or n.split('/')[-1] in W_name or W_name.split('/')[-1] in n]
    msg_lines = [
        f"Peso não encontrado: {W_name}",
        f"Initializers disponíveis ({len(available_inits)}): {available_inits[:20]}{'...' if len(available_inits)>20 else ''}",
        f"Initializers com nomes semelhantes: {similar if similar else 'nenhum similar encontrado'}"
    ]
    raise KeyError("\n".join(msg_lines))


# ============================================================
# EXPORTADOR (HEADER ONLY)
# ============================================================

def export_gemmini_params(onnx_path, out_dir='out', precision=8, batch_size=4):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    graph = model.graph

    inits = {t.name: tensor_to_numpy(t) for t in graph.initializer}
    
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, C, H, W = input_shape if len(input_shape) == 4 else (None, None, None, None)

    basename = os.path.basename(out_dir)
    h_lines = []

    guard = f"{basename.upper()}_PARAMETERS_H"
    h_lines.append(f"#ifndef {guard}\n#define {guard}\n\n#include <include/gemmini_params.h>\n#include <stdbool.h>\n\n")

    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    layer_idx = 1
    output_dims = {}
    inp_name = graph.input[0].name
    output_dims[inp_name] = (C, H, W)
    processed_nodes = set()

    for node in graph.node:
        if node.name in processed_nodes:
            continue

        # =========================
        # CONV
        # =========================
        if node.op_type == 'Conv':
            X_name, W_name = node.input[0], node.input[1]
            B_name = node.input[2] if len(node.input) > 2 else None
            Y_name = node.output[0]

            W_data, scale_real = extract_quantized_weight(W_name, graph, inits)
            
            # Extração segura do Bias (busca recursiva ou fallback para zeros)
            if B_name:
                try:
                    B_data, _ = extract_quantized_weight(B_name, graph, inits)
                except KeyError:
                    B_data = np.zeros(W_data.shape[0], dtype=np.float32)
            else:
                B_data = np.zeros(W_data.shape[0], dtype=np.float32)

            out_ch, in_ch_per_group, kH, _ = W_data.shape
            groups = get_attr(node, 'group', 1)
            in_ch = in_ch_per_group * groups

            strides_attr = get_attr(node, 'strides', [1, 1])
            pads_attr = get_attr(node, 'pads', [0, 0, 0, 0])

            _, h_in, w_in = output_dims.get(X_name, (in_ch, 224, 224))
            out_h, out_w = compute_conv_output(h_in, w_in, kH, strides_attr, pads_attr, [1, 1])
            
            # --- BFS LOOKAHEAD PARA MAXPOOL ---
            pool_size, pool_stride, pool_padding = 1, 1, 0
            out_dim_pooled_h, out_dim_pooled_w = out_h, out_w
            final_Y_name = Y_name
            
            pool_node = find_maxpool_after(Y_name, consumers, graph)
            if pool_node:
                k_shape = get_attr(pool_node, 'kernel_shape', [1, 1])
                p_strides = get_attr(pool_node, 'strides', [1, 1])
                p_pads = get_attr(pool_node, 'pads', [0, 0, 0, 0])
                
                pool_size = k_shape[0]
                pool_stride = p_strides[0]
                pool_padding = p_pads[0]
                pad_h = p_pads[0] + p_pads[2] if len(p_pads) == 4 else 0
                pad_w = p_pads[1] + p_pads[3] if len(p_pads) == 4 else 0
                
                out_dim_pooled_h = (out_h + pad_h - pool_size) // pool_stride + 1
                out_dim_pooled_w = (out_w + pad_w - pool_size) // pool_stride + 1
                
                final_Y_name = pool_node.output[0]
                output_dims[final_Y_name] = (out_ch, out_dim_pooled_h, out_dim_pooled_w)

            output_dims[Y_name] = (out_ch, out_h, out_w)

            qW, scaleW = quantize_tensor_auto(W_data, precision, scale_real)
            qB, _ = quantize_tensor_auto(B_data, precision)

            lname = f"conv_{layer_idx}"
            patch_size = in_ch * kH * kH

            w_t = qW.reshape(out_ch, -1).T
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in w_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"

            n_patches = out_h * out_w * batch_size

            h_lines.extend([
                f"static const elem_t {lname}_w[{patch_size}][{out_ch}] row_align(1) = {w_cstr};\n",
                f"static const acc_t {lname}_b[{out_ch}] row_align_acc(1) = {b_str};\n",
                f"static elem_t {lname}_in[{n_patches}][{patch_size}] row_align(1);\n",
                f"static elem_t {lname}_out[{n_patches}][{out_ch}] row_align(1);\n"
            ])

            if pool_size > 1:
                pooled_patches = batch_size * out_dim_pooled_h * out_dim_pooled_w
                h_lines.append(f"static elem_t {lname}_out_pooled[{pooled_patches}][{out_ch}] row_align(1);\n")

            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0

            h_lines.append(
                f"static const struct ConvParams {lname}_params = {{\n"
                f"    .batch_size={batch_size}, .in_row_dim={h_in}, .in_col_dim={w_in},\n"
                f"    .kernel_size={kH}, .in_channels={in_ch}, .out_channels={out_ch},\n"
                f"    .stride={to_scalar(strides_attr)}, .padding={to_scalar(pads_attr)},\n"
                f"    .bias=1, .depthwise={1 if groups == in_ch else 0},\n"
                f"    .out_row_dim={out_h}, .out_col_dim={out_w},\n"
                f"    .n_patches={n_patches}, .patch_size={patch_size},\n"
                f"    .pool_size={pool_size}, .pool_stride={pool_stride}, .pool_padding={pool_padding},\n"
                f"    .out_dim_pooled={out_dim_pooled_h}, .output_scale=(1.0f / (1 << {shift})),\n"
                f"    .res_scale=1.0f,\n"
                f"    .I={n_patches}, .J={out_ch}, .K={patch_size}\n"
                f"}};\n\n"
            )

            layer_idx += 1
            processed_nodes.add(node.name)

        # =========================
        # GEMM
        # =========================
        elif node.op_type == 'Gemm':
            A_name = node.input[0]
            W_name = node.input[1]
            B_name = node.input[2] if len(node.input) > 2 else None
            Y_name = node.output[0]

            W_data, scale_real = extract_quantized_weight(W_name, graph, inits)
            
            # Extração segura do Bias para a Fully Connected
            if B_name:
                try:
                    B_data, _ = extract_quantized_weight(B_name, graph, inits)
                except KeyError:
                    B_data = np.zeros(W_data.shape[0], dtype=np.float32)
            else:
                B_data = np.zeros(W_data.shape[0], dtype=np.float32)

            out_features, in_features = W_data.shape

            qW, scaleW = quantize_tensor_auto(W_data, precision, scale_real)
            qB, _ = quantize_tensor_auto(B_data, precision)

            qW_t = qW.T
            lname = f"fc_{layer_idx}"

            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in qW_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"

            h_lines.extend([
                f"static const elem_t {lname}_w[{in_features}][{out_features}] row_align(1) = {w_cstr};\n",
                f"static const acc_t {lname}_b[{out_features}] row_align_acc(1) = {b_str};\n",
                f"static elem_t {lname}_out[{batch_size}][{out_features}] row_align(1);\n"
            ])

            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0

            h_lines.append(
                f"static const struct FcParams {lname}_params = {{\n"
                f"    .batch_size={batch_size}, .in_features={in_features},\n"
                f"    .out_features={out_features}, .bias=1,\n"
                f"    .output_scale=(1.0f / (1 << {shift})),\n"
                f"    .I={batch_size}, .J={out_features}, .K={in_features}\n"
                f"}};\n\n"
            )

            layer_idx += 1
            processed_nodes.add(node.name)

        else:
            if not node.input or not node.output:
                continue
            input_name = node.input[0]
            output_name = node.output[0]
            if output_name not in output_dims and input_name in output_dims:
                output_dims[output_name] = output_dims[input_name]
            processed_nodes.add(node.name)

    h_lines.append(f"#endif /* {guard} */\n")

    out_file = os.path.join(out_dir, f'{basename}_params_int{precision}.h')
    with open(out_file, 'w') as f:
        f.write(''.join(h_lines))

    print(f"Header file successfully generated at: {out_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Gera headers do Gemmini baseados em modelos ONNX.")
    p.add_argument('onnx', help='ONNX model path')
    p.add_argument('--out', default='out', help='output directory')
    p.add_argument('--precision', type=int, default=8, choices=[4, 8, 16], help='Bit width target')
    p.add_argument('--batch_size', type=int, default=4)
    args = p.parse_args()

    export_gemmini_params(args.onnx, out_dir=args.out, precision=args.precision, batch_size=args.batch_size)