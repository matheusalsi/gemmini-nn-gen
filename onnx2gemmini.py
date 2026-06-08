#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Optional, Set, Tuple

import numpy as np
import onnx


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


def tensor_to_numpy(tensor: onnx.TensorProto) -> np.ndarray:
    return onnx.numpy_helper.to_array(tensor)


def quantize_tensor_auto(
    tensor: np.ndarray, precision_bits: int = 8, scale: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    dtype = np.int8 if precision_bits == 8 else np.int16
    already_int = tensor.dtype in (np.int8, np.uint8, np.int16)

    # Widening or maintaining width: keep integers as they are
    if already_int:
        return tensor.astype(dtype), float(scale if scale is not None else 1.0)

    qmax = (2 ** (precision_bits - 1)) - 1
    if scale is None:
        maxval = np.max(np.abs(tensor))
        scale = float(maxval / qmax) if maxval != 0 else 1.0

    # Clip BEFORE astype, otherwise values > maxval wrap around
    q = np.clip(np.round(tensor / scale), -qmax - 1, qmax)
    q = q.astype(dtype)
    return q, float(scale)


def get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def compute_conv_output(h_in, w_in, k, stride, pads, dilation) -> Tuple[int, int]:
    h_in, w_in = to_scalar(h_in), to_scalar(w_in)
    if h_in is None or w_in is None:
        raise RuntimeError("Input spatial dims unknown.")
    
    stride_val, dilation_val = to_scalar(stride, 1), to_scalar(dilation, 1)
    pad_h = pads[0] + pads[2] if pads and len(pads) == 4 else 0
    pad_w = pads[1] + pads[3] if pads and len(pads) == 4 else 0
    k_h = k_w = to_scalar(k, 1)
    
    out_h = (h_in + pad_h - dilation_val * (k_h - 1) - 1) // stride_val + 1
    out_w = (w_in + pad_w - dilation_val * (k_w - 1) - 1) // stride_val + 1
    return int(out_h), int(out_w)


def find_maxpool_after(
    start_name: str, consumers: Dict[str, list], graph: onnx.GraphProto
) -> Optional[onnx.NodeProto]:
    """ BFS search for the first MaxPool node, ignoring ReLUs and Quants. """
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


def build_index(
    graph: onnx.GraphProto
) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, np.ndarray]]:
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


def resolve_input_tensor(
    name: str,
    graph: onnx.GraphProto,
    inits: Dict[str, np.ndarray],
    producers_by_output: Dict[str, onnx.NodeProto],
    constants_by_name: Dict[str, np.ndarray],
    visited: Optional[Set[str]] = None
) -> Tuple[Optional[np.ndarray], Optional[float]]:
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
        w_val, _ = resolve_input_tensor(
            real_name, graph, inits, producers_by_output, constants_by_name, visited
        )
        
        scale_val = None
        if scale_name:
            if scale_name in inits:
                arr = inits[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            elif scale_name in constants_by_name:
                arr = constants_by_name[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            else:
                s_val, _ = resolve_input_tensor(
                    scale_name, graph, inits, producers_by_output, constants_by_name, visited
                )
                if s_val is not None:
                    arr = np.array(s_val).reshape(-1)
                    scale_val = float(arr[0]) if arr.size > 0 else None
        return w_val, scale_val

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
        w_val, s_val = resolve_input_tensor(
            inp, graph, inits, producers_by_output, constants_by_name, visited
        )
        if w_val is not None:
            return w_val, s_val

    return None, None


def extract_quantized_weight(
    w_name: str, graph: onnx.GraphProto, inits: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, float]:
    producers_by_output, constants_by_name = build_index(graph)

    if w_name in inits:
        return inits[w_name], 1.0

    w_val, scale = resolve_input_tensor(
        w_name, graph, inits, producers_by_output, constants_by_name, visited=set()
    )

    if w_val is not None:
        return w_val, (float(scale) if scale is not None else 1.0)

    available_inits = list(inits.keys())
    similar = [
        n for n in available_inits 
        if n in w_name or w_name in n or n.split('/')[-1] in w_name or w_name.split('/')[-1] in n
    ]
    msg_lines = [
        f"Weight not found: {w_name}",
        f"Available initializers ({len(available_inits)}): {available_inits[:20]}{'...' if len(available_inits)>20 else ''}",
        f"Initializers with similar names: {similar if similar else 'none found'}"
    ]
    raise KeyError("\n".join(msg_lines))


def export_gemmini_params(
    onnx_path: str, out_dir: str = 'out', precision: int = 8, batch_size: int = 4
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    graph = model.graph

    inits = {t.name: tensor_to_numpy(t) for t in graph.initializer}
    
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, c_in, h_in, w_in = input_shape if len(input_shape) == 4 else (None, None, None, None)

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
    output_dims[inp_name] = (c_in, h_in, w_in)
    processed_nodes = set()

    for node in graph.node:
        if node.name in processed_nodes:
            continue

        # =========================
        # CONV
        # =========================
        if node.op_type == 'Conv':
            x_name, w_name = node.input[0], node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else None
            y_name = node.output[0]

            w_data, scale_real = extract_quantized_weight(w_name, graph, inits)
            
            # Safe bias extraction (recursive search or fallback to zeros)
            if b_name:
                try:
                    b_data, _ = extract_quantized_weight(b_name, graph, inits)
                except KeyError:
                    b_data = np.zeros(w_data.shape[0], dtype=np.float32)
            else:
                b_data = np.zeros(w_data.shape[0], dtype=np.float32)

            out_ch, in_ch_per_group, k_h, _ = w_data.shape
            groups = get_attr(node, 'group', 1)
            in_ch = in_ch_per_group * groups

            strides_attr = get_attr(node, 'strides', [1, 1])
            pads_attr = get_attr(node, 'pads', [0, 0, 0, 0])

            _, feat_h_in, feat_w_in = output_dims.get(x_name, (in_ch, 224, 224))
            out_h, out_w = compute_conv_output(feat_h_in, feat_w_in, k_h, strides_attr, pads_attr, [1, 1])
            
            # --- BFS LOOKAHEAD FOR MAXPOOL ---
            pool_size, pool_stride, pool_padding = 1, 1, 0
            out_dim_pooled_h, out_dim_pooled_w = out_h, out_w
            final_y_name = y_name
            
            pool_node = find_maxpool_after(y_name, consumers, graph)
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
                
                final_y_name = pool_node.output[0]
                output_dims[final_y_name] = (out_ch, out_dim_pooled_h, out_dim_pooled_w)

            output_dims[y_name] = (out_ch, out_h, out_w)

            qw, scale_w = quantize_tensor_auto(w_data, precision, scale_real)
            qb, _ = quantize_tensor_auto(b_data, precision)

            lname = f"conv_{layer_idx}"
            patch_size = in_ch * k_h * k_h

            w_t = qw.reshape(out_ch, -1).T
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in w_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qb.tolist())) + "}"

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

            shift = int(np.round(np.log2(1.0 / scale_w))) if scale_w > 0 else 0

            h_lines.append(
                f"static const struct ConvParams {lname}_params = {{\n"
                f"    .batch_size={batch_size}, .in_row_dim={feat_h_in}, .in_col_dim={feat_w_in},\n"
                f"    .kernel_size={k_h}, .in_channels={in_ch}, .out_channels={out_ch},\n"
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
            a_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else None
            y_name = node.output[0]

            w_data, scale_real = extract_quantized_weight(w_name, graph, inits)
            
            # Safe bias extraction for Fully Connected
            if b_name:
                try:
                    b_data, _ = extract_quantized_weight(b_name, graph, inits)
                except KeyError:
                    b_data = np.zeros(w_data.shape[0], dtype=np.float32)
            else:
                b_data = np.zeros(w_data.shape[0], dtype=np.float32)

            out_features, in_features = w_data.shape

            qw, scale_w = quantize_tensor_auto(w_data, precision, scale_real)
            qb, _ = quantize_tensor_auto(b_data, precision)

            qw_t = qw.T
            lname = f"fc_{layer_idx}"

            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in qw_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qb.tolist())) + "}"

            h_lines.extend([
                f"static const elem_t {lname}_w[{in_features}][{out_features}] row_align(1) = {w_cstr};\n",
                f"static const acc_t {lname}_b[{out_features}] row_align_acc(1) = {b_str};\n",
                f"static elem_t {lname}_out[{batch_size}][{out_features}] row_align(1);\n"
            ])

            shift = int(np.round(np.log2(1.0 / scale_w))) if scale_w > 0 else 0

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
    parser = argparse.ArgumentParser(
        description="Generates Gemmini headers based on ONNX models."
    )
    parser.add_argument('onnx', help='ONNX model path')
    parser.add_argument('--out', default='out', help='Output directory')
    # Removed 4 from choices
    parser.add_argument(
        '--precision', type=int, default=8, choices=[8, 16], help='Bit width target'
    )
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()

    export_gemmini_params(
        args.onnx, 
        out_dir=args.out, 
        precision=args.precision, 
        batch_size=args.batch_size
    )