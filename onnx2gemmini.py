import onnx
import numpy as np
import os
import argparse

def to_scalar(x, default=None):
    """
    Converte x para um int escalar de maneira robusta.
    """
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    try:
        arr = np.array(x).flatten()
        if arr.size == 0:
            return default
        return int(arr.tolist()[0])
    except Exception:
        try:
            return int(x)
        except Exception:
            return default

def tensor_to_numpy(tensor):
    if tensor.data_type == onnx.TensorProto.FLOAT:
        dtype = np.float32
        data = tensor.float_data or np.frombuffer(tensor.raw_data, dtype=dtype)
    elif tensor.data_type == onnx.TensorProto.INT32:
        dtype = np.int32
        data = tensor.int32_data or np.frombuffer(tensor.raw_data, dtype=dtype)
    elif tensor.data_type == onnx.TensorProto.INT64:
        dtype = np.int64
        data = tensor.int64_data or np.frombuffer(tensor.raw_data, dtype=dtype)
    else:
        raise NotImplementedError(f"ONNX tensor dtype {tensor.data_type} not implemented")
    return np.array(data, dtype=dtype).reshape(tensor.dims)

def quantize_tensor(tensor_f32, precision_bits=8):
    qmax = (2 ** (precision_bits - 1)) - 1
    dtype = np.int8 if precision_bits == 8 else np.int16
    
    maxval = np.max(np.abs(tensor_f32))
    scale = float(maxval) / float(qmax) if maxval != 0 else 1.0
    
    q = np.round(tensor_f32 / scale).astype(dtype)
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

    stride_val = to_scalar(stride, 1)
    dilation_val = to_scalar(dilation, 1)

    if pads is not None and len(pads) == 4:
        pad_h = pads[0] + pads[2]
        pad_w = pads[1] + pads[3]
    else:
        pad_h = pad_w = 0

    kH = kW = to_scalar(k, 1)

    out_h = (h_in + pad_h - dilation_val * (kH - 1) - 1) // stride_val + 1
    out_w = (w_in + pad_w - dilation_val * (kW - 1) - 1) // stride_val + 1
    return int(out_h), int(out_w)


def export_gemmini(onnx_path, out_dir='out', precision=8, batch_size=4):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    graph = model.graph
    
    inits = {t.name: tensor_to_numpy(t) for t in graph.initializer}
    
    node_by_input = {i: [] for node in graph.node for i in node.input}
    for node in graph.node:
        for i in node.input:
            if i: 
                node_by_input[i].append(node)

    batch = batch_size
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, C, H, W = input_shape if len(input_shape) == 4 else (None, None, None, None)

    basename = os.path.basename(out_dir)
    h_lines = []
    c_lines = []

    # Boilerplate...
    guard = f"{basename.upper()}_PARAMETERS_H"
    h_lines.append(f"#ifndef {guard}\n#define {guard}\n\n#include <include/gemmini_params.h>\n#include <stdbool.h>\n\n")
    c_lines.extend(['#include <stdio.h>\n', '#include <string.h>\n', '#include <stdbool.h>\n', '#ifndef BAREMETAL\n', '#include <sys/mman.h>\n', '#endif\n', '#include "include/gemmini.h"\n', '#include "include/gemmini_nn.h"\n\n', f'#include "{basename}_params.h"\n', '#include "images.h"\n\n', 'int main (int argc, char * argv[]) {\n', '#ifndef BAREMETAL\n', '    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {\n', '      perror("mlockall failed");\n', '      exit(1);\n', '    }\n', '#endif\n\n', '    gemmini_flush(0);\n\n', '    enum tiled_matmul_type_t tiled_matmul_type = WS;\n', '    if (argc < 2) {\n', '        tiled_matmul_type = WS;\n', '    } else if (strcmp(argv[1], "cpu") == 0) {\n', '        tiled_matmul_type = CPU;\n', '    } else if (strcmp(argv[1], "os") == 0) {\n', '        tiled_matmul_type = OS;\n', '    } else if (strcmp(argv[1], "ws") == 0) {\n', '        tiled_matmul_type = WS;\n', '    } else if (strcmp(argv[1], "-h") == 0) {\n', '        printf("usage: %s [-h] matmul_option [check]\\n  matmul_option may be \'os\', \'ws\', or cpu\'\\n", argv[0]);\n', '        exit(0);\n', '    } else {\n', '        printf("Unknown command-line argument\\n");\n', '        exit(1);\n', '    }\n\n', '    bool check=false;\n', '    if (argc > 2) {\n', '        if (strcmp(argv[2], "check") == 0) {\n', '            check = true;\n', '        } else {\n', '            printf("Unknown command-line argument\\n");\n', '            exit(1);\n', '        }\n', '    }\n\n', '    uint64_t start, end;\n', '    uint64_t conv_cycles = 0, matmul_cycles = 0;\n\n', '    // model execution\n'])

    layer_idx = 1
    tensor_buffer = {}
    output_dims = {}

    inp_name = graph.input[0].name
    tensor_buffer[inp_name] = 'images'
    output_dims[inp_name] = (C, H, W)

    processed_nodes = set()

    for node in graph.node:
        if node.name in processed_nodes:
            continue

        if node.op_type == 'Conv':
            X_name = node.input[0]
            W_name = node.input[1]
            B_name = node.input[2] if len(node.input) > 2 else None
            Y_name = node.output[0]
            W_data = inits[W_name]
            out_ch, in_ch_per_group, kH, kW = W_data.shape
            groups = get_attr(node, 'group', 1)
            in_ch = in_ch_per_group * groups
            strides_attr = get_attr(node, 'strides', [1, 1])
            pads_attr = get_attr(node, 'pads', [0, 0, 0, 0])
            dilations_attr = get_attr(node, 'dilations', [1, 1])
            in_ch_dim, h_in, w_in = output_dims[X_name]
            out_h, out_w = compute_conv_output(h_in, w_in, kH, strides_attr, pads_attr, dilations_attr)
            output_dims[Y_name] = (out_ch, out_h, out_w)
            B_data = inits.get(B_name, np.zeros(out_ch, dtype=np.float32))
            qW, scaleW = quantize_tensor(W_data.astype(np.float32), precision)
            qB, _ = quantize_tensor(B_data.astype(np.float32), precision)
            lname = f"conv_{layer_idx}"
            patch_size = in_ch * kH * kW
            w_t = qW.reshape(out_ch, -1).T
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in w_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"
            n_patches_conv = out_h * out_w * batch
            h_lines.append(f"static const elem_t {lname}_w[{patch_size}][{out_ch}] row_align(1) = {w_cstr};\n")
            h_lines.append(f"static const acc_t {lname}_b[{out_ch}] row_align_acc(1) = {b_str};\n")
            h_lines.append(f"static elem_t {lname}_in[{n_patches_conv}][{patch_size}] row_align(1);\n")
            h_lines.append(f"static elem_t {lname}_out[{n_patches_conv}][{out_ch}] row_align(1);\n")
            tensor_buffer[Y_name] = f"{lname}_out"
            act = 'NONE'
            pool_params = {'pool_size': 1, 'pool_stride': 1, 'pool_padding': 0}
            out_h_pooled, out_w_pooled = out_h, out_w
            current_tensor_name = Y_name
            pool_node = None
            while True:
                next_consumers = node_by_input.get(current_tensor_name, [])
                if not next_consumers: break
                consumer = next_consumers[0]
                if consumer.op_type == 'Relu' and act == 'NONE':
                    act = 'RELU'
                    processed_nodes.add(consumer.name)
                    current_tensor_name = consumer.output[0]
                    output_dims[current_tensor_name] = output_dims[Y_name]
                    tensor_buffer[current_tensor_name] = tensor_buffer[Y_name]
                elif consumer.op_type in ('MaxPool', 'AveragePool'):
                    pool_node = consumer
                    processed_nodes.add(pool_node.name)
                    break
                else: break
            if pool_node:
                pool_k = get_attr(pool_node, 'kernel_shape', [1, 1])
                pool_s = get_attr(pool_node, 'strides', [1, 1])
                pool_p = get_attr(pool_node, 'pads', [0, 0, 0, 0])
                pool_params = {'pool_size': to_scalar(pool_k), 'pool_stride': to_scalar(pool_s), 'pool_padding': to_scalar(pool_p)}
                out_h_pooled, out_w_pooled = compute_conv_output(out_h, out_w, pool_params['pool_size'], pool_params['pool_stride'], pool_p, 1)
                h_lines.append(f"static elem_t {lname}_out_pooled[{batch}][{out_h_pooled}][{out_w_pooled}][{out_ch}];\n")
                pool_out_name = pool_node.output[0]
                output_dims[pool_out_name] = (out_ch, out_h_pooled, out_w_pooled)
                tensor_buffer[pool_out_name] = f"{lname}_out_pooled"
            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0
            params_str = (f"static const struct ConvParams {lname}_params = {{"
                          f".batch_size={batch}, .in_row_dim={h_in}, .in_col_dim={w_in}, "
                          f".kernel_size={kH}, .in_channels={in_ch}, .out_channels={out_ch}, "
                          f".stride={to_scalar(strides_attr)}, .padding={to_scalar(pads_attr)}, "
                          f".bias=1, .depthwise={1 if groups == in_ch else 0}, "
                          f".out_row_dim={out_h}, .out_col_dim={out_w}, "
                          f".n_patches={n_patches_conv}, .patch_size={patch_size}, "
                          f".pool_size={pool_params['pool_size']}, .pool_stride={pool_params['pool_stride']}, .pool_padding={pool_params['pool_padding']}, "
                          f".out_dim_pooled={out_h_pooled}, .output_scale=(1.0 / (1 << {shift})), "
                          f".I={n_patches_conv}, .J={out_ch}, .K={patch_size}}};")
            h_lines.append(params_str)
            h_lines.append("\n\n\n")
            
            c_lines.append(f"    // {lname}\n")
            c_lines.append("    start = read_cycles();\n")
            c_lines.append(f"    tiled_conv_auto(\n        {lname}_params.batch_size, {lname}_params.in_row_dim, {lname}_params.in_col_dim,\n        {lname}_params.in_channels,\n        {lname}_params.out_channels, {lname}_params.out_row_dim, {lname}_params.out_col_dim,\n        {lname}_params.stride, 1, 1, {lname}_params.padding, {lname}_params.kernel_size,\n        false, false, false, false, false,\n\n        (elem_t*){tensor_buffer.get(X_name, 'images')}, (elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out,\n\n        {act}, {lname}_params.output_scale,\n        {lname}_params.pool_size, {lname}_params.pool_stride, {lname}_params.pool_padding,\n\n        tiled_matmul_type);\n")
            c_lines.append("    end = read_cycles();\n")
            c_lines.append(f"    conv_cycles += end - start;\n")
            layer_idx += 1
            processed_nodes.add(node.name)

        elif node.op_type == 'GlobalAveragePool':
            input_name = node.input[0]
            output_name = node.output[0]
            in_dims = output_dims.get(input_name)
            if in_dims:
                output_dims[output_name] = (in_dims[0], 1, 1)
                tensor_buffer[output_name] = tensor_buffer.get(input_name)
            processed_nodes.add(node.name)
            
        elif node.op_type == 'Gemm':
            A_name = node.input[0]
            W_name = node.input[1]
            B_name = node.input[2]
            Y_name = node.output[0]

            W_data = inits[W_name]
            B_data = inits[B_name]
            
            out_features, in_features = W_data.shape

            qW, scaleW = quantize_tensor(W_data.astype(np.float32), precision)
            qB, _ = quantize_tensor(B_data.astype(np.float32), precision)
            
            # Gemmini espera pesos transpostos
            qW_t = qW.T

            lname = f"fc_{layer_idx}"
            
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in qW_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"

            h_lines.append(f"static const elem_t {lname}_w[{in_features}][{out_features}] row_align(1) = {w_cstr};\n")
            h_lines.append(f"static const acc_t {lname}_b[{batch_size}][{out_features}] row_align_acc(1) = {{{b_str}}};\n")
            h_lines.append(f"static elem_t {lname}_out[{batch_size}][{out_features}] row_align(1);\n")
            
            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0
            
            fc_params_str = (f"static const struct FcParams {lname}_params = {{"
                             f".batch_size={batch_size}, .in_features={in_features}, .out_features={out_features}, "
                             f".bias=1, .output_scale=(1.0 / (1 << {shift})), "
                             f".I={batch_size}, .J={out_features}, .K={in_features}}};")
            h_lines.append(fc_params_str)
            h_lines.append("\n\n\n")
            
            tensor_buffer[Y_name] = f"{lname}_out"
            
            c_lines.append(f"    // {lname}\n")
            c_lines.append("    start = read_cycles();\n")
            c_lines.append(f"    tiled_matmul_nn_auto({batch_size}, {out_features}, {in_features}, "
                           f"(elem_t*){tensor_buffer[A_name]}, (elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out, "
                           f"NO_ACTIVATION, 1.0, true, tiled_matmul_type, false, \"{lname}\");\n")
            c_lines.append("    end = read_cycles();\n")
            c_lines.append(f"    matmul_cycles += end - start;\n")
            
            layer_idx += 1
            processed_nodes.add(node.name)

        elif node.input and node.output:
            input_name = node.input[0]
            output_name = node.output[0]
            if input_name in output_dims:
                output_dims[output_name] = output_dims[input_name]
                if input_name in tensor_buffer:
                    tensor_buffer[output_name] = tensor_buffer[input_name]
                processed_nodes.add(node.name)

    # Footer...
    c_lines.extend(["\n    uint64_t total_cycles = conv_cycles + matmul_cycles;\n\n", '    printf("\\nTotal cycles: %llu (100%%)\\n", total_cycles);\n', '    if(total_cycles != 0) {\n', '        printf("Matmul cycles: %llu (%d%%)\\n", matmul_cycles, (int)((matmul_cycles * 100) / total_cycles));\n', '        printf("Conv cycles: %llu (%d%%)\\n", conv_cycles, (int)((conv_cycles * 100) / total_cycles));\n', '    }\n','    printf("PASS\\n");\n\n', '    exit(0);\n', '}\n'])
    h_lines.append(f"#endif\n\n")

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