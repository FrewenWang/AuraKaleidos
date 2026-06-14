"""
Replace backbone in an ONNX model at a specified tensor.

Automated pipeline: split → merge → simplify → verify

Usage (CLI):
    python replace_onnx.py \
        --original full_model.onnx \
        --backbone new_backbone.onnx \
        --tensor_name "/_model/model/conv_exp/activation/Mul_output_0" \
        -o replaced_model.onnx

Usage (as library):
    from replace_onnx import replace_backbone, split_model, merge_models

    # One-step
    replace_backbone("full.onnx", "backbone.onnx", "tensor_name", "output.onnx")

    # Or manual control
    _, tail = split_model("full.onnx", "tensor_name")
    merge_models("backbone.onnx", tail, "tensor_name", "output.onnx")
"""

import argparse
import os
from collections import deque
from copy import deepcopy
from typing import Dict, Set, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference


# ============================================================
# Core Functions
# ============================================================


def _find_upstream_node_indices(graph: onnx.GraphProto, tensor_name: str) -> Set[int]:
    """BFS from tensor_name upward, return set of upstream node indices."""
    print(f"Finding upstream nodes from {tensor_name}")
    output_to_idx = {}
    for idx, node in enumerate(graph.node):
        print(f"Node: {node.name},  idx: {idx}, outputs: {node.output}")
        for out in node.output:
            output_to_idx[out] = idx
    # 
    nodes = list(graph.node)
    # print(f"Found {len(nodes)} nodes")
    # print(f"Starting from {tensor_name}")
    # print(f"Outputs: {[n.output[0] for n in nodes]}")
    
    visited = set()
    node_indices = set()
    # 
    queue = deque([tensor_name])
    print(f"Queue: {queue}")
    while queue:
        t = queue.popleft()
        if t in visited:
            continue
        visited.add(t)
        if t in output_to_idx:
            idx = output_to_idx[t]
            node_indices.add(idx)
            for inp in nodes[idx].input:
                if inp:
                    queue.append(inp)
    # print(f"Visited: {visited}")     
    # print(f"Upstream node indices: {node_indices}")
    # print(f"Downstream node indices: {node_indices}")   
    # print(f"queue: {queue}")   
    return node_indices


def split_model(
    model_path: str, tensor_name: str
) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
    """
    Split model at tensor_name into upstream and downstream parts.

    Args:
        model_path: Path to the ONNX model file.
        tensor_name: The tensor name at which to split.

    Returns:
        (upstream_model, downstream_model)
        - upstream: inputs -> tensor_name (+ any cross-boundary tensors)
        - downstream: tensor_name (+ cross-boundary tensors) -> outputs
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = onnx.load(model_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    graph = model.graph
    # 找到这个onnx的制定tensor的name的前面的字典
    upstream_indices = _find_upstream_node_indices(graph, tensor_name)

    if not upstream_indices:
        all_outputs = set()
        for node in graph.node:
            for out in node.output:
                all_outputs.add(out)
        raise ValueError(
            f"tensor_name '{tensor_name}' not found as any node output in the graph. "
            f"Available node outputs (first 10): {sorted(all_outputs)[:10]}"
        )

    all_nodes = list(graph.node)
    upstream_nodes = [all_nodes[i] for i in range(len(all_nodes)) if i in upstream_indices]
    downstream_nodes = [all_nodes[i] for i in range(len(all_nodes)) if i not in upstream_indices]

    upstream_outputs = set()
    for n in upstream_nodes:
        for o in n.output:
            upstream_outputs.add(o)

    downstream_inputs = set()
    for n in downstream_nodes:
        for inp in n.input:
            if inp:
                downstream_inputs.add(inp)

    init_map = {init.name: init for init in graph.initializer}
    graph_input_names = {inp.name for inp in graph.input}

    upstream_init_names = set()
    for n in upstream_nodes:
        for inp in n.input:
            if inp in init_map:
                upstream_init_names.add(inp)

    downstream_init_names = set()
    for n in downstream_nodes:
        for inp in n.input:
            if inp in init_map:
                downstream_init_names.add(inp)

    cross_boundary = (
        (upstream_outputs & downstream_inputs)
        - downstream_init_names
        - graph_input_names
        - {tensor_name}
    )

    if cross_boundary:
        print(f"Cross-boundary tensors (besides split point): {cross_boundary}")

    vi_map = {vi.name: vi for vi in graph.value_info}
    for inp in graph.input:
        vi_map.setdefault(inp.name, inp)
    for out in graph.output:
        vi_map.setdefault(out.name, out)

    if tensor_name in vi_map:
        tensor_type = deepcopy(vi_map[tensor_name])
    else:
        tensor_type = helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, None)

    # --- Build upstream model ---
    upstream_output_vis = [tensor_type]
    for cb in sorted(cross_boundary):
        if cb in vi_map:
            upstream_output_vis.append(deepcopy(vi_map[cb]))
        else:
            upstream_output_vis.append(
                helper.make_tensor_value_info(cb, TensorProto.FLOAT, None)
            )

    # Only include inputs actually consumed by upstream nodes
    upstream_used_inputs = set()
    for n in upstream_nodes:
        for inp in n.input:
            if inp and inp not in init_map:
                upstream_used_inputs.add(inp)
    upstream_inputs = [inp for inp in graph.input if inp.name in upstream_used_inputs]

    upstream_all_tensors = set()
    for n in upstream_nodes:
        for o in n.output:
            upstream_all_tensors.add(o)
        for inp in n.input:
            if inp:
                upstream_all_tensors.add(inp)
    upstream_input_names = {inp.name for inp in upstream_inputs}
    upstream_output_names = {vi.name for vi in upstream_output_vis}
    upstream_value_info = [
        deepcopy(vi_map[t]) for t in upstream_all_tensors
        if t in vi_map
        and t not in upstream_input_names
        and t not in upstream_init_names
        and t not in upstream_output_names
    ]

    upstream_graph = helper.make_graph(
        upstream_nodes,
        name="upstream",
        inputs=upstream_inputs,
        outputs=upstream_output_vis,
        initializer=[init_map[n] for n in upstream_init_names],
    )
    upstream_graph.value_info.extend(upstream_value_info)
    upstream_model = helper.make_model(upstream_graph, opset_imports=list(model.opset_import))
    upstream_model.ir_version = model.ir_version

    # --- Build downstream model ---
    down_input_vis = [deepcopy(tensor_type)]
    for cb in sorted(cross_boundary):
        if cb in vi_map:
            down_input_vis.append(deepcopy(vi_map[cb]))
        else:
            down_input_vis.append(
                helper.make_tensor_value_info(cb, TensorProto.FLOAT, None)
            )
    for inp in graph.input:
        if inp.name in downstream_inputs and inp.name not in downstream_init_names and inp.name != tensor_name:
            down_input_vis.append(inp)

    downstream_all_tensors = set()
    for n in downstream_nodes:
        for o in n.output:
            downstream_all_tensors.add(o)
        for inp in n.input:
            if inp:
                downstream_all_tensors.add(inp)
    down_input_names = {vi.name for vi in down_input_vis}
    down_output_names = {out.name for out in graph.output}
    downstream_value_info = [
        deepcopy(vi_map[t]) for t in downstream_all_tensors
        if t in vi_map
        and t not in down_input_names
        and t not in downstream_init_names
        and t not in down_output_names
    ]

    downstream_graph = helper.make_graph(
        downstream_nodes,
        name="downstream",
        inputs=down_input_vis,
        outputs=list(graph.output),
        initializer=[init_map[n] for n in downstream_init_names],
    )
    downstream_graph.value_info.extend(downstream_value_info)
    downstream_model = helper.make_model(downstream_graph, opset_imports=list(model.opset_import))
    downstream_model.ir_version = model.ir_version

    print(f"Split done: upstream={len(upstream_nodes)} nodes, downstream={len(downstream_nodes)} nodes")
    return upstream_model, downstream_model


def _rename_graph_tensors(graph: onnx.GraphProto, rename_map: Dict[str, str]):
    """Rename tensors in a graph (in-place), recursing into subgraphs."""
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]
        for i, out in enumerate(node.output):
            if out in rename_map:
                node.output[i] = rename_map[out]
        # Recurse into subgraphs (If/Loop/Scan nodes)
        for attr in node.attribute:
            if attr.g and attr.g.ByteSize() > 0:
                _rename_graph_tensors(attr.g, rename_map)

    for inp in graph.input:
        if inp.name in rename_map:
            inp.name = rename_map[inp.name]

    for out in graph.output:
        if out.name in rename_map:
            out.name = rename_map[out.name]

    for vi in graph.value_info:
        if vi.name in rename_map:
            vi.name = rename_map[vi.name]

    for init in graph.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]


def _rename_tensor_in_tail(tail: onnx.ModelProto, rename_map: Dict[str, str]) -> onnx.ModelProto:
    """Rename tensors in the tail model according to rename_map."""
    if not rename_map:
        return tail
    _rename_graph_tensors(tail.graph, rename_map)
    return tail


def merge_models(
    backbone: Union[str, onnx.ModelProto],
    tail: onnx.ModelProto,
    tensor_name: str,
    output_path: str = None,
    backbone_output_name: str = None,
) -> onnx.ModelProto:
    """
    Merge new backbone with the downstream (tail) model.

    The tail's input is renamed to match the backbone's output (backbone is untouched).

    Args:
        backbone: Path to new backbone ONNX or ModelProto.
        tail: The downstream ModelProto from split_model().
        tensor_name: The connecting tensor name (tail currently expects this as input).
        output_path: Optional save path.
        backbone_output_name: Which backbone output to use as connection. Default: first output.

    Returns:
        Merged ModelProto.
    """
    if isinstance(backbone, str):
        if not os.path.isfile(backbone):
            raise FileNotFoundError(f"Backbone file not found: {backbone}")
        backbone = onnx.load(backbone)

    bb_graph = backbone.graph
    tail_graph = tail.graph

    if len(bb_graph.output) == 0:
        raise ValueError("Backbone model has no outputs.")

    if backbone_output_name is None:
        backbone_output_name = bb_graph.output[0].name

    rename_map = {}
    if tensor_name != backbone_output_name:
        rename_map[tensor_name] = backbone_output_name

    _rename_tensor_in_tail(tail, rename_map)
    tail_graph = tail.graph

    # Handle initializer name conflicts
    bb_init_names = {init.name for init in bb_graph.initializer}
    tail_init_names = {init.name for init in tail_graph.initializer}
    conflicts = bb_init_names & tail_init_names
    if conflicts:
        print(f"Warning: initializer name conflicts detected ({len(conflicts)}), auto-renaming tail side.")
        conflict_rename = {name: f"tail__{name}" for name in conflicts}
        _rename_tensor_in_tail(tail, conflict_rename)
        tail_graph = tail.graph

    # Deduplicate node names
    bb_node_names = {n.name for n in bb_graph.node if n.name}
    for node in tail_graph.node:
        if node.name and node.name in bb_node_names:
            node.name = f"tail__{node.name}"

    merged_nodes = list(bb_graph.node) + list(tail_graph.node)
    merged_initializers = list(bb_graph.initializer) + list(tail_graph.initializer)

    # Inputs = backbone inputs + tail inputs (except connection tensor)
    merged_inputs = []
    seen_input_names = set()
    for inp in bb_graph.input:
        if inp.name not in bb_init_names and inp.name not in seen_input_names:
            merged_inputs.append(inp)
            seen_input_names.add(inp.name)
    tail_init_names_updated = {init.name for init in tail_graph.initializer}

    # Check cross-boundary satisfaction
    bb_all_outputs = set()
    for node in bb_graph.node:
        for out in node.output:
            bb_all_outputs.add(out)

    unresolved_tail_inputs = []
    for inp in tail_graph.input:
        if inp.name == backbone_output_name:
            continue
        if inp.name in tail_init_names_updated:
            continue
        if inp.name in seen_input_names:
            continue
        if inp.name in bb_all_outputs:
            # This cross-boundary tensor is produced by backbone, no need to expose as input
            continue
        merged_inputs.append(inp)
        seen_input_names.add(inp.name)
        unresolved_tail_inputs.append(inp.name)

    if unresolved_tail_inputs:
        print(f"Warning: tail has extra inputs not produced by backbone: {unresolved_tail_inputs}")
        print("  These will be exposed as merged model inputs.")

    merged_outputs = list(tail_graph.output)
    merged_value_info = list(bb_graph.value_info) + list(tail_graph.value_info)

    merged_graph = helper.make_graph(
        merged_nodes,
        name="merged",
        inputs=merged_inputs,
        outputs=merged_outputs,
        initializer=merged_initializers,
    )
    merged_graph.value_info.extend(merged_value_info)

    # Opset: take max version per domain (deepcopy to avoid mutating backbone)
    opset_imports = deepcopy(list(backbone.opset_import))
    for op in tail.opset_import:
        existing = [o for o in opset_imports if o.domain == op.domain]
        if existing:
            existing[0].version = max(existing[0].version, op.version)
        else:
            opset_imports.append(deepcopy(op))

    merged_model = helper.make_model(merged_graph, opset_imports=opset_imports)
    merged_model.ir_version = max(backbone.ir_version, tail.ir_version)

    try:
        merged_model = shape_inference.infer_shapes(merged_model)
    except Exception as e:
        print(f"Warning: shape inference failed: {e}")

    try:
        onnx.checker.check_model(merged_model)
        print("Validation passed.")
    except Exception as e:
        print(f"Warning: validation issue: {e}")

    if output_path:
        onnx.save(merged_model, output_path)
        print(f"Saved: {output_path}")

    return merged_model


# ============================================================
# Utility Functions
# ============================================================


def remove_trailing_mul(model_path: str, output_path: str = None) -> onnx.ModelProto:
    """
    Remove trailing Mul node(s) at the end of an ONNX model.
    Bypasses the Mul by connecting its non-constant input directly to the output.

    Only removes Mul nodes where exactly one input is a constant (initializer).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = onnx.load(model_path)
    graph = model.graph

    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    init_names = {init.name for init in graph.initializer}
    nodes_to_remove = []
    output_remap = {}

    for out in graph.output:
        if out.name not in producer:
            continue
        node = producer[out.name]
        if node.op_type != "Mul":
            continue

        non_const_inputs = [inp for inp in node.input if inp and inp not in init_names]
        if len(non_const_inputs) != 1:
            print(f"Warning: Mul node '{node.name}' has {len(non_const_inputs)} non-const inputs, skipping.")
            continue

        bypass_tensor = non_const_inputs[0]
        nodes_to_remove.append(node)
        output_remap[out.name] = bypass_tensor

    if not nodes_to_remove:
        print("No trailing Mul found.")
        return model

    kept_nodes = [n for n in graph.node if n not in nodes_to_remove]

    # Build a unified rename map: bypass_name -> orig_name
    # Resolve chains: if A->B and B->C, we want A->C
    rename_map = {}
    for orig_name, bypass_name in output_remap.items():
        rename_map[bypass_name] = orig_name

    # Resolve transitive chains in rename_map
    resolved = {}
    for src, dst in rename_map.items():
        # Follow the chain: dst might also be renamed
        final = dst
        seen = {src}
        while final in rename_map and final not in seen:
            seen.add(final)
            final = rename_map[final]
        resolved[src] = final
    rename_map = resolved

    # Apply rename to all kept nodes
    for node in kept_nodes:
        for i, out in enumerate(node.output):
            if out in rename_map:
                node.output[i] = rename_map[out]
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]

    # Also rename in graph.input if bypass_name is a graph input
    for inp in graph.input:
        if inp.name in rename_map:
            inp.name = rename_map[inp.name]

    # Remove initializers only used by removed Mul nodes
    removed_inputs = set()
    for node in nodes_to_remove:
        for inp in node.input:
            if inp:
                removed_inputs.add(inp)
    still_used = set()
    for node in kept_nodes:
        for inp in node.input:
            if inp:
                still_used.add(inp)
    removable_inits = removed_inputs - still_used

    kept_initializers = [init for init in graph.initializer if init.name not in removable_inits]

    # Build value_info: rename and filter
    kept_value_info = []
    renamed_values = set(rename_map.values())
    for vi in graph.value_info:
        if vi.name in rename_map:
            new_vi = deepcopy(vi)
            new_vi.name = rename_map[vi.name]
            if new_vi.name not in removable_inits:
                kept_value_info.append(new_vi)
        elif vi.name not in removable_inits and vi.name not in renamed_values:
            kept_value_info.append(vi)

    new_graph = helper.make_graph(
        kept_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=kept_initializers,
    )
    new_graph.value_info.extend(kept_value_info)

    new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version

    try:
        new_model = shape_inference.infer_shapes(new_model)
    except Exception:
        pass

    print(f"Removed {len(nodes_to_remove)} trailing Mul node(s).")

    if output_path:
        onnx.save(new_model, output_path)
        print(f"Saved: {output_path}")

    return new_model


def _fix_opset_attr_to_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Fix nodes that use old-style attributes which became inputs in newer opsets:
    - ReduceMean: axes attr -> input (opset 18)
    - Unsqueeze/Squeeze: axes attr -> input (opset 13)

    Only converts when the model's opset version requires it.
    """
    graph = model.graph

    default_opset = 1
    for op in model.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            default_opset = op.version
            break

    existing_init_names = {init.name for init in graph.initializer}

    for idx, node in enumerate(graph.node):
        if node.op_type == "ReduceMean" and default_opset < 18:
            continue
        if node.op_type in {"Unsqueeze", "Squeeze"} and default_opset < 13:
            continue
        if node.op_type not in {"ReduceMean", "Unsqueeze", "Squeeze"}:
            continue

        axes_attr = None
        for attr in node.attribute:
            if attr.name == "axes":
                axes_attr = attr
                break
        if axes_attr is None:
            continue

        axes_values = list(axes_attr.ints)
        # Deterministic naming with dedup
        base_name = f"{node.name or (node.op_type + '_' + str(idx))}_axes"
        axes_tensor_name = base_name
        counter = 0
        while axes_tensor_name in existing_init_names:
            counter += 1
            axes_tensor_name = f"{base_name}_{counter}"

        axes_tensor = helper.make_tensor(
            axes_tensor_name, TensorProto.INT64, [len(axes_values)], axes_values
        )
        graph.initializer.append(axes_tensor)
        existing_init_names.add(axes_tensor_name)

        node.attribute.remove(axes_attr)
        if len(node.input) < 2:
            node.input.append(axes_tensor_name)
        else:
            node.input[1] = axes_tensor_name

    return model


def simplify_model(model_path: str, output_path: str = None) -> onnx.ModelProto:
    """Simplify ONNX model using onnxsim (constant folding, dead code elimination, etc.)."""
    import onnxsim

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = onnx.load(model_path)

    _fix_opset_attr_to_input(model)

    valid_vis = []
    for vi in model.graph.value_info:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            valid_vis.append(vi)
    del model.graph.value_info[:]
    model.graph.value_info.extend(valid_vis)

    model_sim, check = onnxsim.simplify(model)
    if not check:
        print("Warning: onnxsim simplify check failed, using original model.")
        model_sim = model

    save_path = output_path or model_path
    onnx.save(model_sim, save_path)
    print(f"Simplified model saved: {save_path}")
    return model_sim


def verify_inference(model_path: str, input_shapes: list = None):
    """
    Verify the ONNX model can run inference with random input via onnxruntime.
    Prints input/output names, shapes, and checks for runtime errors.

    Args:
        input_shapes: list of tuples, one per model input. If fewer shapes than
                      inputs, the last shape is reused. Default: [(1,3,256,256)]
    """
    if input_shapes is None:
        input_shapes = [(1, 3, 256, 256)]
    print(f"\n--- Verifying inference: {model_path} ---")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("Inputs:")
    feed = {}
    for idx, inp in enumerate(sess.get_inputs()):
        cur_shape = input_shapes[idx] if idx < len(input_shapes) else input_shapes[-1]
        shape = inp.shape
        concrete_shape = []
        for i, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            elif i < len(cur_shape):
                concrete_shape.append(cur_shape[i])
            elif i == 0:
                concrete_shape.append(1)
            else:
                concrete_shape.append(1)
        print(f"  {inp.name}: shape={concrete_shape}, dtype={inp.type}")

        if "int" in inp.type.lower():
            feed[inp.name] = np.random.randint(0, 100, size=concrete_shape).astype(np.int64)
        else:
            feed[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)

    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    output_names = [out.name for out in sess.get_outputs()]
    results = sess.run(output_names, feed)

    print("Inference results:")
    for name, result in zip(output_names, results):
        print(f"  {name}: shape={result.shape}, min={result.min():.4f}, max={result.max():.4f}")

    print("--- Inference OK ---\n")
    return results


def export_onnx(
    model: onnx.ModelProto,
    output_path: str,
    output_names: list = None,
    simplify: bool = True,
) -> str:
    """导出 ModelProto 为 ONNX 文件，可重命名输出节点，可选 onnxsim 简化。返回最终文件路径。"""
    if output_names is not None:
        graph = model.graph
        assert len(output_names) == len(graph.output), (
            f"output_names 长度({len(output_names)})与模型输出数({len(graph.output)})不匹配"
        )
        for out, new_name in zip(graph.output, output_names):
            old_name = out.name
            if old_name == new_name:
                continue
            out.name = new_name
            for node in graph.node:
                for i, o in enumerate(node.output):
                    if o == old_name:
                        node.output[i] = new_name
                for i, inp_name in enumerate(node.input):
                    if inp_name == old_name:
                        node.input[i] = new_name
            for vi in graph.value_info:
                if vi.name == old_name:
                    vi.name = new_name

    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, output_path)
    print(f"Saved ONNX model to: {output_path}")

    if simplify:
        sim_path = output_path.replace(".onnx", "_sim.onnx")
        simplify_model(output_path, sim_path)
        return sim_path
    return output_path


# ============================================================
# CLI — 一步完成: split → merge → simplify → verify
# ============================================================


def replace_backbone(
    original: str,
    backbone: str,
    tensor_name: str,
    output: str,
    backbone_output: str = None,
    skip_simplify: bool = False,
    input_shapes: list = None,
) -> str:
    """
    一步完成 backbone 替换全流程。

    Args:
        original: 原始完整模型路径
        backbone: 新 backbone 模型路径
        tensor_name: 切分点 tensor 名称
        output: 输出模型路径
        backbone_output: backbone 输出 tensor 名称（默认使用第一个输出）
        skip_simplify: 是否跳过 onnxsim 简化
        input_shapes: 验证推理用的各输入 shape 列表

    Returns:
        最终输出模型路径
    """
    # 
    print(f"[1/4] Splitting original model at: {tensor_name}")
    _, tail = split_model(original, tensor_name)

    print(f"[2/4] Merging backbone with tail...")
    merge_models(backbone, tail, tensor_name, output, backbone_output)

    if not skip_simplify:
        print(f"[3/4] Simplifying merged model...")
        sim_path = output.replace(".onnx", "_sim.onnx")
        simplify_model(output, sim_path)
        final_path = sim_path
    else:
        print(f"[3/4] Skipping simplification.")
        final_path = output

    print(f"[4/4] Verifying inference...")
    verify_inference(final_path, input_shapes)

    print(f"\nDone! Final model: {final_path}")
    return final_path


def main():
    # 
    parser = argparse.ArgumentParser(
        description="ONNX backbone replacement tool — 一步完成 split → merge → simplify → verify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python replace_onnx.py \\
      --original onnx_files/MergedSceneCls4Head.onnx \\
      --backbone aimet_output_20260512/asd_clip_20260512_no_mul.onnx \\
      --tensor_name "/model/_model/model/conv_exp/activation/Mul_output_0" \\
      -o replaced_model.onnx

  python split_merge_simplify_onnx.py --original merged_scene_cls_4heads.onnx --backbone asd_clip_20260512_no_mul.onnx --tensor_name "mul_50" -o replaced_model.onnx    
""",
    )
    parser.add_argument("--original", help="原始完整 ONNX 模型路径", default="./models/merged_scene_cls_4heads.onnx")
    parser.add_argument("--backbone", help="新 backbone ONNX 模型路径", default="./models/asd_scene_cls_clip_backbone.onnx")
    parser.add_argument("--tensor_name", help="切分点 tensor 名称（backbone 与 tail 的连接处）", default="mul_50")
    parser.add_argument("--backbone_output", default=None, help="backbone 输出 tensor 名称（默认使用第一个输出）")
    parser.add_argument("-o", "--output", help="输出模型路径", default="./replaced_model.onnx")
    parser.add_argument("--skip_simplify", action="store_true", help="跳过 onnxsim 简化")
    parser.add_argument("--input_shape", default="1,3,256,256",
                        help="验证推理用的输入 shape（逗号分隔，多输入用分号分隔，如 '1,3,256,256;1,1,6,512'）")

    args = parser.parse_args()
    input_shapes = [tuple(int(x) for x in s.split(",")) for s in args.input_shape.split(";")]
    # 进行backbone的替换
    replace_backbone(
        original=args.original,
        backbone=args.backbone,
        tensor_name=args.tensor_name,
        output=args.output,
        backbone_output=args.backbone_output,
        skip_simplify=args.skip_simplify,
        input_shapes=input_shapes,
    )


if __name__ == "__main__":
    main()
