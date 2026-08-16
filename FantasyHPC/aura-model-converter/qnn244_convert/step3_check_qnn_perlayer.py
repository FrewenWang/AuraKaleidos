#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import yaml

GREEN = "\033[92m"
END_COLOR = "\033[0m"


def read_raw_file(file_path):
    with open(file_path, "rb") as file:
        data = np.fromfile(file, dtype=np.float32)
    return data


def calculate_cosine_similarity(unquantized_data, dequantized_data):
    unquantized_norm = np.linalg.norm(unquantized_data, ord=2)
    dequantized_norm = np.linalg.norm(dequantized_data, ord=2)
    dp = np.vdot(unquantized_data, dequantized_data)
    sim = dp / (unquantized_norm * dequantized_norm)
    return sim


def calculate_average_difference(unquantized_data, dequantized_data):
    absolute_difference = np.absolute(
        np.subtract(unquantized_data, dequantized_data)
    )
    return np.mean(absolute_difference)


def calculate_max_difference(unquantized_data, dequantized_data):
    absolute_difference = np.absolute(
        np.subtract(unquantized_data, dequantized_data)
    )
    return max(absolute_difference)


def calculate_max_relative_difference(
    unquantized_data, dequantized_data, epsilon=1e-6
):
    absolute_difference = np.absolute(
        np.subtract(unquantized_data, dequantized_data)
    )
    return max(absolute_difference / (epsilon + np.absolute(unquantized_data)))


def calculate_mse(unquantized_data, dequantized_data):
    difference = np.subtract(dequantized_data, unquantized_data)
    squared_difference = np.square(difference)
    return np.mean(squared_difference)


def calculate_psnr(unquantized_data, dequantized_data, max_val):
    difference = np.subtract(dequantized_data, unquantized_data)
    squared_difference = np.square(difference)
    mean_square_error = np.mean(squared_difference)

    if mean_square_error == 0:
        return 100.0
    ratio = max_val * max_val / mean_square_error
    return 10 * np.log10(ratio)


def calculate_sqnr(unquantized_data, dequantized_data):
    difference = np.subtract(dequantized_data, unquantized_data)
    squared_difference = np.square(difference)
    mean_square_error = np.mean(squared_difference)

    if mean_square_error == 0:
        return 100.0
    squared_unquantized = np.square(unquantized_data)
    mean_square_unquantized = np.mean(squared_unquantized)
    ratio = mean_square_unquantized / mean_square_error
    return 10 * np.log10(ratio)


def calculate_extrema(unquantized_data):
    return max(unquantized_data), min(unquantized_data)


DATA_TYPES = {
    "QNN_DATATYPE_UFIXED_POINT_16": "U16",
    "QNN_DATATYPE_UFIXED_POINT_8": "U8",
    "QNN_DATATYPE_SFIXED_POINT_16": "S16",
    "QNN_DATATYPE_SFIXED_POINT_8": "S8",
    "QNN_DATATYPE_UFIXED_POINT_32": "U32",
    "QNN_DATATYPE_SFIXED_POINT_32": "S32",
    "QNN_DATATYPE_FLOAT_16": "F16",
    "QNN_DATATYPE_BOOL_8": "B8",
    "QNN_DATATYPE_FLOAT_32": "F32",
}


def compare_and_save(gold_dir, bench_dir, num=1):

    with open(
        os.path.join(gold_dir, "execution_metadata.yaml"), encoding="utf-8"
    ) as file:
        content = file.read()
        data = yaml.load(content, Loader=yaml.FullLoader)
        graphs = data["graphs"][0]
        output_tensors = graphs["output_tensors"]
    tensor_map = {}
    for i, tensor in enumerate(output_tensors):
        tensor_map[tensor["tensor_name"]] = i

    with open(
        os.path.join(bench_dir, "execution_metadata.yaml"), encoding="utf-8"
    ) as file:
        content = file.read()
        data = yaml.load(content, Loader=yaml.FullLoader)
        graphs = data["graphs"][0]
        output_tensors = graphs["output_tensors"]
    tensor_info = []
    for tensor in output_tensors:
        tensor_info.append(tensor)

    first_sample = True

    sqnr_list = []
    cos_sim_list = []

    for i in range(num):
        gold_dir_cur = os.path.join(gold_dir, f"Result_{i}")
        bench_dir_cur = os.path.join(bench_dir, f"Result_{i}")
        results = []
        for tensor in tensor_info:
            tensor_name = tensor["tensor_name"]
            file = tensor_name + ".raw"
            gold_path = os.path.join(gold_dir_cur, file)
            bench_path = os.path.join(bench_dir_cur, file)
            if os.path.exists(bench_path):
                if not os.path.exists(gold_path):
                    new_tensor_name = tensor_name.split("_converted_", 1)[0]
                    gold_path = os.path.join(
                        gold_dir_cur, new_tensor_name + ".raw"
                    )
                    if os.path.exists(gold_path):
                        if first_sample:
                            print(tensor_name, " found in bench data")
                    else:
                        print(tensor_name, " not found in gold data")
                        continue
                else:
                    new_tensor_name = tensor_name

                unquantized_data = read_raw_file(gold_path)
                dequantized_data = read_raw_file(bench_path)
                if len(unquantized_data) != len(dequantized_data):
                    print(tensor_name)
                    continue

                # if pd.isna(unquantized_data).any or pd.isinf(unquantized_data).any:
                #     print("gold nan or inf: ", tensor_name)
                #     print(unquantized_data)
                #     print(dequantized_data)
                # if np.isnan(dequantized_data).any or np.isinf(dequantized_data).any:
                #     print("quantized nan or inf: ", tensor_name)
                #     print(dequantized_data)

                mse = calculate_mse(unquantized_data, dequantized_data)
                # psnr     = calculate_psnr(unquantized_data, dequantized_data, max_val=255)
                sqnr = calculate_sqnr(unquantized_data, dequantized_data)
                cos_sim = calculate_cosine_similarity(
                    unquantized_data, dequantized_data
                )
                max_diff = calculate_max_difference(
                    unquantized_data, dequantized_data
                )
                avg_diff = calculate_average_difference(
                    unquantized_data, dequantized_data
                )
                maximum, minimum = calculate_extrema(unquantized_data)

                byte = int(tensor["datatype"].split("_").pop()) / 8
                num_bytes = np.prod(tensor["dimensions"]) * byte
                if num_bytes < 1024:
                    size = f"{num_bytes:.2f} B"
                elif num_bytes < 1048576:
                    size = f"{num_bytes / 1024:.2f} KB"
                else:
                    size = f"{num_bytes / 1048576:.2f} MB"

                results.append(
                    {
                        "Name": f"{tensor_name}",
                        "idx": tensor_map[new_tensor_name],
                        "Shape": tensor["dimensions"],
                        "Dtype": DATA_TYPES[tensor["datatype"]],
                        "Size": size,
                        "MSE": mse,
                        "SQNR": sqnr,
                        "Cos-Sim": cos_sim,
                        "MaxDiff": max_diff,
                        "AvgDiff": avg_diff,
                        "Max": maximum,
                        "Min": minimum,
                    }
                )

        sqnr_list.append(sqnr)
        cos_sim_list.append(cos_sim)
        output_tensor_name = tensor_info[-1]["tensor_name"]
        print(f"name:{output_tensor_name} sqnr:{sqnr} cossim:{cos_sim}")

        df = pd.DataFrame(results)
        save_file = os.path.join(
            bench_dir, f"perlayer_activation_analysis_Result_{i}.html"
        )
        df.to_html(save_file, index=False, float_format="%.4f")

        first_sample = False

    print(
        GREEN
        + f"SQNR:mean:{sum(sqnr_list) / len(sqnr_list)} max:{max(sqnr_list)} min:{min(sqnr_list)}"
        + END_COLOR
    )
    print(
        GREEN
        + f"Cossim:mean:{sum(cos_sim_list) / len(cos_sim_list)} max:{max(cos_sim_list)} min:{min(cos_sim_list)}"
        + END_COLOR
    )


def count_result(bench_dir):
    count = 0
    for _dirpath, dirnames, _filenames in os.walk(bench_dir):
        for dirname in dirnames:
            if "Result_" in dirname:
                count += 1
    return count


def main():

    root_dir = os.environ.get("ENV_PROJECT_ROOT_DIR") + "/infer_output"
    model_name = os.environ.get("ENV_ONNX_MODEL_NAME")
    gold_dir = f"{root_dir}/{model_name}_fp32_qnn"
    env_model_dir = os.environ.get("ENV_MODEL_DIR")
    bench_dir = f"{root_dir}/{env_model_dir}/output"

    test_num = min(count_result(bench_dir), count_result(gold_dir))
    compare_and_save(gold_dir, bench_dir, test_num)


if __name__ == "__main__":
    main()
