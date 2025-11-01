import json
import argparse
import glob
import os
import csv
import pandas as pd

if __name__ == '__main__':
    module_black_lists = ['runtime', 'unit']

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, required=True, help="eg: QCOM-8550,QCOM-8650, split by comma and with no space")
    parser.add_argument("-p", "--path", type=str, required=True, help="json file paths, split by comma and with no space, the number should be the same with devices")
    parser.add_argument("-o", "--output_path", type=str, default= "", help="path to save Excel files")
    args = parser.parse_args()
    '''
    eg:
    devices = ['QCOM-8550', 'QCOM-8650']
    json_paths = ["../../../visual_report/auto_test_all_8550.json", "../../../visual_report/auto_test_all_8650.json"]
    '''
    devices = str(args.device).split(",")
    json_paths = str(args.path).split(",")

    if len(devices) != len(json_paths):
        print("numbers of devices and json paths are not equal.")
        exit(1)

    for device, json_path in zip(devices, json_paths):
        device_path = os.path.join(args.output_path, device)
        if not os.path.exists(device_path):
            os.makedirs(device_path)

        with open(json_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)

            module_names = {}
            # select all operators from the same module, eg. matrix
            for test_case_name in json_data['default']:
                module = test_case_name.split('_')[0]

                if module in module_black_lists:
                    continue

                if module not in module_names.keys():
                    module_names.update({module:[test_case_name]})
                    module_names[module].append(test_case_name)
                else:
                    module_names[module].append(test_case_name)

            for module in module_names.keys():
                module_list = {}
                for test_case_name in module_names[module]:
                    module_list[test_case_name] = json_data['default'][test_case_name]

                with pd.ExcelWriter(device_path + "/" + module+'.xlsx') as writer:
                    for test_case_name in module_list.keys():
                        case_datas = module_list[test_case_name]['result']
                        # Adjust the order of case_datas_ori, put accuracy_benchmark and the like at the back, and size information at the front
                        for case_data in case_datas:
                            case_data_ori = case_data.copy()
                            case_data.pop('param')
                            input_str = case_data['input'].replace("(hwc)", " ")
                            output_str = case_data['output'].replace("(hwc)", " ")
                            if input_str[-1] == " ":
                                input_str = input_str[:-1]
                            if output_str[-1] == " ":
                                output_str = output_str[:-1]
                            case_data['input_size (hwc)'] = input_str
                            case_data['output_size (hwc)'] = output_str
                            case_data['param'] = case_data_ori['param']
                            if 'perf_result' in case_data_ori.keys():
                                for top_keys in case_data_ori['perf_result']:
                                    for bottom_keys in case_data_ori['perf_result'][top_keys]:
                                        case_data[top_keys+'(' + bottom_keys + '/ms)'] = round(case_data_ori['perf_result'][top_keys][bottom_keys], 3)
                                case_data.pop('perf_result')
    
                            case_data.pop('perf_status')
                            case_data.pop('accu_benchmark')
                            case_data.pop('accu_result')
                            case_data.pop('accu_status')
                            case_data.pop('input')
                            case_data.pop('output')
                            case_data['perf_status'] = case_data_ori['perf_status']
                            case_data['accu_benchmark'] = case_data_ori['accu_benchmark']
                            case_data['accu_result'] = case_data_ori['accu_result']
                            case_data['accu_status'] = case_data_ori['accu_status']

                        df = pd.DataFrame(module_list[test_case_name]['result'])
                        test_case_name_list = test_case_name.split('_')
                        test_case_name = ''.join(test_case_name_list[1:-1])
                        test_case_name = test_case_name + '_' + test_case_name_list[-1]
                        df.to_excel(writer, sheet_name=test_case_name)
                        worksheet = writer.sheets[test_case_name]
                        worksheet.autofilter(0, 0, len(df.index) - 1, len(df.columns) - 1) # set dropout list
                        cell_format = writer.book.add_format({"align": "center", 'valign': 'vcenter'})

                        for column in df:
                            column_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
                            col_idx = df.columns.get_loc(column) + 1
                            worksheet.set_column(col_idx, col_idx, column_length, cell_format)

                        # freeze A1:C1 columns (input_sizes, output_sizes)
                        worksheet.freeze_panes(0, 2)
                        worksheet.freeze_panes(0, 3)