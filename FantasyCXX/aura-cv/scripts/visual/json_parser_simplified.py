import json
import argparse
import glob
import os
import csv
import sys
import pandas as pd
import openpyxl
from openpyxl.styles import Font
from openpyxl.styles import Border, Side

if __name__ == '__main__':
    input_sizes = ["2048x4096x1(hwc) U8", "2048x4096x1(hwc) U16", "2048x4096x1(hwc) F32", "2048x4096x1(hwc) F16",
                   "1024x2048x3(hwc) U8", "1024x2048x3(hwc) U16", "1024x2048x3(hwc) F32", "1024x2048x3(hwc) F16",
                   "1024x2048x1(hwc) U8", "1024x2048x1(hwc) U16", "1024x2048x1(hwc) F32", "1024x2048x1(hwc) F16",
                   "2048x2048x1(hwc) U8", "2048x2048x1(hwc) U16", "2048x2048x1(hwc) F32", "2048x2048x1(hwc) F16",
                   "1024x1024x1(hwc) F32x1024x1024x1(hwc) F32", "512x1024x1(hwc) F32x1024x512x1(hwc) F32"]

    output_sizes =["2048x2048x1(hwc) U8", "2048x2048x1(hwc) U16", "2048x2048x1(hwc) F32", "2048x2048x1(hwc) F16", \
                   "1024x1024x1(hwc) U8", "1024x1024x1(hwc) U16", "1024x1024x1(hwc) F32", "1024x1024x1(hwc) F16"]

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

    json_param_data = {}
    json_input_sizes = {}
    json_output_sizes = {}
    json_module_names = {}
    json_operator_names = {}
    for device, json_path in zip(devices, json_paths):
        with open(json_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)

            # select all operators from the same module, eg. matrix
            for test_case_name in json_data['default']:
                module = test_case_name.split('_')[0]

                if module in module_black_lists:
                    continue

                operator = '_'.join(test_case_name.split('_')[1:-1])

                if module not in json_module_names.keys():
                    json_module_names.update({module:[operator]})
                    json_operator_names.update({operator: [module+'_'+operator+'_none']})
                elif operator not in json_module_names[module]:
                    json_module_names[module].append(operator)
                    json_operator_names.update({operator: [module+'_'+operator+'_none']})

                if operator not in json_operator_names.keys():
                    json_operator_names.update({operator:[test_case_name]})
                elif test_case_name not in json_operator_names[operator]:
                    json_operator_names[operator].append(test_case_name)

    for device, json_path in zip(devices, json_paths):
        with open(json_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            for module in json_module_names.keys():
                for operator in json_module_names[module]:
                    for test_case_name in json_operator_names[operator]:

                        if test_case_name not in json_data['default'].keys():
                            break

                        impl = test_case_name.split('_')[-1]
                        if impl == "none":
                            for case_data in json_data['default'][test_case_name]['result']:
                                flag = 0

                                if case_data['input'] in input_sizes or case_data['output'] in output_sizes:
                                    flag = 1

                                ## module: matrix
                                if 'matrix' in test_case_name:
                                    if "2048" not in case_data['output'] and "2048" not in case_data['input']:
                                        flag = 0
                                    if "Gemm" in test_case_name:
                                        flag = 1
                                    elif 'Dft' in test_case_name:
                                        sizes = ['1024x1024x2(hwc) F32', '2048x2048x2(hwc) F32']
                                        if case_data['input'] in sizes:
                                            flag = 1

                                    elif 'MulSpectrums' in test_case_name:
                                        sizes = ['1021x1031x2(hwc) F32', '2048x2048x2(hwc) F32']
                                        if case_data['input'] in sizes:
                                            flag = 1

                                    elif 'Split' in test_case_name:
                                        sizes = ["2048x4096x2(hwc) U8", "2048x4096x2(hwc) U16",
                                                 "2048x4096x2(hwc) U32", "2048x4096x2(hwc) F32", "2048x4096x2(hwc) F16"]
                                        if case_data['input'] in sizes:
                                            flag = 1

                                    elif 'Arithmetic' in test_case_name and case_data['input'] != case_data['output']:
                                        flag = 0

                                    elif 'ConvertTo' in test_case_name:
                                        if "F32" not in case_data['output'] or "alpha:3.141000" not in case_data['param']:
                                            flag = 0

                                    elif "GridIDftReal" in test_case_name:
                                        if "2048" not in case_data['input'] or "grid_len: 16" not in case_data['param']:
                                            flag = 0
                                    elif "IDft" in test_case_name and "1021x1031x2" in case_data['input']:
                                        flag = 1

                                    elif "MakeBorder" in test_case_name:
                                        if "F32" not in case_data['input']:
                                            flag = 0

                                    elif "Rotate" in test_case_name:
                                        if "_90" not in case_data['param']:
                                            flag = 0

                                    elif "Integral" in test_case_name:
                                        if "S32" not in case_data['output']:
                                            flag = 0
                                    if "Constant" in case_data['param'] or "Reflect_101" in case_data['param']:
                                        flag = 0
                                ## module: misc
                                elif "misc" in test_case_name:
                                    flag = 1
                                    if "misc_Threshold" in test_case_name:
                                        if "_INV" in case_data['param'] or ("U8" not in case_data['input'] and "F32" not in case_data['input']):
                                            flag = 0
                                ## module: cvtcolor
                                elif "cvtcolor" in test_case_name and "2048" in case_data['input']:
                                    flag = 1

                                ## module: feature2d
                                elif "feature2d" in test_case_name:
                                    flag = 1
                                    if "feature2d_Canny" in test_case_name:
                                        if "low_thresh:60.000000" in case_data['param'] or "high_thresh:150.000000" in case_data['param'] or \
                                                "l2_gradient:1" in case_data['param']:
                                            flag = 0

                                ## module: filter
                                elif "filter" in test_case_name:
                                    flag = 1
                                    if "2048" not in case_data['output'] and "2048" not in case_data['input']:
                                        flag = 0
                                    if "filter_Sobel" in test_case_name:
                                        flag = 1
                                        if "S16" in case_data['output'] or "ksize:1 | dx:2" in case_data['param']:
                                            flag = 0
                                        if "479" not in case_data['input']:
                                            flag = 0
                                    if "filter_Gaussian" in test_case_name:
                                        if "sigma:0" in case_data['param']:
                                            flag = 0
                                    if "filter_Median" in test_case_name:
                                        if "1024" not in case_data['input']:
                                            flag = 0
                                    if "filter_Laplacian" in test_case_name:
                                        flag = 1
                                        if "479" not in case_data['input']:
                                            flag = 0

                                ## module: morph
                                elif "morph_MorphologyEx" in test_case_name:
                                    flag = 1
                                    if "DILATE" in case_data['param'] and ("U8" not in case_data['input'] or "x3" in case_data['input']):
                                        flag = 0
                                    elif "U8" not in case_data['input'] and "F32" not in case_data['input']:
                                        flag = 0
                                    elif "479" not in case_data['input']:
                                        flag = 0
                                    elif "ERODE" in case_data['param']:
                                        flag = 0

                                ## module: warp
                                elif "warp" in test_case_name and "479" in case_data['input']:
                                    flag = 1
                                    if "U8" not in case_data['input'] and "F32" not in case_data['input']:
                                        flag = 0
                                
                                ## module: pyramid
                                elif "pyramid" in test_case_name:
                                    flag = 1
                                
                                ## module: resize
                                elif "resize" in test_case_name:
                                    if ("512" in case_data['input'] or "1024" in case_data['input']) and \
                                            ("2" in case_data['param'] or "4" in case_data['param']):
                                        flag = 1
                                    if "U8" not in case_data['input'] and "F32" not in case_data['input']:
                                        flag = 0
                                    elif "U8" not in case_data['input'] and ("0.25" in case_data['param'] or "4" in case_data['param']):
                                        flag = 0

                                elif "hist_Equalizehist_none" in test_case_name:
                                    flag = 1


                                if flag == 1:
                                    operator_key = test_case_name.replace("_" + impl, "")
                                    if operator_key not in json_input_sizes.keys():
                                        json_input_sizes.update({operator_key:[case_data['input']]})

                                    if operator_key not in json_output_sizes.keys():
                                        json_output_sizes.update({operator_key:[case_data['output']]})
                                    
                                    if operator_key not in json_param_data.keys():
                                        json_param_data.update({operator_key:[case_data['param']]})

                                    new_flag = 1
                                    for __index in range(len(json_input_sizes[operator_key])):
                                        if case_data['input'] == json_input_sizes[operator_key][__index] and \
                                                case_data['output'] == json_output_sizes[operator_key][__index] and \
                                                case_data['param'] == json_param_data[operator_key][__index]:
                                            new_flag = 0
                                            break
                                    if new_flag == 1:
                                        json_input_sizes[operator_key].append(case_data['input'])
                                        json_output_sizes[operator_key].append(case_data['output'])
                                        json_param_data[operator_key].append(case_data['param'])

    json_module_lengths = []
    json_operator_lengths = []
    
    for module in json_module_names.keys():
        module_length = 0
        drop_list=[]
        for operator in json_module_names[module]:
            operator_key = module + "_" + operator
            if operator_key in json_input_sizes.keys():
                json_operator_lengths.append(len(json_input_sizes[operator_key]))
                module_length += len(json_input_sizes[operator_key])
            else:
                print(operator_key + " not writing to excel.")
                # for __index in range(len(json_operator_names[operator])):
                json_operator_names.pop(operator)
                drop_list.append(operator)
        for operator in drop_list:
            json_module_names[module].remove(operator)
        json_module_lengths.append(module_length)

    size_index = 0
    for operator_key in json_input_sizes.keys():
        for __index in range(len(json_input_sizes[operator_key])):
            json_input_sizes[operator_key][__index] = json_input_sizes[operator_key][__index] + "##" + str(size_index)
            json_output_sizes[operator_key][__index] = json_output_sizes[operator_key][__index] + "##" + str(size_index)
            json_param_data[operator_key][__index] = json_param_data[operator_key][__index] + "##" + str(size_index)
            size_index += 1

    json_time_CV = [[] for i in range(len(devices))]
    json_time_none = [[] for i in range(len(devices))]
    json_time_neon = [[] for i in range(len(devices))]
    json_time_opencl = [[] for i in range(len(devices))]
    json_time_hvx = [[] for i in range(len(devices))]
    
    json_index = -1
    for device, json_path in zip(devices, json_paths):
        json_index += 1
        with open(json_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            json_operator_index = -1

            for module in json_module_names.keys():
                for operator in json_module_names[module]:
                    json_operator_index += 1
                    json_time_none[json_index].extend([""] * json_operator_lengths[json_operator_index])
                    json_time_neon[json_index].extend([""] * json_operator_lengths[json_operator_index])
                    json_time_CV[json_index].extend([""] * json_operator_lengths[json_operator_index])
                    json_time_opencl[json_index].extend([""] * json_operator_lengths[json_operator_index])
                    json_time_hvx[json_index].extend([""] * json_operator_lengths[json_operator_index])

                    for test_case_name in json_operator_names[operator]:
                        if test_case_name not in json_data['default'].keys():
                            continue

                        impl = test_case_name.split('_')[-1]
                        operator_key = "_".join(test_case_name.split('_')[:-1])
                        for case_data in json_data['default'][test_case_name]['result']:
                            for __index in range(json_operator_lengths[json_operator_index]):
                                input_size = json_input_sizes[operator_key][__index].split("##")[0]
                                output_size = json_output_sizes[operator_key][__index].split("##")[0]
                                param_data = json_param_data[operator_key][__index].split("##")[0]
                                if input_size == case_data['input'] and output_size == case_data['output'] and param_data == case_data['param']:
                                    time_index = int(json_input_sizes[operator_key][__index].split("##")[1])
                                    if impl == "none":
                                        json_time_none[json_index][time_index] = round(case_data['perf_result']['None']['avg'], 3)
                                        if 'OpenCV' in case_data['perf_result']:
                                            json_time_CV[json_index][time_index] = round(case_data['perf_result']['OpenCV']['avg'], 3)
                                    elif impl == "neon":
                                        json_time_neon[json_index][time_index] = round(case_data['perf_result']['Neon']['avg'], 3)
                                    elif impl == "opencl":
                                        json_time_opencl[json_index][time_index] = round(case_data['perf_result']['Opencl']['avg'], 3)
                                    elif impl == "hvx":
                                        json_time_hvx[json_index][time_index] = round(case_data['perf_result']['Hvx']['avg'], 3)

    begin_letters = []
    end_letters = []
    with pd.ExcelWriter(args.output_path + '/aura2.xlsx') as writer:
        json_dict = {}
        json_dict['input_sizes (hwc)'] = []
        json_dict['output_sizes (hwc)'] = []
        json_dict['param_data'] = []
        size_index = 0
        for operator_key in json_input_sizes.keys():
            for __index in range(len(json_input_sizes[operator_key])):
                json_dict['input_sizes (hwc)'].append(json_input_sizes[operator_key][__index].split("##")[0])
                json_dict['output_sizes (hwc)'].append(json_output_sizes[operator_key][__index].split("##")[0])
                json_dict['param_data'].append(json_param_data[operator_key][__index].split("##")[0])
                json_dict['input_sizes (hwc)'][size_index]  = json_dict['input_sizes (hwc)'][size_index].replace("(hwc)", " ")
                json_dict['output_sizes (hwc)'][size_index] = json_dict['output_sizes (hwc)'][size_index].replace("(hwc)", " ")
                if json_dict['input_sizes (hwc)'][size_index][-1] == " ":
                    json_dict['input_sizes (hwc)'][size_index] = json_dict['input_sizes (hwc)'][size_index][:-1]
                if json_dict['output_sizes (hwc)'][size_index][-1] == " ":
                    json_dict['output_sizes (hwc)'][size_index] = json_dict['output_sizes (hwc)'][size_index][:-1]
                size_index += 1

        df = pd.DataFrame(json_dict)
        df.style.set_properties(**{'border': '1px solid black'})
        df.to_excel(writer, sheet_name="AURA", startrow=1, startcol=2, index=False)
        worksheet = writer.sheets["AURA"]
        cell_format = writer.book.add_format({"align": "center", 'valign': 'vcenter'})
        worksheet.autofilter(1, 2, len(df.index), len(df.columns)) # set dropout list
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column) + 2
            worksheet.set_column(col_idx, col_idx, column_length, cell_format)

        module_str_max_length = len("module")
        operator_str_max_length = len("operator")
        start_index = 3
        module_index = 0
        for module_name in json_module_names.keys():
            module_str_max_length = max(module_str_max_length, len(module_name))
            end_index = start_index + json_module_lengths[module_index] - 1
            str_index = "A" + str(start_index) + ":A" + str(end_index)
            if end_index == start_index:
                worksheet.write("A" + str(start_index), module_name, cell_format)
            else:
                worksheet.merge_range(str_index, module_name, cell_format)
            module_index += 1
            start_index = end_index + 1

        operator_start_index = 3
        operator_index = 0

        for operator_name in json_operator_names.keys():
            operator_str_max_length = max(operator_str_max_length, len(operator_name))
            end_index = operator_start_index + json_operator_lengths[operator_index] - 1
            str_index = "B" + str(operator_start_index) + ":B" + str(end_index)
            if end_index == operator_start_index:
                worksheet.write("B" + str(operator_start_index), operator_name, cell_format)
            else:
                worksheet.merge_range(str_index, operator_name, cell_format)
            operator_index += 1
            operator_start_index = end_index + 1

        worksheet.write("A2", "module", cell_format)
        worksheet.write("B2", "operator", cell_format)
        
        worksheet.set_column(0, 0, module_str_max_length + 2, cell_format)
        worksheet.set_column(1, 1, operator_str_max_length + 2, cell_format)

        json_dict = {}
        begin_letter = "F"
        for __index in range(len(devices)):
            json_dict['time_CV'] = json_time_CV[__index]
            json_dict['time_C'] = json_time_none[__index]
            json_dict['time_neon'] = json_time_neon[__index]
            json_dict['time_opencl'] = json_time_opencl[__index]
            json_dict['time_hvx'] = json_time_hvx[__index]
            df = pd.DataFrame(json_dict)
            df.to_excel(writer, sheet_name="AURA", startrow=1, startcol=5 + __index * 5, index=False)
            end_letter = chr(ord(begin_letter) + 4)
            string = begin_letter + "1:" + end_letter + "1"
            worksheet.merge_range(string, devices[__index] + " (time: ms)", cell_format)
            begin_letters.append(begin_letter)
            end_letters.append(end_letter)
            begin_letter = chr(ord(end_letter) + 1)
        
            for column in df:
                column_length = max(df[column].astype(str).map(len).max(), len(column)) + 2
                col_idx = df.columns.get_loc(column) + 5 + __index * 5
                worksheet.set_column(col_idx, col_idx, column_length, cell_format)

    sheet_length = sum(json_module_lengths) + 3
    wk = openpyxl.load_workbook(args.output_path + '/aura2.xlsx')
    thin = Side(border_style='medium') ## medium border
    sheet = wk['AURA']

    border = Border(left=thin)  # border setting
    for row in sheet['A2:A' + str(sheet_length)]:
        for cell in row:
            cell.border = border

    border = Border(right=thin)
    for row in sheet['E2:E' + str(sheet_length)]:
        for cell in row:
            cell.border = border

    border = Border(top=thin)
    for row in sheet['A2:E2']:
        for cell in row:
            cell.border = border

    for row in sheet['A' + str(sheet_length)+':E' + str(sheet_length)]:
        for cell in row:
            cell.border = border

    for begin_letter, end_letter in zip(begin_letters, end_letters):
        border_str = begin_letter + "1:" + begin_letter + str(sheet_length)
        border = Border(left=thin)
        for row in sheet[border_str]:
            for cell in row:
                cell.border = border

        border_str = end_letter + "1:" + end_letter + str(sheet_length)
        border = Border(right=thin)
        for row in sheet[border_str]:
            for cell in row:
                cell.border = border

        border_str = begin_letter + "1:" + end_letter + "1"
        border = Border(top=thin)
        for cell in sheet[border_str]:
            for cell in row:
                cell.border = border

        border_str = begin_letter + str(sheet_length) + ":"+ end_letter + str(sheet_length)
        for row in sheet[border_str]:
            for cell in row:
                cell.border = border

        font = Font(bold=True, italic=False, strike=False, color='d92121') # Red for time_CV
        border_str = begin_letter + "2:" + begin_letter + str(sheet_length)
        for row in sheet[border_str]:
            for cell in row:
                cell.font = font
    
        font = Font(bold=True, italic=False, strike=False, color='009000') # Green for time_none
        letter = chr(ord(begin_letter) + 1)
        border_str = letter + "2:" + letter + str(sheet_length)
        for row in sheet[border_str]:
            for cell in row:
                cell.font = font

        font = Font(bold=True, italic=False, strike=False, color='0000ff') # Blue for time_neon
        letter = chr(ord(letter) + 1)
        border_str = letter + "2:" + letter + str(sheet_length)
        for row in sheet[border_str]:
            for cell in row:
                cell.font = font

        font = Font(bold=True, italic=False, strike=False, color='4b0082') # Pigment Indigo for time_opencl
        letter = chr(ord(letter) + 1)
        border_str = letter + "2:" + letter + str(sheet_length)
        for row in sheet[border_str]:
            for cell in row:
                cell.font = font

        font = Font(bold=True, italic=False, strike=False, color='ffa700') # Chrome Yellow for time_hvx
        letter = chr(ord(letter) + 1)
        border_str = letter + "2:" + letter + str(sheet_length)
        for row in sheet[border_str]:
            for cell in row:
                cell.font = font

    border = Border(left=thin, right=thin, top=thin, bottom=thin) 
    for row in sheet['A2:' + end_letters[-1] + "2"]:
        for cell in row:
            cell.border = border

    # freeze "A1:B1" columns (module and operator columns)
    sheet.freeze_panes = "B1"
    sheet.freeze_panes = "C1"
    wk.save(args.output_path + '/aura2.xlsx')
