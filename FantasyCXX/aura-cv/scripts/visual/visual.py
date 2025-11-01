#encoding:utf-8
import json
import argparse
import glob
import os
import sys
from collections import OrderedDict
from pyecharts import options as opts
from pyecharts.charts import Bar, Tree, Line, Page, Pie
from pyecharts.globals import ThemeType
from pyecharts import __version__ as pyecharts_version

click_code = '''
                    chart_treemap.on('click', function(params)
                        {
                            if (params.value == 0)
                            {
                                window.open("./interfaces/" + params.name + ".html");
                            }
                        }
                    );

             '''

data_view_code  = '''.setOption({
                                toolbox: {
                                    feature: {
                                        dataView: {
                                            optionToContent:  (opt) => {
                                                let axisData = opt.xAxis[0].data;
                                                let series = opt.series;
                                                let tdHeads = '<td style="background-color: #eeeeee;font-weight: 700;color: #333333">输入输出类型</td>';
                                                let tdBodys = '';
                                                console.log(series)
                                                series.forEach(function (item) {
                                                  tdHeads += `<td style="background-color: #eeeeee;font-weight: 700;color: #333333">${item.name}</td>`;
                                                });
                                                let table = `<table border="1" style=" width: 95%;margin-left:20px;border-collapse:collapse;table-layout：fixed;font-size:14px;text-align:center" class="dataViewTable"><tbody><tr>${tdHeads} </tr>`;
                                                for (let i = 0, l = axisData.length; i < l; i++) {
                                                  for (let j = 0; j < series.length; j++) {
                                                    if (j < 5){
                                                        tdBodys += `<td ><input class="${j}x" type="text" value="${series[j].data[i]}" style="width: 120px;border: none;text-align: center;color: #444444;color: #444444"></td>`;
                                                    }
                                                    else{
                                                        let ratio = series[j].data[i][1];
                                                        tdBodys += `<td><input class="${j}x" type="text" value="${ratio}" style="width: 120px;border: none;text-align: center;color: #444444;color: #444444"></td>`;
                                                    }
                                                  }
                                                  table += `<tr><td><input type="text" value="${axisData[i]}" style="width: 500px;border: none;text-align: left;color: #444444"> </td>${tdBodys}</tr>`;
                                                  tdBodys = '';
                                                }
                                                table += '</tbody></table>';
                                                return table;
                                              },

                                              contentToOption: (HTMLDomElement, opt) => {
                                                if(document.getElementsByClassName('dataViewTable').length>1){
                                                    window.alert("有其他未关闭的数据视图，请关闭后重试");
                                                }
                                                else{
                                                    for(let i = 0;i < opt.series.length;i++){
                                                      var name = 'dataX' + i;
                                                      window[name] = []
                                                      for (let j of document.getElementsByClassName(`${i}x`) ){
                                                        window[name].push(j.value)
                                                      }
                                                      opt.series[i].data = window[name]
                                                    }
                                                return opt;}
                                              },
                                        }
                                    }
                                }
                            })
                  '''

filter_w = 0

def make_ratio(res0, res1):
    if len(res0) == 0 or len(res1) == 0:
        return ""
    else:
        if float(res0) > 0.0 and float(res1) > 0.0:
            return str(round(float(res0) / float(res1), 3))
        else:
            return ""

def make_bar(chart_id, title, subtitle, result) -> Bar:

    param_list = []
    opencv_result = []
    none_result = []
    neon_result = []
    opencl_result = []
    hvx_result = []

    num = 0
    for x, y in result.items():
        param_list.append("] ".join(("[" + str(num), x)))
        num += 1

        if "OpenCV" in y:
            opencv_result.append(y["OpenCV"])
        else:
            opencv_result.append("")

        if "None" in y:
            none_result.append(y["None"])
        else:
            none_result.append("")

        if "Neon" in y:
            neon_result.append(y["Neon"])
        else:
            neon_result.append("")

        if "Opencl" in y:
            opencl_result.append(y["Opencl"])
        else:
            opencl_result.append("")

        if "Hvx" in y:
            hvx_result.append(y["Hvx"])
        else:
            hvx_result.append("")

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="70%", height="700px", chart_id=chart_id, page_title=title))
        .add_xaxis(param_list)
        .add_yaxis("OpenCV", opencv_result, gap="0%", category_gap="10%", color="#D3ECFA")
        .add_yaxis("None", none_result, gap="0%", category_gap="10%", color="#C5F576")
        .add_yaxis("Neon", neon_result, gap="0%", category_gap="10%", color="#FFC068")
        .add_yaxis("OpenCL", opencl_result, gap="0%", category_gap="10%", color="#1C7FE8")
        .add_yaxis("HVX", hvx_result, gap="0%", category_gap="10%", color="#E04020")
        .extend_axis(
            yaxis=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value} X")
                )
        )
        .add_js_funcs("chart_" + chart_id + data_view_code)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, subtitle=subtitle, subtitle_textstyle_opts=opts.TextStyleOpts(font_size=15)),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(margin=5)),
            yaxis_opts=opts.AxisOpts(name="time(ms)", name_location="middle", axislabel_opts=opts.LabelOpts(formatter="{value}")),
            legend_opts=opts.LegendOpts(type_="scroll", item_gap=8, item_width=20, pos_left="15%", pos_right="20%"),
            toolbox_opts=opts.ToolboxOpts(orient="horizontal",
                                          feature=opts.ToolBoxFeatureOpts(
                                              magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                                              data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False),
                                              save_as_iaura=opts.ToolBoxFeatureSaveAsIauraOpts(type_="png", background_color="#fff")
                                          )
                                         ),
            brush_opts=opts.BrushOpts(tool_box=[]),
            datazoom_opts=[opts.DataZoomOpts(pos_bottom="0%"), opts.DataZoomOpts(type_="inside")],
        )
    )

    opencv_none_ratio = [make_ratio(opencv_result[i], none_result[i]) for i in range(len(opencv_result))]
    opencv_neon_ratio = [make_ratio(opencv_result[i], neon_result[i]) for i in range(len(opencv_result))]
    opencv_opencl_ratio = [make_ratio(opencv_result[i], opencl_result[i]) for i in range(len(opencv_result))]
    opencv_hvx_ratio = [make_ratio(opencv_result[i], hvx_result[i]) for i in range(len(opencv_result))]

    none_neon_ratio = [make_ratio(none_result[i], neon_result[i]) for i in range(len(none_result))]
    none_opencl_ratio = [make_ratio(none_result[i], opencl_result[i]) for i in range(len(none_result))]
    none_hvx_ratio = [make_ratio(none_result[i], hvx_result[i]) for i in range(len(none_result))]

    neon_opencl_ratio = [make_ratio(neon_result[i], opencl_result[i]) for i in range(len(neon_result))]
    neon_hvx_ratio = [make_ratio(neon_result[i], hvx_result[i]) for i in range(len(neon_result))]

    line = (
            Line()
            .add_xaxis(param_list)
            .add_yaxis(
                series_name="OpenCV/None",
                is_selected=False,
                y_axis=opencv_none_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="OpenCV/Neon",
                is_selected=False,
                y_axis=opencv_neon_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="OpenCV/Opencl",
                is_selected=False,
                y_axis=opencv_opencl_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="OpenCV/HVX",
                is_selected=False,
                y_axis=opencv_hvx_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="None/Neon",
                is_selected=False,
                y_axis=none_neon_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="None/Opencl",
                is_selected=False,
                y_axis=none_opencl_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="None/HVX",
                is_selected=False,
                y_axis=none_hvx_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="Neon/Opencl",
                is_selected=False,
                y_axis=neon_opencl_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            .add_yaxis(
                series_name="Neon/HVX",
                is_selected=False,
                y_axis=neon_hvx_ratio,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                z_level=1,
                yaxis_index=1)
            )

    bar.overlap(line)

    return bar

def make_pie(passed_num, failed_num):
    pie = (
            Pie(init_opts=opts.InitOpts(width="20%", height="700px"))
            .add("", [("passed", passed_num), ("failed", failed_num)],
                 center=["50%", "55%"], radius=["25%", "50%"],
                 label_opts=opts.LabelOpts(is_show=False))
            .set_colors(["#7AFA8A", "#FA6F56"])
            .set_global_opts(title_opts=opts.TitleOpts(title="Test Status", pos_top = "20%"),
                             legend_opts=opts.LegendOpts(pos_top="25%"))
        )
    return pie

class Param:
    def __init__(self, name):
        self.name = name
        self.idx = 0
        self.acce_result = {}
        self.pn = 0
        self.fn = 0

    def load_data(self, input_type, output_type, impl, value):
        iw = 0
        ow = 0
        if input_type.find("x") != -1:
            index = input_type.find("x")
            str_w = input_type[index + 1:input_type.find("x", index + 1)]
            if str_w.isdigit():
                iw = int(str_w)

        if output_type.find("x") != -1:
            index = output_type.find("x")
            str_w = output_type[index + 1:output_type.find("x", index + 1)]
            if str_w.isdigit():
                ow = int(str_w)

        if iw > filter_w or ow > filter_w:
            key = "\n -> ".join((input_type, output_type))
            if key in self.acce_result:
                self.acce_result[key][impl] = value
            else:
                self.acce_result.setdefault(key, {})[impl] = value
        elif iw == 0 and ow == 0:
            key = "\n -> ".join((input_type, output_type))
            if key in self.acce_result:
                self.acce_result[key][impl] = value
            else:
                self.acce_result.setdefault(key, {})[impl] = value

    def load_status(self, status):
        if status == "passed":
            self.pn += 1
        elif status == "failed":
            self.fn += 1

class Interface:
    def __init__(self, name):
        self.name = name
        self.params = {}

    def load_data(self, param, input_type, output_type, perf_result):
        if param not in self.params:
            self.params[param] = Param(param)

        for impl, value in perf_result.items():
            avg_time = str(round(float(value["avg"]), 2))
            self.params[param].load_data(input_type, output_type, impl, avg_time)

    def load_status(self, param, status):
        if param not in self.params:
            self.params[param] = Param(param)

        self.params[param].load_status(status)

    def make_chart(self, path):
        page = Page(layout=Page.SimplePageLayout, page_title=self.name)
        number = 0
        for key, value in self.params.items():
            title = self.name
            subtitle = key
            chart_id = title + str(number)
            page.add(make_bar(chart_id, title, subtitle, value.acce_result))
            page.add(make_pie(value.pn, value.fn))
            number += 1

        page.render(path + self.name + ".html")

def load_test_json(filepath, aura_tree):
    with open(filepath, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    if "default" in data:
        for test_case, test_info in data["default"].items():
            if test_case != "":
                md_idx = -1
                if_idx = -1
                if test_info["module"] != "":
                    module_name = test_info["module"]
                    interface_name = test_info["interface"]
                    for i in range(len(aura_tree["children"])):
                        if module_name in aura_tree["children"][i].values():
                            md_idx = i
                            break
                    if md_idx == -1:
                        if aura_tree["children"][md_idx]:
                            aura_tree["children"].append({})
                        aura_tree["children"][md_idx]["name"] = test_info["module"]
                        aura_tree["children"][md_idx]["children"] = []
                        aura_tree["children"][md_idx]["children"].append({"name": interface_name, "value": 0, "data": Interface(interface_name)})
                    else:
                        for i in range(len(aura_tree["children"][md_idx]["children"])):
                            if interface_name in aura_tree["children"][md_idx]["children"][i].values():
                                if_idx = i
                                break
                        if if_idx == -1:
                            aura_tree["children"][md_idx]["children"].append({"name": interface_name, "value": 0, "data": Interface(interface_name)})

                    test_result = test_info["result"]
                    interface = aura_tree["children"][md_idx]["children"][if_idx]
                    for test_data in test_result:

                        interface["data"].load_status(test_data["param"], test_data["accu_status"])

                        if test_data["perf_status"] == "passed":
                            interface["data"].load_data(test_data["param"],
                                                        test_data["input"],
                                                        test_data["output"],
                                                        test_data["perf_result"])
    f.close()

if __name__ == '__main__':
    if pyecharts_version[:-2] != "1.9":
        print("pyecharts must be version 1.9.x, please check it.")
        sys.exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--device", nargs='?', const="1", help="android device serial")
    parser.add_argument("-p", "--path", required=True, help="json file path or folder path")
    parser.add_argument("-f", "--filter", type=int, default=0, help="filter of width for input/output data, int type")
    parser.add_argument("-o", "--output", default="visual_report", help="the absolute path of output folder name")
    args = parser.parse_args()

    filter_w = args.filter

    if ".json" in args.path:
        if args.path.rfind("/") != -1:
            folder = args.path[args.path.rfind("/") + 1:args.path.rfind(".")]
        else:
            folder = args.path[:args.path.rfind(".")]
    else:
        folder = "visual_report"

    output_path = None
    if args.output is not None:
        output_path = os.path.join(args.output, folder)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "interfaces/"), exist_ok=True)

        output_path = output_path + "/"
    else:
        os.system("mkdir -p " + folder)
        os.system("mkdir -p ./" + folder + "/interfaces")
        output_path = "./" + folder + "/"

    if args.device == "1":
        os.system("adb pull " + args.path)
    elif args.device != None:
        os.system("adb -s " + args.device + " pull " + args.path)

    if args.device == None:
        filepath = args.path
    else:
        filepath = "." + args.path[args.path.rfind("/"):]

    # tree_map init
    aura_tree = {
                    "name": "ops",
                    "children": [{}],
                }

    # load json file
    if ".json" in args.path:
        load_test_json(filepath, aura_tree)
    else:
        for f in glob.glob(filepath + "/*.json"):
            load_test_json(f, aura_tree)

    # make charts
    count = 0
    if {} in aura_tree["children"]:
        print("No module found! Please check the path")
    else:
        for module in aura_tree["children"]:
            for interface in module["children"]:
                interface["data"].make_chart(output_path + "interfaces/")
                count += 1

        if (count > 1):
            c = (
                Tree(init_opts=opts.InitOpts(width="1200px", height="800px", chart_id="treemap", page_title="Aura 2.0"))
                .add("", [aura_tree], collapse_interval=2,
                    label_opts=opts.LabelOpts(position="top"),
                    leaves_label_opts=opts.LabelOpts(position="right"))
                .add_js_funcs(click_code)
                .set_global_opts(title_opts=opts.TitleOpts(title="Aura 2.0"))
                .render(output_path + "Aura 2.0.html")
            )

        print("Visualisation Finished!")
