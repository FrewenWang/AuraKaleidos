//
// aura-config-parser 示例：解析 / 读取 / 修改 / 保存 INI 配置。
//

#include "ini_parser.h"

#include <iostream>
#include <fstream>

int main() {
    const std::string path = "demo_config.ini";

    // 写一份示例 INI
    {
        std::ofstream ofs(path);
        ofs << "[server]\n";
        ofs << "host = 127.0.0.1\n";
        ofs << "port = 8080\n";
        ofs << "; this is a comment\n";
        ofs << "[client]\n";
        ofs << "timeout = 30\n";
    }

    IniParser parser(path);

    std::cout << "host   = " << parser.GetValue<std::string>("server", "host", "") << "\n";
    std::cout << "port   = " << parser.GetValue<int>("server", "port", 0) << "\n";
    std::cout << "timeout= " << parser.GetValue<int>("client", "timeout", 0) << "\n";
    std::cout << "HasSection(client) = " << parser.HasSection("client") << "\n";
    std::cout << "HasKey(server,host)= " << parser.HasKey("server", "host") << "\n";

    // 修改并保存
    parser.SetValue("server", "port", "9090");
    parser.Save();
    std::cout << "after SetValue+Save, port = " << parser.GetValue<int>("server", "port", 0) << "\n";

    return 0;
}
