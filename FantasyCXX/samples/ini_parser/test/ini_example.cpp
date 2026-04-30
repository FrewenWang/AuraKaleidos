//
// Created by wangzhijiang on 25-4-30.
//
#include <iostream>
#include "xiaomi/parser/iniparser.h"

using namespace xiaomi::parser;

int main() {

  IniParser parser;
  auto url = "/Users/frewen/03.ProgramSpace/01.WorkSpace/AuraKaleidoscope/AuraFantasyCXX/ini_parse/assets/example.ini";

  if (parser.Parse(url)) {
    // read
    const int port = parser.GetInt("udp", "port", 8080);
    const char *ip = parser.GetStr("udp", "ip", "127.0.0.1");
    const char* family = parser.GetStr("udp", "family", "AF");
    const float mtu = parser.GetFloat("udp", "mtu", 1500.0f);


    std::cout << "GetInt: " << port << "\n";
    std::cout << "GetString: " << ip << "\n";
    std::cout << "GetString: " << family << "\n";
    std::cout << "GetFloat: " << mtu << "\n";

    // 修改配置
    parser.SetStr("udp", "family", "AF_Cumstom");
    parser.SetInt("udp", "port", 9090);
    parser.SetFloat("udp", "mtu", 16000.0f);
    parser.SetFloat("my_section", "my_entry", 100.0f);
    const bool ret = parser.Save("example_new.ini");
    std::cout << "save ini file ret: " << ret << "\n";
  }
  return 0;



  return 0;
}