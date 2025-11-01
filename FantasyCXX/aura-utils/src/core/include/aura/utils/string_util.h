//
// Created by frewen on 1/4/23.
//
#pragma once

#include <iostream>
#include <vector>


namespace aura::utils {

class StringUtil {
public:
	/**
	 * 进行字符串分割
	 * @param content  需要分割的内容
	 * @param delimiter
	 * @param results
	 * @return
	 */
	static bool splitStr(const std::string &content, char delimiter, std::vector<std::string> &results);
	
};

}
