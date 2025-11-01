//
// Created by frewen on 11/28/22.
//
#include <sstream>

#include "aura/utils/string_util.h"

namespace aura::utils {

bool StringUtil::splitStr(const std::string &content, char delimiter, std::vector<std::string>& results) {
	std::stringstream ss(content);
	std::string token;
	while (std::getline(ss, token, delimiter)) {
		results.push_back(token);
	}
	return true;
}


}  // namespace aura_lib