//
// Created by frewen on 11/28/22.
//
#include <sstream>

#include "aura/aura_utils/utils/StringUtil.h"

namespace aura {
namespace aura_utils {

bool StringUtil::splitStr(const std::string &content, char delimiter, std::vector<std::string>& results) {
	std::stringstream ss(content);
	std::string token;
	
	while (std::getline(ss, token, delimiter)) {
		results.push_back(token);
	}
	return true;
}


} // namespace aura
} // namespace aura_lib