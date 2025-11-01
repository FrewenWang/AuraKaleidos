//
// Created by Frewen.Wang on 2022/7/23.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <string>

using namespace std;

int main(int argc, char **argv) {
	ALOGD("TAG", "[AuraCV] Hello AuraCVUnitTester");
	std::string imagePath;
	if (argc > 1) {
		imagePath = argv[1];
	}
	testing::GTEST_FLAG(output) = "AuraKaleidoCXXUTReport.json";
	testing::InitGoogleTest(&argc, argv);
	if (RUN_ALL_TESTS() == 0) {
		return 0;
	}
	return -1;
}