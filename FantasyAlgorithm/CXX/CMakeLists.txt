# 第一步： 设置CMake版本号
cmake_minimum_required(VERSION 3.10)
# 第二步： 设置工程名称
# 设置工具类库的名字
set(TARGET_NAME AliceAlgorithm)

message(STATUS "[${TARGET_NAME}] ====================Start Build============================")
# 第三步：设置CMake设置的C++的版本
project(${TARGET_NAME})
set(CMAKE_CXX_STANDARD 11)

# 第四步：指定生成的目标产出
#add_executable(${TARGET_NAME} src/L0005_longest-palindromic-substring.cc)
#add_executable(${TARGET_NAME} src/L0003_longest-substring-without-repeating-characters.cc)
#add_executable(${TARGET_NAME} src/L0027_remove_element.cc)
#add_executable(${TARGET_NAME} src/L0026_remove-duplicates-from-sorted-array.cc)
#add_executable(${TARGET_NAME} src/L0035_search-insert-position.cc)
#add_executable(${TARGET_NAME} src/L0079_word-search.cc)
#add_executable(${TARGET_NAME} src/L0080_remove-duplicates-from-sorted-array-ii)
add_executable(${TARGET_NAME} src/L0122_best-time-to-buy-and-sell-stock-ii.cc)
#add_executable(${TARGET_NAME} src/L0392_is-subsequence.cc)


message(STATUS "[${TARGET_NAME}] ====================Build End============================")