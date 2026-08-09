# Alice Algorithm C++

小型、可测试的 C++17 算法库。

- `include/alice_algorithm/`：公共头文件。
- `src/`：实际参与库构建的实现。
- `tests/`：CTest 单元测试。
- `examples/leetcode/`：彼此独立的历史题解片段，不自动参与库构建。
- `examples/sorting/`：排序算法教学代码。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
