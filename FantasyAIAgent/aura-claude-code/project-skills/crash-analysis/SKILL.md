---
name: crash-analysis
description: 从 logs/crash 中读取 Android/native crash 日志和带符号 SO，解析堆栈并生成详细中文分析报告。
---

# Crash Analysis Skill

当用户要求分析 crash、tombstone、native crash、ASan crash、SIGSEGV/SIGABRT，或者类似的崩溃问题时，使用这个 skill。

分析必须结合：

- 当前工程源码。
- crash 日志。
- 带符号表的 `.so` 库。

最终必须生成一份详细的中文 markdown crash 分析文档，除非缺少必要输入文件。

## 默认输入

默认当前工作目录就是项目根目录。

默认 crash 文件目录：

```text
logs/crash/
```

该目录中通常应包含：

- 一个或多个 crash 日志，例如 `.log`、`.txt`、tombstone 文件、Android logcat 导出文件。
- 一个或多个带符号表的共享库，例如 `.so` 文件。

如果 `logs/crash/` 不存在，或者目录中没有可用 crash 日志，或者没有带符号表的 `.so` 文件，需要先明确告诉用户缺少哪些文件，并让用户添加到该目录后再继续。不要在缺少关键文件时凭空猜测。

如果用户显式提供了其他 crash 目录或具体文件，则优先使用用户提供的位置，而不是 `logs/crash/`。

## 目标

分析结束时，必须生成一份详细的中文 crash 分析文档。文档应包含：

- crash 摘要。
- 崩溃进程、线程、signal、fault address 等运行时信息。
- 关键原始日志摘录。
- 符号化后的 native 堆栈。
- 源码映射，使用可点击的文件链接和行号。
- 代码路径分析。
- 根因判断。
- 修复建议。
- 验证建议。
- 当前证据不足或仍需补充的信息。

## 分析流程

### 1. 定位输入文件

先列出 crash 目录中的文件。

优先使用：

```bash
rg --files logs/crash
```

需要识别：

- crash 日志候选文件。
- 带符号表的 `.so` 候选文件。
- 是否存在压缩包或尚未解压的文件。

不要仅凭 `.so` 文件名相似就认为它和 crash 日志匹配。日志中如果有 BuildId，需要优先用 BuildId 校验。

### 2. 从日志中提取 crash 关键信息

搜索 native crash 常见关键字：

```bash
rg -n "Fatal signal|signal [0-9]+|SIGSEGV|SIGABRT|AddressSanitizer|backtrace:|Abort message|pid: [0-9]+, ppid:|tid:|fault addr|SUMMARY:|BuildId|\.so\+0x" logs/crash
```

如果输出过大，需要缩小范围，例如按以下关键字继续搜索：

- `AddressSanitizer`
- `Fatal signal`
- `backtrace:`
- 目标 `.so` 名称
- 进程名
- 线程名

需要截取足够的上下文，理解崩溃前后的时间线：

- crash 前的关键业务日志。
- ASan 或 tombstone 主体。
- backtrace 块。
- 进程退出、服务重启、abort 信息。

如果日志文件很大，不要一次性读取全部内容。应先通过 `rg -n` 定位行号，再按行号窗口读取关键片段。

### 3. 判断 crash 类型

根据日志对 crash 类型做初步分类：

- `SIGSEGV`：非法内存访问。
- `SIGABRT`：通常是 ASan 捕获问题后主动 abort，或者代码显式 abort。
- ASan `SEGV on unknown address`：常见于空指针、野指针、use-after-free、对象状态损坏。
- ASan `heap-use-after-free`、`stack-use-after-return`、`heap-buffer-overflow` 等：以 ASan 分类作为最强证据。

需要记录：

- 进程名。
- 线程名和 tid。
- signal 和 code。
- fault address。
- PC。
- ASan summary。
- 相关 SO 的 BuildId。

### 4. 匹配运行时 SO 和本地符号 SO

对于日志中的帧，例如：

```text
/path/libxxx.so+0x123456 (BuildId: abc...)
```

需要在 crash 目录中找到对应的本地 `.so` 文件。

用以下方式检查 BuildId：

```bash
readelf -n logs/crash/libxxx.so | rg -i "Build ID|BuildId"
```

如果 BuildId 不匹配，需要在分析中说明符号化结果可能不可靠，并让用户补充匹配的带符号 SO。

### 5. 符号化 native 堆栈

使用 `addr2line` 或 `llvm-addr2line` 解析本地带符号 SO。

必须使用日志中的模块相对偏移，例如 `libxxx.so+0x123456` 里的 `0x123456`，不要直接使用运行时绝对地址。

示例：

```bash
addr2line -f -C -e logs/crash/libxxx.so 0x123456 0x234567 0x345678
```

如果日志中同时出现：

- `pc 000000...`
- `libxxx.so+0x...`

优先使用模块相对偏移。

符号化结果需要整理为：

- 函数名。
- 源文件路径。
- 源码行号。

如果解析出的源码路径来自其他机器，不要认为它无效。应根据路径后缀在当前工程中查找对应文件。用户已说明：解析路径可能不是本地路径，但代码节点是对齐的。

### 6. 阅读关键源码

读取崩溃首帧和若干调用者附近的源码。

重点关注：

- 精确崩溃行。
- 崩溃行使用的输入对象和变量。
- 对象所有权和生命周期。
- 数组、vector、指针、引用的访问。
- 是否有空指针检查。
- 错误路径和清理路径。
- 异步调用边界，例如 `AsyncCall`、future、worker pool。
- `Free`、`Delete`、`clear`、析构、timeout、early exit 等路径。

搜索相关变量和生命周期函数时，优先使用 `rg`。

示例：

```bash
rg -n "variable_name|FreeMems|clear\(|DeleteArray|AsyncCall|WaitFutureWithTimeout" develop/src
```

### 7. 建立因果链

分析时按以下顺序组织证据：

1. 崩溃现象：signal、进程、线程、fault address。
2. 崩溃位置：符号化后的函数、源码文件、行号。
3. 崩溃行正在执行什么操作。
4. 要让该操作崩溃，哪些状态必须是异常的。
5. 哪些代码路径可能造成这些异常状态。
6. 当前证据能证明直接原因，还是只能推断更深层根因。

必须区分确定性：

- “直接原因”：已经由日志和源码明确证明的 fault 指令或源码行。
- “高概率根因”：从对象状态、生命周期、上下文推断出来，但日志未完全闭环证明的原因。
- “待确认项”：当前日志和代码还不能证明，需要额外日志或复现验证。

不要把推断写成铁证。证据不足时要明确说明。

### 8. 生成分析文档

在 crash 目录下创建 markdown 报告。

推荐文件名：

```text
logs/crash/crash_analysis_<YYYYMMDD_HHMMSS>.md
```

如果 crash 日志文件名中自带时间戳，也可以使用 crash 日志时间戳。若同目录有多个 crash，文件名中加入简短标识以避免混淆。

报告建议使用以下中文结构：

```markdown
# Crash 分析报告

## 1. 结论摘要

## 2. 输入文件

## 3. 崩溃现场信息

## 4. 符号化堆栈

## 5. 源码映射

## 6. 代码路径分析

## 7. 根因判断

## 8. 修复建议

## 9. 验证建议

## 10. 待确认问题
```

源码引用必须使用可点击链接和行号。

引用当前工程文件时，优先使用相对项目根目录的链接，例如：

```markdown
[develop/src/foo.cpp:123](develop/src/foo.cpp#L123)
```

需要包含关键日志摘录，但不要粘贴大段无关日志。日志摘录应服务于结论。

### 9. 最终回复

生成文档后，最终回复需要包含：

- 分析报告路径，可点击链接。
- 最可能原因的一句话摘要。
- 如果存在缺失文件、BuildId 不匹配、证据不足，需要明确说明。

不要只描述分析步骤。除非被缺失输入阻塞，否则必须完成详细 crash 分析文档的生成。

## 当前项目注意事项

本项目默认将 crash 相关文件放在：

```text
logs/crash/
```

旧的 crash 文件也可能出现在：

```text
crash/
```

但除非 `logs/crash/` 缺失且用户同意，或者用户明确指定，否则不要默认使用 `crash/`。

本项目包含 Android camera/native 算法代码。符号化结果中的源码路径可能来自其他机器，路径前缀不同是正常情况。需要通过文件名和路径后缀映射到当前工程源码。
