# FantasySwiftIOS

Swift/iOS 学习工程集合。当前包含 `AliceSwiftIOSApp`，这是一个 UIKit + Storyboard 示例，最低
部署版本为 iOS 16.2，并包含 XCTest 单元测试与 XCUITest 启动测试。

## 打开与构建

使用 Xcode 打开 `AliceSwiftIOSApp.xcodeproj`，选择 `AliceSwiftIOSApp` scheme 和可用模拟器。
命令行可先检查工程结构，再执行无签名的模拟器构建：

```bash
cd FantasySwiftIOS
xcodebuild -list -project AliceSwiftIOSApp.xcodeproj
xcodebuild build-for-testing \
  -project AliceSwiftIOSApp.xcodeproj \
  -scheme AliceSwiftIOSApp \
  -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO
```

## 测试

将示例设备名称替换为当前 Xcode 已安装的模拟器：

```bash
xcodebuild -showdestinations \
  -project AliceSwiftIOSApp.xcodeproj -scheme AliceSwiftIOSApp

xcodebuild test \
  -project AliceSwiftIOSApp.xcodeproj \
  -scheme AliceSwiftIOSApp \
  -destination 'platform=iOS Simulator,name=iPhone 15'
```

`AliceSwiftIOSAppTests` 验证主控制器能够加载视图；`AliceSwiftIOSAppUITests` 验证应用能进入前台，
并保留启动性能测试。完整 `test` 需要 Simulator 服务可正常启动；无图形会话的 macOS runner
至少应执行 `build-for-testing`。iOS 工程无法在 Windows/Linux 构建。
