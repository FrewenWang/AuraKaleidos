# AliceFlutter

Flutter 计数器和技术笔记示例。当前代码来自 Flutter 1.x 时代，`pubspec.yaml` 限定 Dart
`>=2.1.0 <3.0.0`；需要使用匹配 SDK，或先按上级 README 的说明完成 Flutter 3 迁移。

```bash
flutter pub get
flutter analyze
flutter test
flutter run
```

自动测试位于 `test/widget_test.dart`，覆盖计数器初值和点击递增。`android/`、`ios/` 是平台
宿主工程，Android 构建需要 Android SDK，iOS 构建只能在安装 Xcode 的 macOS 上执行。

## 学习文档

- [如何创建 Flutter Plugin Module 并发布](blog/docs/01.how-to-create-flutter-plugin-module-and-publish-it.md)
- [Flutter 启动流程](blog/docs/flutter-engine/01.Flutter启动流程.md)
