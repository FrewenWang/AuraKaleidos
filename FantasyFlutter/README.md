# FantasyFlutter

Flutter 工程集合。每个应用遵循 Flutter 官方的 `lib/`、`test/`、`assets/`、`android/` 和 `ios/`
布局，资源需同时在 `pubspec.yaml` 中声明。

当前 `AliceFlutter/` 是 Flutter 1.x/Dart 2 的历史计数器示例，SDK 约束为 Dart 2.1–2.x，不能
直接假定兼容当前 Flutter 3。使用匹配的旧 SDK 时：

```bash
cd FantasyFlutter/AliceFlutter
flutter pub get
flutter analyze
flutter test
flutter build apk       # Android SDK 环境
# flutter build ios     # macOS + Xcode
```

`test/widget_test.dart` 验证初始计数和按钮递增。迁移至 Flutter 3 时需同时处理 null safety、
`TextTheme.display1` 等废弃 API、Dart SDK 约束、依赖锁文件，以及 Android/iOS 平台模板；应使用
`flutter create .` 的差异作为参考并分别完成 Android/iOS 构建测试。
