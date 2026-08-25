# FantasyKotlin

Kotlin/Android 工程集合。Android 项目保留 Gradle 标准布局，多模块工程由根 `settings.gradle` 管理。

| 工程 | Android/Gradle 基线 | 推荐 JDK | 验证 |
|---|---|---:|---|
| `NyxGitHubKotlin` | compileSdk 29、AGP 4.0、Gradle 6.1.1 | 8 或 11 | `./gradlew test` |
| `hello_kotlin_android` | compileSdk 28、AGP 3.5、Gradle 5.4.1 | 8 | `./gradlew test` |

这两个工程用于保留旧版 Android/Kotlin API 示例，当前 wrapper 与 JDK 17/20 不兼容。请使用
Android Studio 为对应工程选择兼容 JDK，并安装 SDK Platform 28/29；不要把作者机器上的
`local.properties` 提交到仓库。

```bash
cd FantasyKotlin/hello_kotlin_android
./gradlew test --no-daemon
./gradlew assembleDebug --no-daemon
```

设备测试使用 `./gradlew connectedAndroidTest`，需要已启动的模拟器或设备。若要升级到当前
Android Gradle Plugin，应把 Gradle、Kotlin、AndroidX、namespace、compileSdk 和废弃 API
作为一次独立迁移并完成真机回归；只升级 wrapper 通常无法得到可运行工程。
