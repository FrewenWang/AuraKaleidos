# AliceAndroid

Android 工程集合。各工程遵循 Gradle 的 module 结构：`src/main`、`src/test` 和 `src/androidTest`，
不额外套用通用 `src/` 目录。

## IPCSample

`IPCSample/` 演示跨进程 AIDL 服务，默认构建客户端 `app` 和独立服务端
`ipc_service_sample`。`ipc_service` 是同一应用内远程 Service 的备用实现，需要时可在
`settings.gradle` 中切换 module 组合。

构建要求：JDK 17、Android SDK Platform 28，以及可访问 Google Maven/Maven Central 的网络。
工程使用 Android Gradle Plugin 8.3 和 Gradle 8.4，仓库配置已移除停止服务的 JCenter。

```bash
cd AliceAndroid/IPCSample
./gradlew test --no-daemon
./gradlew assembleDebug --no-daemon
```

Windows 使用 `gradlew.bat`。若命令提示找不到 SDK，请通过 Android Studio 安装 Platform 28，
并在本机 `local.properties` 中设置 `sdk.dir`；该文件包含机器路径，不应提交。`src/test` 是无需
设备的 JVM 单元测试，`src/androidTest` 需要模拟器或 Android 设备：

```bash
./gradlew connectedAndroidTest
```

当前示例仍使用 Android Support Library 28，用于保留旧 IPC 教程行为；如迁移到 AndroidX，
应单独完成依赖、import、manifest 和真机 IPC 回归，不要只替换依赖名称。
