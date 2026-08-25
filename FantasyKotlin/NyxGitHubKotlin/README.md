# NyxGitHubKotlin

基于 Kotlin、MVVM、Data Binding、Dagger、Retrofit 与 Android IPC 示例的多模块学习应用。
`nyx-github-app` 是应用 module，`nyx-github-app-library` 保存公共 UI、网络与 ViewModel 基础代码。

工程保留 Kotlin 1.3.72、Android Gradle Plugin 4.0、Gradle 6.1.1 和 compileSdk 29 的历史组合，
推荐使用 JDK 8/11 与包含 Android SDK Platform 29 的 Android Studio 环境。

```bash
./gradlew test --no-daemon
./gradlew assembleDebug --no-daemon
./gradlew connectedAndroidTest          # 需要设备或模拟器
```

部分示例连接 GitHub API、WebSocket 或其他外部服务；单元测试不得使用真实 token。升级依赖前先
核对各示例的 manifest 权限、ProGuard、Data Binding 生成代码和 AndroidX 兼容性。
