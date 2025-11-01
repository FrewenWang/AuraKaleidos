-dontoptimize
-keepattributes InnerClasses
-keep public class com.baidu.iov.vision.ability.core.bean.*{*;}
-keep public class com.baidu.iov.vision.ability.core.request.*{public *;}
-keep public class com.baidu.iov.vision.ability.core.result.*{public *;}
-keep public class com.baidu.iov.vision.ability.core.VisionCallback{public *;}
-keep public class com.baidu.iov.vision.ability.core.VisionCallbackAdapter{public *;}
-keep public class com.baidu.iov.vision.ability.util.PerformanceUtil{*;}
-keep public class com.baidu.iov.vision.ability.util.VisionNativeHelper{public *;}
-keep public class com.baidu.iov.vision.ability.config.VisionConfig{public *;}
-keep public class com.baidu.iov.vision.ability.config.VisionConfig$*{*;}
-keep public class com.baidu.iov.vision.ability.config.VisionConfig$Key
-keep public class com.baidu.iov.vision.ability.VisionInitializer{public *;}
-keep public class com.baidu.iov.vision.ability.VisionInitializer$*{*;}
-keep public class com.baidu.iov.vision.ability.VisionInitializer$Callback
-keep public class com.baidu.iov.vision.ability.VisionService{public *;}

# 保留本地 native 方法不被混淆
-keepclasseswithmembernames class * {
    native <methods>;
}

# 避免混淆泛型
-keepattributes Signature

# 指定不去忽略非公共库的类成员
-dontskipnonpubliclibraryclassmembers

# 抛出异常时保留代码行号
-keepattributes SourceFile,LineNumberTable