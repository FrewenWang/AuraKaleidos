## Dual camera detect demo
双摄像头检测示例程序
### 编译说明
demo工程需要单独手动编译，CMakeLists中仅配置了Release模式下的mac编译脚本，如果在其他平台，需要修改头文件、库的路径；
### 代码说明
demo演示了libvision的多摄像头支持能力，基本调用方式如下：
```c++
    // 多实例仅需要一次初始化，完成模型加载
    VisionInitializer initializer;
    initializer.init();

    // 定义VisionService多实例
    VisionService service1;
    VisionService service2;
    // 每个实例单独初始化，包括参数配置和开关设置
    init_service1(&service1);
    init_service2(&service2);

    cv::Mat frame1;
    cv::Mat frame2;    
    while(true) {
        cap1 >> frame1;
        if (frame1.empty()) {
            std::cerr << "frame1 is empty" << std::endl;
            break;
        }

        cap2 >> frame2;
        if (frame2.empty()) {
            std::cerr << "frame2 is empty" << std::endl;
            break;
        }

        // 分别调用detect接口进行检测
        detect(&service1, frame1, 1);
        detect(&service2, frame2, 2);

        cv::waitKey(1);
    }
```