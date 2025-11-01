#include <android/asset_manager_jni.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <jni.h>
#include <map>
#include <numeric>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>
#include <unistd.h>

#include "vision/VisionAbility.h"
#include "dmips_util.h"
#include "log_util.h"
#include "mem_util.h"

static const char* img_buffer = nullptr;
static vision::VisionService serv;
static int g_progress;
static std::string g_progress_tag;

static constexpr int TEST_CNT = 500; // 测试次数
static constexpr float DMIPS_COEFF = 75.52f; // dmips 系数，不同 cpu 需要查询确定
static constexpr int CORE_NUM = 6; // 核数
static constexpr int TEST_FPS = 10; // 除 FaceID 功能外，其他功能 cpumem 的测试帧率
static constexpr int TEST_FACEID_FPS = 3; // FaceID 功能cpumem 的测试帧

using namespace nd;
using namespace vision;
using namespace std::chrono;

// 配置要测试的能力（组合）
std::unordered_map<std::string, bool> _test_switch {
        {"Single",  true},   // 所有单项能力
        {"FaceID",  true},   // FaceID组合能力
        {"DMS",     true},   // DMS 组合能力
        {"MMI",     true},   // MMI 组合能力
        {"DMS_MMI", true},   // DMS + MMI 组合能力
        {"TOTAL",   true}     // 所有能力
};

// 测试的能力列表
// 单项能力
std::vector<std::pair<AbilityId, std::string>> _ability_list {
        {AbilityId::ABILITY_FACE_RECT,                  "faceRect"},
        {AbilityId::ABILITY_FACE_LANDMARK,              "faceLandmark"},
        {AbilityId::ABILITY_FACE_QUALITY,               "faceQuality"},
        {AbilityId::ABILITY_FACE_NO_INTERACTIVE_LIVING, "faceLive"},
        {AbilityId::ABILITY_FACE_FEATURE,               "faceFeature"},
        {AbilityId::ABILITY_FACE_CALL,                  "faceCall"},
        {AbilityId::ABILITY_FACE_DANGEROUS_DRIVING,     "dangerDrive"},
        {AbilityId::ABILITY_FACE_EMOTION,               "emotion"},
        {AbilityId::ABILITY_FACE_ATTRIBUTE,             "attribute"},
        {AbilityId::ABILITY_FACE_EYE_CENTER,            "eyeCenter"},
        {AbilityId::ABILITY_FACE_EYE_GAZE,              "eyeGaze"},
        {AbilityId::ABILITY_FACE_FATIGUE,               "fatigue"},
        {AbilityId::ABILITY_GESTURE_RECT,               "gestureRect"},
        {AbilityId::ABILITY_GESTURE_LANDMARK,           "gestureLandmark"}
};

// 组合能力
std::vector<AbilityId> _faceid_ability_list {
        AbilityId::ABILITY_FACE_RECT,
        AbilityId::ABILITY_FACE_LANDMARK,
        AbilityId::ABILITY_FACE_QUALITY,
        AbilityId::ABILITY_FACE_NO_INTERACTIVE_LIVING,
        AbilityId::ABILITY_FACE_FEATURE
};

std::vector<AbilityId> _dms_ability_list {
        AbilityId::ABILITY_FACE_RECT,
        AbilityId::ABILITY_FACE_LANDMARK,
        AbilityId::ABILITY_FACE_DANGEROUS_DRIVING,
        AbilityId::ABILITY_FACE_CALL
};

std::vector<AbilityId> _mmi_ability_list {
        AbilityId::ABILITY_FACE_RECT,
        AbilityId::ABILITY_FACE_LANDMARK,
        AbilityId::ABILITY_GESTURE_RECT,
        AbilityId::ABILITY_GESTURE_LANDMARK,
        AbilityId::ABILITY_GESTURE_TYPE
};

std::vector<AbilityId> _dms_mmi_ability_list {
        AbilityId::ABILITY_FACE_RECT,
        AbilityId::ABILITY_FACE_LANDMARK,
        AbilityId::ABILITY_FACE_DANGEROUS_DRIVING,
        AbilityId::ABILITY_FACE_CALL,
        AbilityId::ABILITY_GESTURE_RECT,
        AbilityId::ABILITY_GESTURE_LANDMARK,
        AbilityId::ABILITY_GESTURE_TYPE
};

std::vector<AbilityId> _total_ability_list {
        AbilityId::ABILITY_FACE_RECT,
        AbilityId::ABILITY_FACE_LANDMARK,
        AbilityId::ABILITY_FACE_QUALITY,
        AbilityId::ABILITY_FACE_CALL,
        AbilityId::ABILITY_FACE_DANGEROUS_DRIVING,
        AbilityId::ABILITY_FACE_EMOTION,
        AbilityId::ABILITY_FACE_ATTRIBUTE,
        AbilityId::ABILITY_FACE_EYE_CENTER,
        AbilityId::ABILITY_FACE_EYE_GAZE,
        AbilityId::ABILITY_FACE_FATIGUE,
        AbilityId::ABILITY_GESTURE_RECT,
        AbilityId::ABILITY_GESTURE_LANDMARK
};

enum class MixAbilityType {
    FACE_ID = 0,
    DMS = 1,
    MMI = 2,
    DMS_MMI = 3,
    TOTAL = 4
};

struct CpuMemEntry {
    float cpu;
    float mem;
    float cost_time;
};

void set_predetect_switch() {
    serv.set_switches(
            std::unordered_map<short, bool> {
                    {ABILITY_ALL, true},
                    {ABILITY_FACE, true},
                    {ABILITY_FACE_DETECTION, true},
                    {ABILITY_FACE_RECT, true},
                    {ABILITY_FACE_LANDMARK, true},
                    {ABILITY_GESTURE, true},
                    {ABILITY_GESTURE_RECT, true},
                    {ABILITY_GESTURE_LANDMARK, true},
            });
}

static bool g_inited = false;
void init() {
    if (g_inited) {
        return;
    }

    VisionInitializer initializer;
    initializer.init();

    serv.init();
    serv.set_config(ParamKey::USE_INTERNAL_MEM, true);
    serv.set_config(ParamKey::FRAME_CONVERT_FORMAT, COLOR_YUV2BGR_NV21);
    serv.set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    serv.set_config(ParamKey::RELEASE_MODE, BENCHMARK_TEST);
    set_predetect_switch();
    g_inited = true;
    WLog::i("init DONE, libvision version: " + std::string(serv.get_version()));
}

void clear_switch() {
    for (auto& a : _ability_list) {
        serv.set_switch(a.first, false);
    }

    for (auto& a : _faceid_ability_list) {
        serv.set_switch(a, false);
    }

    for (auto& a : _dms_ability_list) {
        serv.set_switch(a, false);
    }

    for (auto& a : _mmi_ability_list) {
        serv.set_switch(a, false);
    }

    for (auto& a : _total_ability_list) {
        serv.set_switch(a, false);
    }
}

void write_file(const std::string& s, const std::string& file_name) {
    std::ofstream ofs(file_name.c_str());
    if (!ofs.is_open()) {
        WLog::e("Write result file failed!");
        return;
    }
    ofs << s;
    ofs.close();
}

void write_dmips_result(const std::string& s) {
    write_file(s, "/sdcard/dmips_result.csv");
}

void write_cpu_mem_result(const std::string& s) {
    write_file(s, "/sdcard/cpu_mem_result.csv");
}

std::string calculate_cpu_mem_result(const std::vector<CpuMemEntry>& v, const std::string& tag) {
    auto max_cpu_iter = std::max_element(v.begin(), v.end(),
                                         [&](const CpuMemEntry& e1, const CpuMemEntry& e2) {
                                             return e1.cpu < e2.cpu;
                                         });
    auto min_cpu_iter = std::min_element(v.begin(), v.end(),
                                         [&](const CpuMemEntry& e1, const CpuMemEntry& e2) {
                                             return e1.cpu < e2.cpu;
                                         });
    auto avg_cpu = std::accumulate(v.begin(), v.end(), 0.f,
                                   [&] (float s, const CpuMemEntry& e) {
                                       return s + e.cpu;
                                   }) / TEST_CNT;
    WLog::i(tag + " CPU max: " + std::to_string(max_cpu_iter->cpu * CORE_NUM)  +
            ", min: " + std::to_string(min_cpu_iter->cpu * CORE_NUM) +
            ", avg: " + std::to_string(avg_cpu * CORE_NUM));

    auto max_mem_iter = std::max_element(v.begin(), v.end(),
                                         [](const CpuMemEntry& e1, const CpuMemEntry& e2) {
                                             return e1.mem < e2.mem;
                                         });
    auto min_mem_iter = std::min_element(v.begin(), v.end(),
                                         [](const CpuMemEntry& e1, const CpuMemEntry& e2) {
                                             return e1.mem < e2.mem;
                                         });
    auto avg_mem = std::accumulate(v.begin(), v.end(), 0.f,
                                   [&] (float s, const CpuMemEntry& e) {
                                       return s + e.mem;
                                   }) / TEST_CNT;
    WLog::i(tag + " MEM max: " + std::to_string(max_mem_iter->mem)  +
            ", min: " + std::to_string(min_mem_iter->mem) +
            ", avg: " + std::to_string(avg_mem));

    auto max_dur_iter = std::max_element(v.begin(), v.end(),
                                         [&](const CpuMemEntry& e1, const CpuMemEntry& e2) {
                                             return e1.cost_time < e2.cost_time;
                                         });
    auto min_dur_iter = std::min_element(v.begin(), v.end(),
                                         [&](const CpuMemEntry& e1, const CpuMemEntry& e2) {
                                             return e1.cost_time < e2.cost_time;
                                         });
    auto avg_dur = std::accumulate(v.begin(), v.end(), 0.f,
                                   [&] (float s, const CpuMemEntry& e) {
                                       return s + e.cost_time;
                                   }) / TEST_CNT;
    WLog::i(tag + " COST_TIME max: " + std::to_string(max_dur_iter->cost_time)  +
            ", min: " + std::to_string(min_dur_iter->cost_time) +
            ", avg: " + std::to_string(avg_dur));
    std::string res;
    res = tag + ", " + std::to_string(avg_dur) + ", "
            + std::to_string(avg_cpu * CORE_NUM) + ", "
            + std::to_string(max_cpu_iter->cpu * CORE_NUM) + ", "
            + std::to_string(min_cpu_iter->cpu * CORE_NUM) + ", "
            + std::to_string(avg_mem) + ", "
            + std::to_string(max_cpu_iter->mem) + ", "
            + std::to_string(min_cpu_iter->mem);
    return res;
}

std::string test_single_dmips(const char* img_data) {
    std::string result;

    auto* req = serv.make_request();
    auto* res = serv.make_result();
    req->clear_all();
    res->clear_all();
    req->_width = 1280;
    req->_height = 720;
    req->_frame = (unsigned char*)img_data;

    // 预检测，为了测试单帧性能，先检测人脸关键点和手势关键点
    set_predetect_switch();
    serv.detect(req, res);
    g_progress = 0;
    g_progress_tag = "single abilities";

    int ability_cnt = static_cast<int>(_ability_list.size());
    int cnt = 0;
    for (auto& ability : _ability_list) {
        WLog::i(" ");
        WLog::i("=== Test DMIPS ability: " + ability.second + " ===");
        req->set_specific_ability(ability.first);
        auto start_total_cpu = DmipsUtil::get_total_cpu_occupy();
        auto start_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());
        auto start = high_resolution_clock::now();
//        WLog::i("START total: " + std::to_string(start_total_cpu) + ", proc: " + std::to_string(start_proc_cpu));

        for (int i = 0; i < TEST_CNT; ++i) {
            serv.detect(req, res);
//            if (i % 100 == 0) {
//                WLog::i(ability.second + " progress: " + std::to_string(i) + " of 500");
//            }
            g_progress = (i * 100 / TEST_CNT  +  cnt * 100) / ability_cnt;
        }

        auto end = high_resolution_clock::now();
        auto end_total_cpu = DmipsUtil::get_total_cpu_occupy();
        auto end_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());
//        WLog::i("END total: " + std::to_string(end_total_cpu) + ", proc: " + std::to_string(end_proc_cpu));

        auto occu = DmipsUtil::get_cpu_oocupancy(start_total_cpu, end_total_cpu, start_proc_cpu, end_proc_cpu);
        auto dur = duration_cast<milliseconds>(end - start).count() / 1000;
        auto dmips = occu * dur * DMIPS_COEFF / TEST_CNT;
        WLog::i("DMIPS RESULT: " + std::to_string(dmips) + "\n");
        result += ability.second + ", " + std::to_string(dmips) + "\n";
        cnt++;
    }

    serv.recycle_request(req);
    serv.recycle_result(res);
    

    return result;
}

std::string test_mixed_dmips(const char* img_data, MixAbilityType type) {
    std::string result;

    auto* req = serv.make_request();
    auto* res = serv.make_result();
    req->clear_all();
    res->clear_all();
    req->_width = 1280;
    req->_height = 720;
    req->_frame = (unsigned char*)img_data;

    std::vector<AbilityId> ability_list;
    std::string mix_type_name;
    if (type == MixAbilityType::FACE_ID) {
        ability_list = _faceid_ability_list;
        mix_type_name = "FaceID";
    } else if (type == MixAbilityType::DMS) {
        ability_list = _dms_ability_list;
        mix_type_name = "DMS";
    } else if (type == MixAbilityType::MMI) {
        ability_list = _mmi_ability_list;
        mix_type_name = "MMI";
    } else if (type == MixAbilityType::DMS_MMI) {
        ability_list = _dms_mmi_ability_list;
        mix_type_name = "DMS_MMI";
    } else if (type == MixAbilityType::TOTAL) {
        ability_list = _total_ability_list;
        mix_type_name = "TOTAL";
    }

    clear_switch();
    g_progress = 0;
    g_progress_tag = mix_type_name;
    for (auto& ability : ability_list) {
        serv.set_switch(ability, true);
    }

    auto start_total_cpu = DmipsUtil::get_total_cpu_occupy();
    auto start_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());
    auto start = high_resolution_clock::now();
//    WLog::i("START total: " + std::to_string(start_total_cpu) + ", proc: " + std::to_string(start_proc_cpu));

    for (int i = 0; i < TEST_CNT; ++i) {
        serv.detect(req, res);
//        if (i % 100 == 0) {
//            WLog::i(mix_type_name + " progress: " + std::to_string(i) + " of 500");
//        }
        g_progress = i * 100 / TEST_CNT;
    }

    auto end = high_resolution_clock::now();
    auto end_total_cpu = DmipsUtil::get_total_cpu_occupy();
    auto end_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());
//    WLog::i("END total: " + std::to_string(end_total_cpu) + ", proc: " + std::to_string(end_proc_cpu));

    auto occu = DmipsUtil::get_cpu_oocupancy(start_total_cpu, end_total_cpu, start_proc_cpu, end_proc_cpu);
    auto dur = duration_cast<milliseconds>(end - start).count() / 1000;
    auto dmips = occu * dur * DMIPS_COEFF / TEST_CNT;
    WLog::i("DMIPS RESULT: " + std::to_string(dmips) + "\n");
    result = mix_type_name + ", " + std::to_string(dmips) + "\n";

    serv.recycle_request(req);
    serv.recycle_result(res);

    return result;
}

std::string test_single_cpumem(const char* img_data, int fps) {
    std::string result;

    auto* req = serv.make_request();
    auto* res = serv.make_result();
    req->clear_all();
    res->clear_all();
    req->_width = 1280;
    req->_height = 720;
    req->_frame = (unsigned char*)img_data;

    // 预检测，为了测试单帧性能，先检测人脸关键点和手势关键点
    set_predetect_switch();
    serv.detect(req, res);
    g_progress = 0;
    g_progress_tag = "single abilities";

    std::vector<CpuMemEntry> cpu_mem_list(TEST_CNT);
    int interval = 1000 / fps;

    int ability_cnt = static_cast<int>(_ability_list.size());
    int cnt = 0;
    for (auto& ability : _ability_list) {
        WLog::i(" ");
        WLog::i("=== Test CPUMEM ability: " + ability.second + " ===");
        req->set_specific_ability(ability.first);

        for (int i = 0; i < TEST_CNT; ++i) {
            auto start_total_cpu = DmipsUtil::get_total_cpu_occupy();
            auto start_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());
            auto start = high_resolution_clock::now();

            serv.detect(req, res);

//            if (i % 100 == 0) {
//                WLog::i(ability.second + " progress: " + std::to_string(i) + " of 500");
//            }
            g_progress = (i * 100 / TEST_CNT  +  cnt * 100) / ability_cnt;

            auto end = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(end - start).count();

            // 控制帧率
            std::this_thread::sleep_for(milliseconds(interval - dur));

            auto end_total_cpu = DmipsUtil::get_total_cpu_occupy();
            auto end_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());

            auto occu = DmipsUtil::get_cpu_oocupancy(start_total_cpu, end_total_cpu, start_proc_cpu, end_proc_cpu);
            auto mem = MemUtil::get_proc_mem(getpid());
            cpu_mem_list[i] = CpuMemEntry {occu, (float)mem / 1000.f, (float)dur};
        }
        auto ret = calculate_cpu_mem_result(cpu_mem_list, ability.second);
        result += ret + "\n";
        cnt++;
    }

    serv.recycle_request(req);
    serv.recycle_result(res);

    return result;
}

std::string test_mixed_cpumem(const char* img_data, MixAbilityType type, int fps) {
    std::string result;

    auto* req = serv.make_request();
    auto* res = serv.make_result();
    req->clear_all();
    res->clear_all();
    req->_width = 1280;
    req->_height = 720;
    req->_frame = (unsigned char*)img_data;

    std::vector<AbilityId> ability_list;
    std::string mix_type_name;
    if (type == MixAbilityType::FACE_ID) {
        ability_list = _faceid_ability_list;
        mix_type_name = "FaceID";
    } else if (type == MixAbilityType::DMS) {
        ability_list = _dms_ability_list;
        mix_type_name = "DMS";
    } else if (type == MixAbilityType::MMI) {
        ability_list = _mmi_ability_list;
        mix_type_name = "MMI";
    } else if (type == MixAbilityType::DMS_MMI) {
        ability_list = _dms_mmi_ability_list;
        mix_type_name = "DMS_MMI";
    } else if (type == MixAbilityType::TOTAL) {
        ability_list = _total_ability_list;
        mix_type_name = "TOTAL";
    }

    clear_switch();
    g_progress = 0;
    g_progress_tag = mix_type_name;
    for (auto& ability : ability_list) {
        serv.set_switch(ability, true);
    }

    std::vector<CpuMemEntry> cpu_mem_list(TEST_CNT);
    int interval = 1000 / fps;

    for (int i = 0; i < TEST_CNT; ++i) {
        auto start_total_cpu = DmipsUtil::get_total_cpu_occupy();
        auto start_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());
        auto start = high_resolution_clock::now();

        serv.detect(req, res);
//        if (i % 100 == 0) {
//            WLog::i(mix_type_name + " progress: " + std::to_string(i) + " of 500");
//        }
        g_progress = i * 100 / TEST_CNT;

        auto end = high_resolution_clock::now();
        auto dur = duration_cast<milliseconds>(end - start).count();
        // 控制帧率
        std::this_thread::sleep_for(milliseconds(interval - dur));

        auto end_total_cpu = DmipsUtil::get_total_cpu_occupy();
        auto end_proc_cpu = DmipsUtil::get_proc_cpu_occupy(getpid());

        auto occu = DmipsUtil::get_cpu_oocupancy(start_total_cpu, end_total_cpu, start_proc_cpu, end_proc_cpu);
        auto mem = MemUtil::get_proc_mem(getpid());
        cpu_mem_list[i] = CpuMemEntry {occu, (float)mem / 1000.f, (float)dur};
    }
    result = calculate_cpu_mem_result(cpu_mem_list, mix_type_name) + "\n";

    serv.recycle_request(req);
    serv.recycle_result(res);

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_baidu_admips_MainActivity_testDmips(
        JNIEnv* env,
        jobject /* this */) {
    std::string info = "ability, dmips\n";
    auto mem = MemUtil::get_proc_mem(getpid());
    WLog::i("mem size0: " + std::to_string(mem));
    init();
    if (img_buffer) {
        if (_test_switch["Single"]) {
            info += test_single_dmips(img_buffer); // 单项能力测试
        }
        if (_test_switch["FaceID"]) {
            info += test_mixed_dmips(img_buffer, MixAbilityType::FACE_ID); // FaceID 测试
        }
        if (_test_switch["DMS"]) {
            info += test_mixed_dmips(img_buffer, MixAbilityType::DMS); // DMS 测试
        }
        if (_test_switch["MMI"]) {
            info += test_mixed_dmips(img_buffer, MixAbilityType::MMI); // MMI 测试
        }
        if (_test_switch["DMS_MMI"]) {
            info += test_mixed_dmips(img_buffer, MixAbilityType::DMS_MMI); // DMS + MMI测试
        }
        if (_test_switch["TOTAL"]) {
            info += test_mixed_dmips(img_buffer, MixAbilityType::TOTAL);
        }
        write_dmips_result(info);
    } else {
        WLog::i("testDmips failed: image data error!" );
    }
    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_baidu_admips_MainActivity_testCpuMem(
        JNIEnv* env,
        jobject /* this */) {
    std::string info = "ability, cost_time/ms, avg_cpu, max_cpu, min_cpu, avg_mem, max_mem, min_mem\n";
    init();
    if (img_buffer) {
        if (_test_switch["Single"]) {
            info += test_single_cpumem(img_buffer, TEST_FPS); // 单项能力测试
        }
        if (_test_switch["FaceID"]) {
            info += test_mixed_cpumem(img_buffer, MixAbilityType::FACE_ID, TEST_FACEID_FPS);
        }
        if (_test_switch["DMS"]) {
            info += test_mixed_cpumem(img_buffer, MixAbilityType::DMS, TEST_FPS);
        }
        if (_test_switch["MMI"]) {
            info += test_mixed_cpumem(img_buffer, MixAbilityType::MMI, TEST_FPS);
        }
        if (_test_switch["DMS_MMI"]) {
            info += test_mixed_cpumem(img_buffer, MixAbilityType::DMS_MMI, TEST_FPS);
        }
        if (_test_switch["TOTAL"]) {
            info += test_mixed_cpumem(img_buffer, MixAbilityType::TOTAL, TEST_FPS);
        }
        write_cpu_mem_result(info);
    } else {
        WLog::i("test CpuMem failed: image data error!" );
    }

    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_baidu_admips_MainActivity_readImage(JNIEnv *env, jobject, jobject assetManager) {
    auto* asset_manager = AAssetManager_fromJava(env, assetManager);
    auto* asset = AAssetManager_open(asset_manager, "test_image_yuv.bin", AASSET_MODE_BUFFER);
    img_buffer = (char *) AAsset_getBuffer(asset);
    auto buf_len = static_cast<int>(AAsset_getLength(asset));
    WLog::i("image buf_len=" + std::to_string(buf_len));
    if (img_buffer == nullptr) {
        WLog::i("read image error!");
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_baidu_admips_MainActivity_getProgress(
        JNIEnv* env,
        jobject /* this */) {
    std::string progress = g_progress_tag + "... " + std::to_string(g_progress);
    return env->NewStringUTF(progress.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_baidu_admips_MainActivity_setEnv(JNIEnv* env, jobject /* this */, jstring envName,
        jstring envPath){
    const char *env_name = env->GetStringUTFChars(envName, 0);
    const char *env_path = env->GetStringUTFChars(envPath, 0);
    WLog::i("set_env:");
    WLog::i(env_name);
    WLog::i(env_path);
    return setenv(env_name, env_path, 1 /*override*/) == 0;
}