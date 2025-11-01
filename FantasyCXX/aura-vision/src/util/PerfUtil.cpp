//
// Created by Li,Wendong on 2019-01-13.
//

#include "vision/util/PerfUtil.h"
#include "vision/util/log.h"
#include "util/SystemClock.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace aura::vision {

using namespace std;


const std::string PerfUtil::TAG_TOTAL = "[TotalFrame]";

int PerfUtil::qnnLoopModel = 0;
std::mutex PerfUtil::sMutexGlobal;
std::map<string, PerfUtil *> PerfUtil::sGlobals;

PerfUtil * PerfUtil::global(int tag) {
    lock_guard<mutex> lock(sMutexGlobal);
    string t;
    if (tag == -1) {
#ifndef ANDROID
        char pname[16];
        pthread_getname_np(pthread_self(), pname, 16);
        t = pname;
#endif
    } else {
        t = to_string(tag);
    }
    PerfUtil *pu = sGlobals[t];
    if (pu == nullptr) {
        pu = new PerfUtil(t);
        sGlobals[t] = pu;
    }
    return pu;
}

void PerfUtil::globalClear(int tag) {
    lock_guard<mutex> lock(sMutexGlobal);
    if (tag == -1) {
        for (auto g : sGlobals) {
            if (!g.second->mIsLogging) {
                g.second->clear();
            }
        }
    } else {
        string t = to_string(tag);
        PerfUtil *pu = sGlobals[t];
        if (pu != nullptr) {
            pu->clear();
        }
    }
}

void PerfUtil::globalPrint(int tag) {
    lock_guard<mutex> lock(sMutexGlobal);
    if (tag == -1) {
        for (auto g : sGlobals) {
            if (!g.second->mIsLogging) {
                g.second->printRecords();
            }
        }
    } else {
        string t = to_string(tag);
        PerfUtil *pu = sGlobals[t];
        if (pu != nullptr) {
            pu->printRecords();
        }
    }
}

PerfUtil::PerfUtil(int &tag) {
    mTag = to_string(tag);
    mPrintLoop = 1;
    mCurLoop = 0;
    clear();
}

PerfUtil::PerfUtil(string &tag) {
    mTag = tag;
    mPrintLoop = 1;
    mCurLoop = 0;
    clear();
}

void PerfUtil::setLogging(bool log) {
    lock_guard<mutex> lock(sMutexGlobal);
    mIsLogging = log;
}

bool PerfUtil::isLogging() {
    lock_guard<mutex> lock(sMutexGlobal);
    return mIsLogging;
}

void PerfUtil::tick(const std::string &tag) {
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto& it : mRecords) {
        if (get<0>(it) == tag) {
            get<1>(it) = SystemClock::nowMillis() - get<1>(it);
            get<2>(it) ++;
            return;
        }
    }
    mRecords.emplace_back(tag, SystemClock::nowMillis(), 1);
}

void PerfUtil::tick(int tagInt) {
    std::lock_guard<std::mutex> lock(mMutex);
    string tag = std::to_string(tagInt);
    mRecords.emplace_back(tag, SystemClock::nowMillis(), 1);
}

void PerfUtil::tock(const std::string &tag) {
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto& it : mRecords) {
        if (get<0>(it) == tag) {
            get<1>(it) = SystemClock::nowMillis() - get<1>(it);
            break;
        }
    }
}
void PerfUtil::tock(int tagInt) {
    std::lock_guard<std::mutex> lock(mMutex);
    string tag = std::to_string(tagInt);
    for (auto& it : mRecords) {
        if (get<0>(it) == tag) {
            get<1>(it) = SystemClock::nowMillis() - get<1>(it);
            break;
        }
    }
}

void PerfUtil::clear() {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mCurLoop == 0) {
        mRecords.clear();
    }
}

vector<tuple<string, uint64_t, int >> &PerfUtil::get_records() {
    std::lock_guard<std::mutex> lock(mMutex);
    return mRecords;
}

std::int64_t PerfUtil::get_record(const std::string& tag) {
    std::lock_guard<std::mutex> lock(mMutex);
    std::int64_t time = -1;
//    auto it = _records.find(tag);
//    if (it != _records.end()) {
//        time = _records[tag];
//    }
    for (auto it : mRecords) {
        if (get<0>(it) == tag) {
            time = get<1>(it);
            break;
        }
    }
    return time;
}

std::uint64_t PerfUtil::getTotalTime() {
    std::lock_guard<std::mutex> lock(mMutex);
    std::uint64_t time = 0;
//    for (auto it : _records) {
//        time += it.second;
//    }
    for (auto it : mRecords) {
        time += get<1>(it);
    }
    return time;
}

void PerfUtil::setPrintLoop(int loop) {
    mPrintLoop = loop;
}

int PerfUtil::getPrintLoop() {
    return mPrintLoop;
}

void PerfUtil::printDetectRecords() {
    if (++mCurLoop != mPrintLoop) {
        return;
    }
    mCurLoop = 0;

    std::lock_guard<std::mutex> lock(mMutex);
    int64_t sum = 0;
    int64_t sumPredict = 0;
    int64_t sumNcnn = 0;
    int64_t sumQnn = 0;
    int64_t sumDetectPrepare = 0;
    int64_t sumDetectProcess = 0;
    int64_t sumDetectPost = 0;
    int mgrCounter = 0;

    string str("[PerfAD][");
    str.append(mTag).append("]: ============== Ability Detect Performance (ms) ==============\n");
    for (auto it : mRecords) {
        string tag = get<0>(it);
        uint64_t interval = get<1>(it);
        if (tag.at(0) != '[') {
            continue;
        }

        if (tag == TAG_TOTAL) {
            sum = interval;
            continue;
        }

        if (tag.find("Qnn") != string::npos) {
            sumQnn += interval;
        } else if (tag.find("Ncnn") != string::npos) {
            sumNcnn += interval;
        }

        if (tag.find("Detector-pre") != string::npos) {
            sumDetectPrepare += interval;
        }

        if (tag.find("Detector-pro") != string::npos) {
            sumDetectProcess += interval;
        }

        if (tag.find("Detector-pos") != string::npos) {
            sumDetectPost += interval;
        }

        int wid = 44;
        int wid2 = 0;
        str.append("[PerfAD][").append(mTag).append("]: ");
        if (tag.at(1) == 'D') {
            str.append("    ");
            wid -= 4;
            wid2 = 4;
        } else if (tag.at(1) == 'P') {
            str.append("        ");
            wid -= 8;
            wid2 = 8;
        } else if (tag.at(1) == 'M') {
            mgrCounter++;
        }
//        cout << setw(wid) << left << it.first << " : " << setw(wid2) <<  left << "" << (it.second / mPrintLoop) << endl;
        str.append(tag);
        wid = wid - tag.size();
        if (wid > 0) {
            for (int i = 0; i < wid; i++) {
                str.append(" ");
            }
        }
        str.append(" : ");
        for (int i = 0; i < wid2; i++) {
            str.append(" ");
        }
        str.append(to_string(interval / mPrintLoop)).append("\n");
    }
    sum /= mPrintLoop;
    sumPredict = (sumQnn + sumNcnn) / mPrintLoop;
    sumQnn /= mPrintLoop;
    sumNcnn /= mPrintLoop;

    sumDetectPrepare /= mPrintLoop;
    sumDetectProcess /= mPrintLoop;
    sumDetectPost /= mPrintLoop;

    str.append("[PerfAD][").append(mTag).append("]: ")
       .append("[Sum : ").append(to_string(sum)).append("] ")
       .append("[SumDetectPre : ").append(to_string(sumDetectPrepare)).append("] ")
       .append("[SumDetectPro : ").append(to_string(sumDetectProcess)).append("] ")
       .append("[SumDetectPost : ").append(to_string(sumDetectPost)).append("] ")
       .append("[SumPredict : ").append(to_string(sumPredict)).append("] ")
       .append("[SumQnn : ").append(to_string(sumQnn)).append("] ")
       .append("[Manager Count : ").append(to_string(mgrCounter)).append("]")
       .append("\n");
    cout << str;
    //    VLOGD("VisionPerf", str.c_str());
}

void PerfUtil::printRecords() {
    stringstream ss;
    ss << "\n[PerfAR][" << mTag << "]: ------ Ability Performances (ms) ------" << endl;
    for (auto it : mRecords) {
        ss << "[PerfAR][" << mTag << "][" << setw(35) << left << get<0>(it) << "] : [" <<
           setw(3) << left << get<1>(it) << "] ms/" <<
           setw(3) << left << get<2>(it) << " cns - [" <<
           setw(3) << left << get<1>(it) / get<2>(it) << "] ms/avg " << endl;
    }
    ss << "\n";
    cout << ss.str();
//    VLOGD("VisionPerf", ss.str().c_str());
}

PerfAuto::PerfAuto(PerfUtil* perf, const std::string& tag)
    : _perf(perf), _tag(tag), _duration(0){
#ifndef ENABLE_PERF
    return;
#endif
    if (_perf) {
        _perf->tick(_tag);
    }
}

PerfAuto::PerfAuto(double& duration)
        : _perf(nullptr), _duration(&duration){
#ifndef ENABLE_PERF
    return;
#endif
    _start = clock();
}

PerfAuto::~PerfAuto() {
#ifndef ENABLE_PERF
    return;
#endif
    if (_perf) {
        _perf->tock(_tag);
//        VLOGD("AutoPerf", "[%s] costs %ld ms", _tag.c_str(), _perf->get_record(_tag));
    }
    if (_duration) {
        *_duration = (double)(clock() - _start) / CLOCKS_PER_SEC * 1000;
    }
}

} // namespace aura::vision
