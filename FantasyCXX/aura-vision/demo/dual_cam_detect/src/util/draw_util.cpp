#include "draw_util.h"

#include <cstdlib>

static std::vector<std::string> attention_labels {
        "",
        "UnFocused",
        "Uncoordinated",
        "Forward",
        "Left",
        "Right",
        "Up",
        "Down"
};

static std::vector<std::string> fatigue_labels {
        "Normal",
        "Yawn",
        "EyeClose",
        "Yawn+EyeClose"
};

static std::vector<std::string> eye_blink_labels {
        "Unknow",
        "YES",
        "NO"
};

static std::vector<std::string> danger_labels {
        "",
        "Normal",
        "Smoke",
        "Silence",
        "Drink",
        "OpenMouth",
        "Eat"
};

static std::vector<std::string> nodshake_labels {
        "None",
        "Shake",
        "Nod",
        "Go_on"
};

static std::vector<std::string> call_labels {
        "Normal",
        "Call"
};

static std::vector<std::string> emotion_labels {
        "Normal",
        "Like",
        "Dislike",
        "Surprised"
};

static std::vector<std::string> glass_labels {
        "Sunglasses",
        "NoGlasses",
        "Glasses"
};

static std::vector<std::string> gender_labels {
        "Male",
        "Female"
};

static std::vector<std::string> race_labels {
        "Black",
        "White",
        "Yellow"
};

static std::vector<std::string> age_labels {
        "Baby",
        "Teenager",
        "Youth",
        "Midlife",
        "Senior"
};

static std::vector<std::string> gesture_static_labels {
        "none",
        "good",
        "ok",
        "dislike",
        "fist",
        "1",
        "2",
        "3",
        "4",
        "5",
        "previous",
        "next",
        "left5",
        "right5",
        "heart",
        "rock"
};

static std::vector<std::string> gesture_dynamic_labels {
        "none",
        "wave",
        "flip",
        "pinch",
        "grasp",
        "left_wave",
        "right_wave"
};

static std::vector<std::string> cover_labels {
        "no",
        "cover"
};

static std::vector<std::string> track_labels {
        "unknown",
        "init",
        "tracking",
        "miss",
};

cv::Mat& DrawUtil::drawline(cv::Mat &image, const GestureInfo *gesture, const cv::Scalar &rect_scalar, int offset) {
#ifdef OPENCV2
    line(image, cv::Point(gesture->_landmark_21[0].x, gesture->_landmark_21[0].y),
         cv::Point(gesture->_landmark_21[1 + offset].x, gesture->_landmark_21[1 + offset].y), rect_scalar, 2);
    line(image, cv::Point(gesture->_landmark_21[1 + offset].x, gesture->_landmark_21[1 + offset].y),
         cv::Point(gesture->_landmark_21[2 + offset].x, gesture->_landmark_21[2 + offset].y), rect_scalar, 2);
    line(image, cv::Point(gesture->_landmark_21[2 + offset].x, gesture->_landmark_21[2 + offset].y),
         cv::Point(gesture->_landmark_21[3 + offset].x, gesture->_landmark_21[3 + offset].y), rect_scalar, 2);
    line(image, cv::Point(gesture->_landmark_21[3 + offset].x, gesture->_landmark_21[3 + offset].y),
         cv::Point(gesture->_landmark_21[4 + offset].x, gesture->_landmark_21[4 + offset].y), rect_scalar, 2);
#endif
#ifdef OPENCV4
#endif
    return image;
}

void DrawUtil::draw(cv::Mat &image, VisionResult* result, int id) {
    static cv::Scalar landmark_scalar(255, 0, 0);
    static cv::Scalar rect_scalar(0, 255, 255);
    static cv::Scalar eye_circle(0, 255, 0);
    static cv::Scalar eye_circle_cmp(0, 0, 255);
    static cv::Scalar face_result_color(0, 255, 0);

    auto* face_result = result->get_face_result();
    auto* gest_result = result->get_gesture_result();

    int x_pos = 50;
    int offset = 40;
    double font_scale = 0.6;
    int font_thick = 2;
    if (!face_result->no_face()) {
        for (int i = 0; i < face_result->_face_count; i++) {
            std::string face_info_text = "";
            FaceInfo *face = face_result->_face_infos[i];
            if (face->_id <= 0) {
                continue;
            }
            int y_pos = 180;
            x_pos += 350 * i;

            // 绘制人脸框
            float left_up_x = face->_rect_lt.x;
            float left_up_y = face->_rect_lt.y;
            int w = face->_rect_rb.x - left_up_x;
            int h = face->_rect_rb.y - left_up_y;
            cv::Rect r(left_up_x, left_up_y, w, h);
            cv::rectangle(image, r, rect_scalar, 3);
            cv::putText(image, std::to_string(static_cast<int>(face->_id)), cv::Point(left_up_x, left_up_y), cv::FONT_HERSHEY_SIMPLEX, 2, rect_scalar, 4, 8);

            // 绘制人脸关键点
            cv::Point point;
            for (auto &lmk : face->_landmark_2d_106) {
                point.x = lmk.x;
                point.y = lmk.y;
                cv::circle(image, point, 3, landmark_scalar, -1);
            }
        } 
    }

    if (!gest_result->no_gesture()) {
        GestureInfo *gesture = gest_result->_gesture_infos[0];
        cv::Scalar rect_scalar(0, 255, 255);
        int w = gesture->_rect_rb.x - gesture->_rect_lt.x;
        int h = gesture->_rect_rb.y - gesture->_rect_lt.y;
        cv::Rect r(gesture->_rect_lt.x, gesture->_rect_lt.y, w, h);
        cv::rectangle(image, r, rect_scalar, 3);

        cv::Point point;
        for (auto & lmk : gesture->_landmark_21) {
            point.x = lmk.x;
            point.y = lmk.y;
            cv::circle(image, point, 3, landmark_scalar, -1);
        }

        image = drawline(image, gesture, rect_scalar, 0);
        image = drawline(image, gesture, rect_scalar, 4);
        image = drawline(image, gesture, rect_scalar, 8);
        image = drawline(image, gesture, rect_scalar, 12);
        image = drawline(image, gesture, rect_scalar, 16);

        std::string result = " ";
        int type = static_cast<int>(gesture->_static_type);
        type = std::max(0, type);
        result = gesture_static_labels[type];
        cv::putText(image, result, cv::Point(100, 50), cv::FONT_HERSHEY_SIMPLEX, 1, eye_circle, 4, 8);

        int dy_type = static_cast<int>(gesture->_dynamic_type);
        dy_type = std::max(0, dy_type);
        result = gesture_dynamic_labels[dy_type];
        cv::putText(image, result, cv::Point(200, 50), cv::FONT_HERSHEY_SIMPLEX, 1, eye_circle, 4, 8);
    }

    std::stringstream name;
    name << "camera" << id;
    imshow(name.str(), image);
}  