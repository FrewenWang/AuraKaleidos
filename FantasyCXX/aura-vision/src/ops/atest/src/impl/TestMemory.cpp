#include <string>
#include "TestMemory.h"
#include "ops/op.h"
#include "OpsTestUtil.h"

using namespace std;

namespace aura::vision {
namespace op {

static const char* TAG = "TestMemory";
static std::string test_img_2560x1440 = "./res/2560x1440.jpeg";

vector<double> TestMemory::testMemcpy() {
    cv::Mat matSrc = cv::imread(test_img_2560x1440);
    size_t size = matSrc.total();
    unsigned char *matDst = (unsigned char *) malloc(size);
    long duration = 0;
    {
        TIME_PERF(duration);
        op::memcpy(matDst, matSrc.data, size);
    }

    return vector<double>{};
}


} // namespace op
} // namespace aura::vision