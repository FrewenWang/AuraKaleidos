//
// LightBuffer compile-time error tracking.
//
#include "lightbuffer/error.h"

#include <iostream>

namespace aura::light_buffer {

ErrCode Error::errCode = ErrCode::OK;

void Error::log(ErrCode code, const std::string &errmsg, LineInfo *li) {
    errCode = code;
    std::cerr << "[LightBuffer] error " << static_cast<int>(code) << ": " << errmsg;
    if (li != nullptr) {
        std::cerr << " (line " << li->line << ")";
    }
    std::cerr << std::endl;
}

ErrCode Error::getLastError() {
    return errCode;
}

void Error::clearError() {
    errCode = ErrCode::OK;
}

} // namespace aura::light_buffer
