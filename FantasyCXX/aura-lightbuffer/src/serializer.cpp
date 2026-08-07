//
// LightBuffer binary (de)serializer implementation.
//
#include "plainbuffer/serializer.h"

#include <cstring>

namespace plainbuffer {

// ----------------------------------------------------------------------------
// Serializer
// ----------------------------------------------------------------------------

bool Serializer::serialize_char(char v, char *data, int size, int &offset) {
    return serialize_primary_element<char>(v, data, size, offset);
}

bool Serializer::serialize_short(short v, char *data, int size, int &offset) {
    return serialize_primary_element<short>(v, data, size, offset);
}

bool Serializer::serialize_int(int v, char *data, int size, int &offset) {
    return serialize_primary_element<int>(v, data, size, offset);
}

bool Serializer::serialize_int64_t(int64_t v, char *data, int size, int &offset) {
    return serialize_primary_element<int64_t>(v, data, size, offset);
}

bool Serializer::serialize_float(float v, char *data, int size, int &offset) {
    return serialize_primary_element<float>(v, data, size, offset);
}

bool Serializer::serialize_double(double v, char *data, int size, int &offset) {
    return serialize_primary_element<double>(v, data, size, offset);
}

bool Serializer::serialize_bool(bool v, char *data, int size, int &offset) {
    return serialize_primary_element<bool>(v, data, size, offset);
}

bool Serializer::serialize_string(const std::string &s, char *data, int size, int &offset) {
    int len = static_cast<int>(s.size());
    if (!serialize_primary_element<int>(len, data, size, offset)) {
        return false;
    }
    if (offset + len > size) {
        return false;
    }
    if (len > 0) {
        std::memcpy(data + offset, s.data(), static_cast<size_t>(len));
    }
    offset += len;
    return true;
}

bool Serializer::serialize_bytes(const char *bytes, size_t bytes_len, char *data, int size, int &offset) {
    int len = static_cast<int>(bytes_len);
    if (!serialize_primary_element<int>(len, data, size, offset)) {
        return false;
    }
    if (offset + len > size) {
        return false;
    }
    if (len > 0 && bytes != nullptr) {
        std::memcpy(data + offset, bytes, static_cast<size_t>(len));
    }
    offset += len;
    return true;
}

bool Serializer::serialize_vector_bytes(const std::vector<const char *> &v,
                                        const std::vector<size_t> &v_len,
                                        char *data, int size, int &offset) {
    int v_size = static_cast<int>(v.size());
    if (!serialize_primary_element<int>(v_size, data, size, offset)) {
        return false;
    }
    for (int i = 0; i < v_size; ++i) {
        if (!serialize_bytes(v[i], v_len[i], data, size, offset)) {
            return false;
        }
    }
    return true;
}

bool Serializer::serialize_message(const aura::light_buffer::MessageBase &msg, char *data, int size, int &offset) {
    int need = static_cast<int>(msg.ByteSizeLong());
    if (offset + need > size) {
        return false;
    }
    if (!msg.SerializeToArray(data + offset, need)) {
        return false;
    }
    offset += need;
    return true;
}

// ----------------------------------------------------------------------------
// Deserializer
// ----------------------------------------------------------------------------

bool Deserializer::deserialize_char(char &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<char>(v, data, size, offset);
}

bool Deserializer::deserialize_short(short &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<short>(v, data, size, offset);
}

bool Deserializer::deserialize_int(int &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<int>(v, data, size, offset);
}

bool Deserializer::deserialize_int64_t(int64_t &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<int64_t>(v, data, size, offset);
}

bool Deserializer::deserialize_float(float &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<float>(v, data, size, offset);
}

bool Deserializer::deserialize_double(double &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<double>(v, data, size, offset);
}

bool Deserializer::deserialize_bool(bool &v, const char *data, int size, int &offset) {
    return deserialize_primary_element<bool>(v, data, size, offset);
}

bool Deserializer::deserialize_string(std::string &s, const char *data, int size, int &offset) {
    int len = 0;
    if (!deserialize_primary_element<int>(len, data, size, offset)) {
        return false;
    }
    if (len < 0 || offset + len > size) {
        return false;
    }
    s.assign(data + offset, static_cast<size_t>(len));
    offset += len;
    return true;
}

bool Deserializer::deserialize_bytes(const char *&bytes, size_t &bytes_len, const char *data, int size, int &offset) {
    int len = 0;
    if (!deserialize_primary_element<int>(len, data, size, offset)) {
        return false;
    }
    if (len < 0 || offset + len > size) {
        return false;
    }
    bytes = data + offset;
    bytes_len = static_cast<size_t>(len);
    offset += len;
    return true;
}

bool Deserializer::deserialize_vector_bytes(std::vector<const char *> &v, std::vector<size_t> &v_len,
                                            const char *data, int size, int &offset) {
    int v_size = 0;
    if (!deserialize_primary_element<int>(v_size, data, size, offset)) {
        return false;
    }
    v.clear();
    v.resize(v_size);
    v_len.clear();
    v_len.resize(v_size);
    for (int i = 0; i < v_size; ++i) {
        const char *ptr = nullptr;
        size_t len = 0;
        if (!deserialize_bytes(ptr, len, data, size, offset)) {
            return false;
        }
        v[i] = ptr;
        v_len[i] = len;
    }
    return true;
}

bool Deserializer::deserialize_message(aura::light_buffer::MessageBase &msg, const char *data, int size, int &offset) {
    int need = static_cast<int>(msg.ByteSizeLong());
    if (offset + need > size) {
        return false;
    }
    if (!msg.ParseFromArray(data + offset, need)) {
        return false;
    }
    offset += need;
    return true;
}

} // namespace plainbuffer
