//
// Template implementations for the LightBuffer (de)serializer.
// Included at the end of serializer.h.
//
#pragma once

namespace plainbuffer {

template<typename T>
bool Serializer::serialize_primary_element(T v, char *data, int size, int &offset) {
    int data_len = static_cast<int>(sizeof(T));
    if (offset + data_len > size) {
        return false;
    }
    std::memcpy(data + offset, &v, static_cast<size_t>(data_len));
    offset += data_len;
    return true;
}

template<typename T>
bool Serializer::serialize_vector(const std::vector<T> &v, char *data, int size, int &offset,
                                  typename std::enable_if<std::is_base_of<aura::light_buffer::MessageBase, T>::value, int>::type) {
    int v_size = static_cast<int>(v.size());
    if (!serialize_primary_element<int>(v_size, data, size, offset)) {
        return false;
    }
    for (int i = 0; i < v_size; ++i) {
        if (!serialize_message(v[i], data, size, offset)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Serializer::serialize_vector(const std::vector<T> &v, char *data, int size, int &offset,
                                  typename std::enable_if<!std::is_base_of<aura::light_buffer::MessageBase, T>::value, int>::type) {
    int v_size = static_cast<int>(v.size());
    if (!serialize_primary_element<int>(v_size, data, size, offset)) {
        return false;
    }
    for (int i = 0; i < v_size; ++i) {
        if (!serialize_primary_element<T>(v[i], data, size, offset)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Deserializer::deserialize_primary_element(T &v, const char *data, int size, int &offset) {
    int data_len = static_cast<int>(sizeof(T));
    if (offset + data_len > size) {
        return false;
    }
    std::memcpy(&v, data + offset, static_cast<size_t>(data_len));
    offset += data_len;
    return true;
}

template<typename T>
bool Deserializer::deserialize_vector(std::vector<T> &v, const char *data, int size, int &offset,
                                      typename std::enable_if<std::is_base_of<aura::light_buffer::MessageBase, T>::value, int>::type) {
    int v_size = 0;
    if (!deserialize_primary_element<int>(v_size, data, size, offset)) {
        return false;
    }
    v.clear();
    v.resize(v_size);
    for (int i = 0; i < v_size; ++i) {
        if (!deserialize_message(v[i], data, size, offset)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Deserializer::deserialize_vector(std::vector<T> &v, const char *data, int size, int &offset,
                                      typename std::enable_if<!std::is_base_of<aura::light_buffer::MessageBase, T>::value, int>::type) {
    int v_size = 0;
    if (!deserialize_primary_element<int>(v_size, data, size, offset)) {
        return false;
    }
    v.clear();
    v.resize(v_size);
    for (int i = 0; i < v_size; ++i) {
        T value;
        if (!deserialize_primary_element<T>(value, data, size, offset)) {
            return false;
        }
        v[i] = value;
    }
    return true;
}

} // namespace plainbuffer
