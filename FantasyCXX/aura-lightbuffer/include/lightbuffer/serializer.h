// #ifndef PLAIN_BUFFER_SERIALIZER_H
// #define PLAIN_BUFFER_SERIALIZER_H
//
// #include <cstdint>
// #include <cstring>
// #include <string>
// #include <type_traits>
// #include <vector>
//
// #include "plainbuffer/message_base.h"
//
// namespace plainbuffer {
//
// template<class T>
// using is_string = std::is_same<typename std::decay<T>::type, std::string>;
//
// class Serializer {
// public:
//     static bool serialize_char(char v, char *data, int size, int &offset);
//     static bool serialize_short(short v, char *data, int size, int &offset);
//     static bool serialize_int(int v, char *data, int size, int &offset);
//     static bool serialize_int64_t(int64_t v, char *data, int size, int &offset);
//     static bool serialize_float(float v, char *data, int size, int &offset);
//     static bool serialize_double(double v, char *data, int size, int &offset);
//     static bool serialize_bool(bool v, char *data, int size, int &offset);
//     static bool serialize_string(const std::string &s, char *data, int size, int &offset);
//     static bool serialize_bytes(const char *bytes, size_t bytes_len, char *data, int size, int &offset);
//     static bool serialize_vector_bytes(const std::vector<const char *> &v, const std::vector<size_t> &v_len, char *data, int size, int &offset);
//     static bool serialize_message(const MessageBase &msg, char *data, int size, int &offset);
//
//     template<typename T>
//     static bool serialize_vector(const std::vector<T> &v, char *data, int size, int &offset,
//                                  typename std::enable_if<std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type = 0);
//
//     template<typename T>
//     static bool serialize_vector(const std::vector<T> &v, char *data, int size, int &offset,
//                                  typename std::enable_if<!std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type = 0);
//
// protected:
//     template<typename T>
//     static bool serialize_primary_element(T v, char *data, int size, int &offset);
// };
//
// class Deserializer {
// public:
//     static bool deserialize_char(char &v, const char *data, int size, int &offset);
//     static bool deserialize_short(short &v, const char *data, int size, int &offset);
//     static bool deserialize_int(int &v, const char *data, int size, int &offset);
//     static bool deserialize_int64_t(int64_t &v, const char *data, int size, int &offset);
//     static bool deserialize_float(float &v, const char *data, int size, int &offset);
//     static bool deserialize_double(double &v, const char *data, int size, int &offset);
//     static bool deserialize_bool(bool &v, const char *data, int size, int &offset);
//     static bool deserialize_string(std::string &s, const char *data, int size, int &offset);
//     static bool deserialize_bytes(const char *&bytes, size_t &bytes_len, const char *data, int size, int &offset);
//     static bool deserialize_vector_bytes(std::vector<const char *> &v, std::vector<size_t> &v_len, const char *data, int size, int &offset);
//     static bool deserialize_message(MessageBase &msg, const char *data, int size, int &offset);
//
//     template<typename T>
//     static bool deserialize_vector(std::vector<T> &v, const char *data, int size, int &offset,
//                                    typename std::enable_if<std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type = 0);
//
//     template<typename T>
//     static bool deserialize_vector(std::vector<T> &v, const char *data, int size, int &offset,
//                                    typename std::enable_if<!std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type = 0);
//
// protected:
//     template<typename T>
//     static bool deserialize_primary_element(T &v, const char *data, int size, int &offset);
// };
//
// template<typename T>
// bool Serializer::serialize_primary_element(T v, char *data, int size, int &offset) {
//     int data_len = sizeof(T);
//     if (offset + data_len > size) {
//         printf("buffer size is too small, current offset=%d, data_len=%d, sizeof buffer=%d\n",
//                offset, data_len, size);
//         return false;
//     }
//     memcpy(data + offset, &v, data_len);
//     offset += data_len;
//     return true;
// }
//
// template<typename T>
// bool Serializer::serialize_vector(const std::vector<T> &v, char *data, int size, int &offset,
//                                   typename std::enable_if<std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type) {
//     int v_size = static_cast<int>(v.size());
//     serialize_primary_element<int>(v_size, data, size, offset);
//     if (v_size == 0) {
//         return true;
//     }
//     for (int i = 0; i < v_size; ++i) {
//         const auto &msg = v[i];
//         if (!serialize_message(msg, data, size, offset)) {
//             return false;
//         }
//     }
//     return true;
// }
//
// template<typename T>
// bool Serializer::serialize_vector(const std::vector<T> &v, char *data, int size, int &offset,
//                                   typename std::enable_if<!std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type) {
//     int v_size = static_cast<int>(v.size());
//     serialize_primary_element<int>(v_size, data, size, offset);
//     if (v_size == 0) {
//         return true;
//     }
//
//     for (int i = 0; i < v_size; ++i) {
//         if (!serialize_primary_element<T>(v[i], data, size, offset)) {
//             return false;
//         }
//     }
//     return true;
// }
//
// template<>
// inline bool Serializer::serialize_vector<std::string>(const std::vector<std::string> &v, char *data, int size, int &offset,
//                                                       typename std::enable_if<true, int>::type) {
//     int v_size = static_cast<int>(v.size());
//     serialize_primary_element<int>(v_size, data, size, offset);
//     if (v_size == 0) {
//         return true;
//     }
//
//     for (int i = 0; i < v_size; ++i) {
//         if (!serialize_string(v[i], data, size, offset)) {
//             return false;
//         }
//     }
//     return true;
// }
//
// template<typename T>
// bool Deserializer::deserialize_primary_element(T &v, const char *data, int size, int &offset) {
//     int data_len = sizeof(T);
//     if (offset + data_len > size) {
//         return false;
//     }
//     const auto *ptr = reinterpret_cast<const T *>(data + offset);
//     v = *(ptr);
//     offset += data_len;
//     return true;
// }
//
// template<typename T>
// bool Deserializer::deserialize_vector(std::vector<T> &v, const char *data, int size, int &offset,
//                                       typename std::enable_if<std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type) {
//     int v_size = 0;
//     if (!deserialize_primary_element<int>(v_size, data, size, offset)) {
//         return false;
//     }
//
//     if (v_size == 0) {
//         return true;
//     }
//
//     v.clear();
//     v.resize(v_size);
//     for (int i = 0; i < v_size; ++i) {
//         T msg;
//         if (!deserialize_message(msg, data, size, offset)) {
//             return false;
//         }
//         v[i] = msg;
//     }
//     return true;
// }
//
// template<typename T>
// bool Deserializer::deserialize_vector(std::vector<T> &v, const char *data, int size, int &offset,
//                                       typename std::enable_if<!std::is_base_of<plainbuffer::MessageBase, T>::value, int>::type) {
//     int v_size = 0;
//     if (!deserialize_primary_element<int>(v_size, data, size, offset)) {
//         return false;
//     }
//
//     if (v_size == 0) {
//         return true;
//     }
//
//     v.clear();
//     v.resize(v_size);
//     for (int i = 0; i < v_size; ++i) {
//         T value;
//         if (!deserialize_primary_element<T>(value, data, size, offset)) {
//             return false;
//         }
//         v[i] = value;
//     }
//     return true;
// }
//
// template<>
// inline bool Deserializer::deserialize_vector<std::string>(std::vector<std::string> &v, const char *data, int size, int &offset,
//                                                           typename std::enable_if<true, int>::type) {
//     int v_size = 0;
//     if (!deserialize_primary_element<int>(v_size, data, size, offset)) {
//         return false;
//     }
//
//     if (v_size == 0) {
//         return true;
//     }
//
//     v.clear();
//     v.resize(v_size);
//     for (int i = 0; i < v_size; ++i) {
//         if (!deserialize_string(v[i], data, size, offset)) {
//             return false;
//         }
//     }
//     return true;
// }
//
// } // namespace plainbuffer
//
// #endif // PLAIN_BUFFER_SERIALIZER_H
