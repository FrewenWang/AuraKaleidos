#pragma once

#include <unordered_map>
#include <vector>

namespace aura::light_buffer {
class TextFormat;
using MessageDescriptor = std::unordered_map<std::string, std::string>; // message level field descriptor
using MessageDescriptorMap = std::unordered_map<std::string, MessageDescriptor>; // nested message descriptor
using DataDescriptor = std::unordered_map<std::string, std::vector<std::string> >; // message level data list
using DataDescriptorMap = std::unordered_map<std::string, DataDescriptor>; // nested message data list

class MessageBase {
public:
    virtual ~MessageBase() = default;

    bool SerializeToArray(void *data, int size) const {
        return InternalSerialize((char *) data, size);
    }

    bool SerializeToArray(char *data, int size) const {
        return InternalSerialize(data, size);
    }

    bool ParseFromArray(const void *data, int size) {
        return InternalDeserialize((const char *) data, size);
    }

    bool ParseFromArray(const char *data, int size) {
        return InternalDeserialize(data, size);
    }

    bool ParseFromDescriptor(DataDescriptorMap &data_map, const std::string &key) {
        return InternalParseFromDescriptor(data_map, key);
    }


    size_t ByteSizeLong() const {
        return InternalBytesSize();
    }

    MessageDescriptorMap GetDescriptor() const {
        return InternalGetDescriptor();
    }

    std::string GetTypeName() const {
        return InternalGetTypeName();
    }

protected:
    /**
     * 纯虚函数：序列化方法
     * @param data
     * @param size
     * @return
     */
    virtual bool InternalSerialize(char *data, int size) const = 0;

    /**
     * 纯虚函数：反序列化方法
     * @param data
     * @param size
     * @return
     */
    virtual bool InternalDeserialize(const char *data, int size) = 0;

    /**
     *
     * @param data_map
     * @param key
     * @return
     */
    virtual bool InternalParseFromDescriptor(DataDescriptorMap &data_map, const std::string &key) = 0;

    /**
     *
     * @return
     */
    virtual size_t InternalBytesSize() const = 0;

    /**
     *
     * @return
     */
    virtual MessageDescriptorMap InternalGetDescriptor() const = 0;

    /**
     *
     * @return
     */
    virtual std::string InternalGetTypeName() const = 0;

private:
    /**
     * 有缘类
     */
    friend class TextFormat;
};
} // namespace aura::light_buffer
