#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace aura::light_buffer {

class TextFormat;

// Maps a field name to its declared type tag (e.g. "int32", "string", "Address_$v").
using MessageDescriptor = std::unordered_map<std::string, std::string>;
// Maps a message type name to its field descriptor.
using MessageDescriptorMap = std::unordered_map<std::string, MessageDescriptor>;
// Maps a field name to a list of text values (for repeated fields there are several).
using DataDescriptor = std::unordered_map<std::string, std::vector<std::string>>;
// Maps a nested message path (e.g. "Person", "PersonAddress0") to its data descriptor.
using DataDescriptorMap = std::unordered_map<std::string, DataDescriptor>;

/// @brief Base class for every generated LightBuffer message.
///
/// The C++ code generator (CodeGeneratorCpp) emits message structs that inherit
/// from MessageBase and override the six pure-virtual `internal_*` methods. The
/// public wrappers forward to those overrides and provide a protobuf-compatible API.
class MessageBase {
public:
    virtual ~MessageBase() = default;

    bool SerializeToArray(void *data, int size) const {
        return internal_serialize((char *) data, size);
    }

    bool SerializeToArray(char *data, int size) const {
        return internal_serialize(data, size);
    }

    bool ParseFromArray(const void *data, int size) {
        return internal_deserialize((const char *) data, size);
    }

    bool ParseFromArray(const char *data, int size) {
        return internal_deserialize(data, size);
    }

    bool ParseFromDescriptor(DataDescriptorMap &data_map, const std::string &key) {
        return internal_parse_from_descriptor(data_map, key);
    }

    size_t ByteSizeLong() const {
        return internal_bytes_size();
    }

    MessageDescriptorMap GetDescriptor() const {
        return internal_get_descriptor();
    }

    std::string GetTypeName() const {
        return internal_get_typename();
    }

protected:
    /// @brief Serialize this message into a flat binary buffer.
    virtual bool internal_serialize(char *data, int size) const = 0;

    /// @brief Deserialize this message from a flat binary buffer.
    virtual bool internal_deserialize(const char *data, int size) = 0;

    /// @brief Populate this message from a text-format descriptor map.
    virtual bool internal_parse_from_descriptor(DataDescriptorMap &data_map, const std::string &key) = 0;

    /// @brief Number of bytes this message occupies when serialized.
    virtual size_t internal_bytes_size() const = 0;

    /// @brief Field descriptor map (for reflection / debugging).
    virtual MessageDescriptorMap internal_get_descriptor() const = 0;

    /// @brief Fully-qualified message type name.
    virtual std::string internal_get_typename() const = 0;

private:
    friend class TextFormat;
};

} // namespace aura::light_buffer
