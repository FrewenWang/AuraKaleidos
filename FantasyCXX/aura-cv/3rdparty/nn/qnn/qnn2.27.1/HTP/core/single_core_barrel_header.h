#ifndef SINGLE_CORE_BARREL_HEADER_H_
#define SINGLE_CORE_BARREL_HEADER_H_

#include <stdexcept>

#include "pickle_header_tags.h"

// get the pointer to the CONTENTS record of the barrel
static inline void *htp_header_get_record_ptr(const void *buf, size_t buflen, HTP_header_const rec, unsigned min_len)
{
    // Make sure it's a barrel
    if (buflen < 32 || htp_header_get_MAGIC(buf) != Hdr_MAGIC_MULTI) return 0;

    void *toc_payload_p = nullptr;
    int const toc_payload_len = htp_header_locate_field(buf, buflen, rec, &toc_payload_p);

    if (toc_payload_len < int(min_len * sizeof(unsigned))) return 0;

    return toc_payload_p;
}

// get the pointer to the CONTENTS record of the barrel
static inline void *htp_header_get_contents_ptr(const void *buf, size_t buflen)
{
    return htp_header_get_record_ptr(buf, buflen, HdrTag_CONTENTS, 4);
}

// get the pointer to the MULTI record of the barrel
static inline void *htp_header_get_multi_rec_ptr(const void *buf, size_t buflen)
{
    return htp_header_get_record_ptr(buf, buflen, HdrTag_MULTI, 2);
}

// make sure it's a single core barrel.
// the number of nsps should be equal to 1.
// blob id should be (1 << 16u).
static inline bool htp_header_verify_single_core_barrel(const void *buf, size_t buflen)
{
    unsigned const *toc_payload_p = (unsigned const *)htp_header_get_contents_ptr(buf, buflen);
    if (!toc_payload_p) throw std::runtime_error("Unable to get content record from barrel.");

    // toc_ptr[0] is the location of `number of nsps`
    // make sure it's 1
    if (toc_payload_p[0] != 1) return false;

    // just make sure that the blob id is correct !
    // toc_ptr[2] is the blod id
    unsigned int const nsp_index = 0;
    unsigned int const blob_id = (nsp_index + 1) << 16u;
    if (toc_payload_p[2] != blob_id) return false;

    return true;
}

// determine the number of NSPs in a barrel
static inline size_t htp_header_get_num_blobs(const void *buf, size_t buflen)
{
    unsigned const *toc_payload_p = (unsigned const *)htp_header_get_contents_ptr(buf, buflen);
    if (!toc_payload_p) throw std::runtime_error("Unable to get content record from barrel.");

    // toc_ptr[0] is the location of `number of nsps`
    return toc_payload_p[0];
}

// determine the size of the multicore shared spill/fill buffer.
static inline size_t htp_header_get_mc_shared_buf_size(const void *buf, size_t buflen)
{
    unsigned const *multi_payload_p = (unsigned const *)htp_header_get_multi_rec_ptr(buf, buflen);
    if (!multi_payload_p) throw std::runtime_error("Unable to get multi record from barrel.");

    // ptr[1] is the location of size of the shared buf, in units of 256B.
    return multi_payload_p[1] * 256;
}

// For a single core barrel, determine the blob0 offset and blob_size
static inline size_t htp_header_single_core_barrel_loc(const void *buf, size_t buflen, size_t &blob_size)
{
    unsigned const *toc_payload_p = (unsigned const *)htp_header_get_contents_ptr(buf, buflen);
    if (!toc_payload_p) throw std::runtime_error("Unable to get content record from barrel.");

    // blob_offset and blob size in bytes
    size_t blob_offset = toc_payload_p[1] << 4;
    blob_size = toc_payload_p[3] << 4;

    return blob_offset;
}

#endif /* SINGLE_CORE_BARREL_HEADER_H_ */
