#ifndef AURA_RUNTIME_CORE_TYPES_KEYPOINT_HPP__
#define AURA_RUNTIME_CORE_TYPES_KEYPOINT_HPP__

#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/core/defs.hpp"

#if !defined(AURA_BUILD_XTENSA)
#  include <iostream>
#  include <sstream>
#  include <string>
#  include <vector>
#endif

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup types Runtime Core Types
 *      @}
 * @}
*/

namespace aura
{

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief Template class for data structure for salient point detectors.
 *
 * The KeyPoint_ class stores a keypoint member m_pt, which is an instance of the Point2_ class, and also records
 * its feature parameters. Both the keypoint and its features are provided by a keypoint detector.
 *
 * It supports operations for converting between 2D points and keypoints, and also includes printing operations
 * for its members.
 */
template<typename Tp0>
class KeyPoint_
{
public:
    typedef Tp0 value_type;

    /**
     * @brief Default constructor for keypoint class.
     */
    KeyPoint_() : m_pt(0, 0), m_size(0.f), m_angle(0.f), m_response(0.f),
                  m_octave(0), m_class_id(0)
    {}

    /**
     * @brief Constructor initializing the keypoint with specified parameters.
     *
     * @param pt Point2_ object representing the keypoint's x and y coordinates.
     * @param size Keypoint diameter.
     * @param angle The orientation angle of the keypoint.
     * @param response keypoint detector response on the keypoint (that is, strength of the keypoint).
     * @param octave Pyramid octave in which the keypoint has been detected.
     * @param class_id The class identifier for the keypoint.
     */
    KeyPoint_(Point2_<Tp0> pt, DT_F32 size, DT_F32 angle = -1, DT_F32 response = 0,
              DT_S32 octave = 0, DT_S32 class_id = -1) : m_pt(pt), m_size(size),
              m_angle(angle), m_response(response), m_octave(octave), m_class_id(class_id)
    {}

    /**
     * @brief Constructor initializing the keypoint with specified parameters.
     *
     * @param x x-coordinate of the keypoint.
     * @param y y-coordinate of the keypoint.
     * @param size Keypoint diameter.
     * @param angle The orientation angle of the keypoint.
     * @param response Keypoint detector response on the keypoint (that is, strength of the keypoint).
     * @param octave Pyramid octave in which the keypoint has been detected.
     * @param class_id The class identifier for the keypoint.
     */
    KeyPoint_(Tp0 x, Tp0 y, DT_F32 size, DT_F32 angle = -1, DT_F32 response = 0,
              DT_S32 octave = 0, DT_S32 class_id = -1) : m_pt(x, y), m_size(size),
              m_angle(angle), m_response(response), m_octave(octave), m_class_id(class_id)
    {}

#if !defined(AURA_BUILD_XTENSA)
    /**
     * @brief Overloaded stream insertion operator to print the KeyPoint_ details.
     *
     * @param os Output stream to write the KeyPoint_ information.
     * @param kp KeyPoint_ object to print.
     *
     * @return Output stream adding KeyPoint_ information.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const KeyPoint_ &kp)
    {
        os << "(" << kp.m_pt << ", " << kp.m_size << ", "
           << kp.m_angle << ", " << kp.m_response << ", "
           << kp.m_octave << ", " <<  kp.m_class_id <<")";
        return os;
    }

    /**
     * @brief Converts the KeyPoint_ object to a string representation.
     *
     * @return std::string String representation of the KeyPoint_ object.
     */
    std::string ToString() const
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    /**
     * @brief Converts a vector of KeyPoint_ objects to a vector of 2D points.
     *
     * This method converts vector of keypoints to vector of points or the reverse, where each keypoint is
     * assigned the same size and the same orientation.
     *
     * @tparam Tp1 Data type for the converted points.
     *
     * @param keypoints Keypoints obtained from any feature detection algorithm.
     * @param points2d Vector of 2D points to store the converted keypoints.
     * @param kp_indexes Array of indexes of keypoints to be converted to points. (Acts like a mask to
     *                   convert only specified keypoints)
     */
    template<typename Tp1>
    AURA_INLINE DT_VOID Convert(const std::vector<KeyPoint_> &keypoints,
                                std::vector<Point2_<Tp1> > &points2d,
                                const std::vector<DT_S32> &kp_indexes = std::vector<DT_S32>())
    {
        if (kp_indexes.empty())
        {
            size_t kp_sizes = keypoints.size();
            points2d.resize(kp_sizes);
            for (size_t i = 0; i < kp_sizes; ++i)
            {
                points2d[i] = keypoints[i].m_pt;
            }
        }
        else
        {
            size_t kp_sizes = kp_indexes.size();
            points2d.resize(kp_sizes);
            for (size_t i = 0; i < kp_sizes; ++i)
            {
                DT_S32 idx = kp_indexes[i];
                if (idx >= 0)
                {
                    points2d[i] = keypoints[idx].m_pt;
                }
            }
        }
    }

    /**
     * @brief Converts a vector of 2D points to KeyPoint_ objects.
     *
     * @tparam Tp1 The data type of the input vector of 2D points.
     *
     * @param points2d Vector of 2D points to convert.
     * @param keypoints Keypoints obtained from any feature detection algorithm.
     * @param size Keypoint diameter
     * @param response Keypoint detector response on the keypoint (that is, strength of the keypoint)
     * @param octave Pyramid octave in which the keypoint has been detected
     * @param class_id The class identifier for the keypoint.
     */
    template<typename Tp1>
    AURA_INLINE DT_VOID Convert(const std::vector<Point2_<Tp1> > &points2d,
                                std::vector<KeyPoint_> &keypoints,
                                DT_F32 size = 1.f, DT_F32 response = 1.f,
                                DT_S32 octave = 0, DT_S32 class_id = 1)
    {
        size_t kp_sizes = points2d.size();
        keypoints.resize(kp_sizes);
        for (size_t i = 0; i < kp_sizes; ++i)
        {
            keypoints[i] = KeyPoint_(points2d[i], size, -1, response, octave, class_id);
        }
    }
#endif

    Point2_<Tp0> m_pt;  /*!< Coordinates of the keypoints */
    DT_F32 m_size;      /*!< Diameter of the meaningful keypoint neighborhood */
    DT_F32 m_angle;     /*!< Computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees
                             and measured relative to iaura coordinate system, ie in clockwise*/
    DT_F32 m_response;  /*!< The response by which the most strong keypoints have been selected. Can be used for
                             the further sorting or subsampling */
    DT_S32 m_octave;    /*!< Octave (pyramid layer) from which the keypoint has been extracted */
    DT_S32 m_class_id;  /*!< Object class (if the keypoints need to be clustered by an object they belong to) */
};

typedef KeyPoint_<DT_S32> KeyPointi;
typedef KeyPoint_<DT_F32> KeyPoint;

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_KEYPOINT_HPP__