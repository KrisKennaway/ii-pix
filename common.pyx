# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False


cdef inline float clip(float a, float min_value, float max_value) noexcept nogil:
    """Clip a value between min_value and max_value inclusive."""
    return min(max(a, min_value), max_value)


cdef inline float3 convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) noexcept nogil:
    """Converts floating point (r,g,b) valueto 3-tuple in CAM16UCS colour space, via 24-bit RGB lookup matrix."""

    cdef unsigned int rgb_24bit = (<unsigned int>(r*255) << 16) + (<unsigned int>(g*255) << 8) + <unsigned int>(b*255)
    cdef float3 res
    cdef int i
    for i in range(3):
        res.data[i] = rgb_to_cam16ucs[rgb_24bit][i]
    return res


cdef inline float colour_distance_squared(float[3] colour1, float[3] colour2) noexcept nogil:
    """Computes Euclidean squared distance between two floating-point colour 3-tuples."""

    return (
        (colour1[0] - colour2[0]) * (colour1[0] - colour2[0]) +
        (colour1[1] - colour2[1]) * (colour1[1] - colour2[1]) +
        (colour1[2] - colour2[2]) * (colour1[2] - colour2[2])
    )

