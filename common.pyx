# cython: infer_types=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False


cdef float clip(float a, float min_value, float max_value) nogil:
    """Clip a value between min_value and max_value inclusive."""
    return min(max(a, min_value), max_value)


cdef inline float[::1] convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) nogil:
    """Converts floating point (r,g,b) valueto 3-tuple in CAM16UCS colour space, via 24-bit RGB lookup matrix."""

    cdef unsigned int rgb_24bit = (<unsigned int>(r*255) << 16) + (<unsigned int>(g*255) << 8) + <unsigned int>(b*255)
    return rgb_to_cam16ucs[rgb_24bit]


cdef inline double colour_distance_squared(float[::1] colour1, float[::1] colour2) nogil:
    """Computes Euclidean squared distance between two floating-point colour 3-tuples."""

    return (colour1[0] - colour2[0]) ** 2 + (colour1[1] - colour2[1]) ** 2 + (colour1[2] - colour2[2]) ** 2

