cdef float clip(float a, float min_value, float max_value) nogil

cdef float[::1] convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) nogil

cdef double colour_distance_squared(float[::1] colour1, float[::1] colour2) nogil