cdef float clip(float a, float min_value, float max_value) nogil

# This is used to avoid passing around float[::1] memoryviews in the critical path.  These seem to
# require reference counting which has a large performance overhead.
cdef packed struct float3:
    float[3] data

cdef float3 convert_rgb_to_cam16ucs(float[:, ::1] rgb_to_cam16ucs, float r, float g, float b) nogil

cdef float colour_distance_squared(float[3] colour1, float[3] colour2) nogil
