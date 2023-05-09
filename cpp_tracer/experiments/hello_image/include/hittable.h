#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

struct hit_record{
    point3 p;
    vec3 normal;
    double t;
    bool front_face;

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class hittable{
    public:
        // Use a virtual function so that any hittable object that extends this class can override it
        // The = 0 makes it a pure virtual function, i.e, it has to be derived. The const keyword means data cannot be changed
        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif