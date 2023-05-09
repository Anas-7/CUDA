#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

// This will extend the hittable class
class sphere : public hittable{
    public:
        point3 center;
        double radius;
    public:
        sphere(){}
        sphere(point3 c, double r): center(c), radius(r){};

        // Override function defined in hittable.h
        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
};
// Implement the function
bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const{
    // This is what we earlier had in the main file itself in the hit_sphere() method
    vec3 oc = r.origin() - center;
    auto a = r.direction().length() * r.direction().length();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length() * oc.length() - radius*radius;

    auto discr = half_b*half_b - a*c;
    if (discr < 0) return false;

    auto sqrt_discr = std::sqrt(discr);
    // Find the root that lies in the acceptable range
    auto root = (-half_b - sqrt_discr) / a;
    // If it violates any condition
    // The diffuse issue was fixed by this: https://github.com/RayTracing/raytracing.github.io/issues/875#issuecomment-1013842362
    if (root <= t_min || root >= t_max){
        // Assign the other root
        root = (-half_b + sqrt_discr) / a;
        // Check if this violates the condition too
        if (root <= t_min || root >= t_max){
            return false;
        }
    }
    // Update the hit record with the root which is in acceptable range
    rec.t = root;
    rec.p = r.at(rec.t);
    // Always calculate the outward normal this way, then use the method to check if ray hit outside or inside the sphere
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    
    return true;
}

#endif