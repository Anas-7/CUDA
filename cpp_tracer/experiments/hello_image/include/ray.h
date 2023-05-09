#ifndef RAY_H
#define RAY_H
#include "vec3.h"

// Define a ray class
// A ray has an origin and a direction
// A point on the ray is given by origin + t * direction
// This is basic vectors

class ray{
    public:
        point3 orig;
        vec3 dir;
    
    public:
        ray() {}
        ray(const point3& o, const point3& d): orig(o), dir(d){}

        point3 origin() const{
            return orig;
        }

        vec3 direction() const{
            return dir;
        }

        point3 at(double t) const{
            return orig + dir * t;
        }
};


#endif