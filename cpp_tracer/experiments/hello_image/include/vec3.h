#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3{
    public:
        // Its convention in graphics to refer to vector as e, for element
        vec3() : e{0,0,0} {}
        vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

        // Getters
        double x() const { return e[0]; }
        double y() const { return e[1]; }
        double z() const { return e[2]; }

        // Operator overloading
        vec3 operator- () const{
            return vec3(-e[0], -e[1], -e[2]);
        }

        double operator[] (int i) const {return e[i];}
        double& operator[] (int i) {return e[i];}

        vec3& operator+= (const vec3 other){
            e[0] += other.x();
            e[1] += other.y();
            e[2] += other.z();
            return *this;
        }
        vec3& operator-= (const vec3 other){
            e[0] -= other.x();
            e[1] -= other.y();
            e[2] -= other.z();
            return *this;
        }
        vec3& operator*= (const double t){
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }
        vec3& operator/= (const double t){
            *this *= 1/t;
            return *this;
        }

        double length(){
            return std::sqrt((e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]));
        }

    public:
        double e[3];

};

using point3 = vec3;
using color = vec3;

// Define the remaining functions

// The inline keyword ensures that the code executes as if it was present in the calling function itself

inline std::ostream& operator<< (std::ostream &out, const vec3 v){
    return out << v.x() << " " << v.y() << " " << v.z() << " " << std::endl;
}

inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]; 
}

inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2], u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif // VEC3_H




