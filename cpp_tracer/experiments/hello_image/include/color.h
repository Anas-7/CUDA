#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <iostream>

void write_color(std::ostream &out, color v){
    out << static_cast<int>(255.999 * v[0]) << " " << static_cast<int>(255.999 * v[1]) << " " << static_cast<int>(255.999 * v[2]) << '\n';
}

#endif

