#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <iostream>
#include "rtweekend.h"
// void write_color(std::ostream &out, color v){
//     out << static_cast<int>(255.999 * v[0]) << " " << static_cast<int>(255.999 * v[1]) << " " << static_cast<int>(255.999 * v[2]) << '\n';
// }
// Update the write color function to use the average of samples
void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples as the pixel color we receive has the summation of colors over all samples
    // auto scale = 1.0 / samples_per_pixel;
    // r *= scale;
    // g *= scale;
    // b *= scale;

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif

