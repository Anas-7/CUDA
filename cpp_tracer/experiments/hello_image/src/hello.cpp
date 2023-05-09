#include<iostream>
#include "vec3.h"
#include "color.h"
// using namespace std;

int main()
{
    // Image dimensions. const not needed, but good practice
    const int image_width = 256;
    const int image_height = 256;

    // https://en.wikipedia.org/wiki/Netpbm#PPM_example
    /*
     "P3" means this is a RGB color image in ASCII
     "256 256" is the width and height of the image in pixels
     "255" is the maximum value for each color
    */
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for(int j = image_height - 1; j >= 0; j--){
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for(int i = 0; i < image_width; i++){
            // The values stored in r g b are floats, but they are casted to int when written to the file
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.25;
            
            // Could have used normal (int) cast, but static_cast is more strict and runs at compile time
            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    std::cerr << "\nDone\n" << std::flush;
    return 0;
}