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
            // Using header imported functions
            color pixel_color(double(i) / (image_width - 1), double(j) / (image_height - 1), 0.25);
            write_color(std::cout, pixel_color);
        }
    }
    std::cerr << "\nDone\n" << std::flush;
    return 0;
}