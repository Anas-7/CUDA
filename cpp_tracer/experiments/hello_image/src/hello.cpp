#include<iostream>
#include "vec3.h"
#include "color.h"
#include "ray.h"
// using namespace std;

//The ray_color(ray) function linearly blends white and blue depending on the height of the y
//coordinate after scaling the ray direction to unit length (so −1.0<y<1.0)
color ray_color(const ray& r){
    vec3 unit_dir = unit_vector(r.direction());
    // Choosing a point on the ray
    auto t = 0.5 * (unit_dir.y() + 1.0);
    // When t=1.0 I want blue. When t=0.0 I want white. In between, I want a blend. 
    // This forms a “linear blend”, or “linear interpolation”, or “lerp” for short, between two things.
    color c = (1 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
    return c;
}

int main()
{
    // For the camera
    const auto aspect_ratio = 16.0 / 9.0;
    // Image dimensions. const not needed, but good practice
    const int image_width = 400;
    const int image_height = image_width / aspect_ratio;

    // https://en.wikipedia.org/wiki/Netpbm#PPM_example
    /*
     "P3" means this is a RGB color image in ASCII
     "256 256" is the width and height of the image in pixels
     "255" is the maximum value for each color
    */
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    /*
    (1) calculate the ray from the eye to the pixel, 
    (2) determine which objects the ray intersects, and 
    (3) compute a color for that intersection point.

    In addition to setting up the pixel dimensions for the rendered image, 
    we also need to set up a virtual viewport through which to pass our scene rays

    For the standard square pixel spacing, the viewport's aspect ratio should be the same as our rendered image
    We'll just pick a viewport two units in height.
    
    We'll also set the distance between the projection plane and the projection point to be one unit. 
    This is referred to as the “focal length”

    Put the “eye” (or camera center if you think of a camera) at (0,0,0)
    
    y-axis goes up, and the x-axis to the right. 
    In order to respect the convention of a right handed coordinate system, into the screen is the negative z-axis

    Traverse the screen from the upper left hand corner, and use two offset vectors u and v along the screen sides to move the ray endpoint across the screen.
    
    Note that we havent made the ray direction a unit length vector, but do so in the ray_color function


    */



    // Define the camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);

    // If viewport_width was 800, height 600, and focal length 1, then lower_left_corner would be (-400, -300, -1)
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    for(int j = image_height - 1; j >= 0; j--){
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for(int i = 0; i < image_width; i++){
            // u and v act as offsets
            auto u = double(i) / (image_width - 1);
            auto v = double(j) / (image_height - 1);

            /*
            Since j is iterating over image height, double(j) / (image_height - 1); is a ratio of the height
            Multiplying this ratio with the vertical height of the viewport, we get the y direction of the ray
            
            Similarly, i gives us width, so double(i) / (image_width - 1) * horizontal gives the width of the viewport
            
            The ray is calculated according to the viewport because the viewport represents 
            the 2D plane in space where the virtual image is projected

            Rays go from the camera origin to the viewport, and the viewport is a plane in space
            The rays then go from the pixel on the viewport to the objects in the scene
            */
            // The - origin is the camera origin
            ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
            // Using header imported functions
            color pixel_color = ray_color(r);
            write_color(std::cout, pixel_color);
        }
    }
    std::cerr << "\nDone\n" << std::flush;
    return 0;
}