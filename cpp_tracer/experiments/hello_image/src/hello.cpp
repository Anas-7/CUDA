#include<iostream>
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "rtweekend.h"
#include "sphere.h"
#include "camera.h"
// using namespace std;

/*
 the equation for a sphere centered at the origin of radius R is x^2+y^2+z^2=R^2

 if the sphere center is at (Cx,Cy,Cz): (x−Cx)^2+(y−Cy)^2+(z−Cz)^2=r^2

 Center C = vec(Cx, Cy, Cz)
 Point P = vec(x, y, z)

 (P - C).(P - C) = r^2

 But P = A + tB
 (A + tB - C).(A + tB - C) = r^2
 t^2(b*b) + 2*t*b*(A−C) + (A−C)⋅(A−C)−r^2=0

 The roots of this give the intersection points of the ray
 0 roots => no intersection, 1 root => tangent, 2 root => 2 points on sphere

*/
double hit_sphere(const point3& center, double radius, const ray& r){
    vec3 pc = (r.origin() - center);
    auto a = dot(r.direction(), r.direction());
    // auto b = 2.0 * dot(pc, r.direction());  Assume b = 2h and simplify the operations by using h
    auto half_b = dot(pc, r.direction());
    // auto c = dot(pc,pc) - radius*radius;    Dot product of a vector with itself can be replaced with its length squared
    auto c = pc.length() * pc.length() - radius*radius;
    auto discr = half_b*half_b - a*c;
    if (discr < 0){
        return -1;
    }
    // Since we arent assuming the case for t < 0, we will choose the closest root to us from the quadratic equation
    return (-half_b - std::sqrt(discr)) / a;
}

color ray_color(const ray& r, const hittable& world) {
    hit_record rec;
    // Check if the ray hits any hittable object
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    // Otherwise display the background gradient
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

int main()
{
    // For the camera
    const auto aspect_ratio = 16.0 / 9.0;
    // Image dimensions. const not needed, but good practice
    const int image_width = 400;
    const int image_height = image_width / aspect_ratio;
    const int samples_pixel = 100;
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


    camera cam;
    // Add the hittable list of objects, two spheres currently
    hittable_list world;
    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    for(int j = image_height - 1; j >= 0; j--){
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for(int i = 0; i < image_width; i++){
            color pixel_color(0, 0, 0);
            for (int k = 0; k < samples_pixel; k++){
                // u and v act as offsets
                auto u = double(i) / (image_width - 1);
                auto v = double(j) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world);
            }
            write_color(std::cout, pixel_color, samples_pixel);
        }
    }
    std::cerr << "\nDone\n" << std::flush;
    return 0;
}