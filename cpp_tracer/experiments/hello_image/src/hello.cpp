#include<iostream>
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "rtweekend.h"
#include "sphere.h"
#include "camera.h"

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

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

     // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    if (world.hit(r, 0, infinity, rec)) {
        point3 target = rec.p + rec.normal + random_in_unit_sphere();
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth-1);
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
    const int depth = 50;

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

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
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, depth);
            }
            write_color(std::cout, pixel_color, samples_pixel);
        }
    }
    std::cerr << "\nDone\n" << std::flush;
    return 0;
}