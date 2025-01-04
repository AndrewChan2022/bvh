#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include "load_bin.h"

#include <iostream>
#include <fstream>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;


namespace TraverseUtils {

static BVH_ALWAYS_INLINE Scalar FastDot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Utility function to compute the closest point on a triangle
static inline Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, Scalar& u, Scalar& v) {
    // Vectors from triangle vertices to the point
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;

    // Compute barycentric coordinates for point projection onto triangle plane
    Scalar d1 = FastDot(ab, ap);
    Scalar d2 = FastDot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a; // Closest to vertex A

    Vec3 bp = p - b;
    Scalar d3 = FastDot(ab, bp);
    Scalar d4 = FastDot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b; // Closest to vertex B

    Vec3 cp = p - c;
    Scalar d5 = FastDot(ab, cp);
    Scalar d6 = FastDot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c; // Closest to vertex C

    // Check if point is on edge AB
    Scalar vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        Scalar v = d1 / (d1 - d3);
        return a + v * ab; // Closest to edge AB
    }

    // Check if point is on edge AC
    Scalar vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        Scalar w = d2 / (d2 - d6);
        return a + w * ac; // Closest to edge AC
    }

    // Check if point is on edge BC
    Scalar va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        Scalar w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // Closest to edge BC
    }

    // Point is inside the triangle
    Scalar denom = 1.0f / (va + vb + vc);
    v = vb * denom;
    Scalar w = vc * denom;
    u = 1.0f - v - w;
    return a + ab * v + ac * w;
}

// return optional uv
// todo: modify ray.tmin/tmax
static inline std::optional<std::pair<Scalar, Scalar>> sphere_triangle_intersect(const Ray& ray, const bvh::v2::PrecomputedTri<Scalar>& precomputed_tri) {
    const Vec3& center = ray.org;
    Scalar radius = ray.tmax;

    // Extract triangle vertices from PrecomputedTri
    const Vec3& p0 = precomputed_tri.p0;
    const Vec3& p1 = precomputed_tri.p0 - precomputed_tri.e1;
    const Vec3& p2 = precomputed_tri.e2 + precomputed_tri.p0;

    // Find the closest point on the triangle to the sphere center
    Scalar u, v;
    Vec3 closest_point = closest_point_on_triangle(center, p0, p1, p2, u, v);

    // Compute the distance between the sphere center and the closest point
    Vec3 diff = closest_point - center;
    Scalar dist_squared = FastDot(diff, diff);

    // Check if the distance is less than or equal to the sphere radius squared
    if( dist_squared <= radius * radius) {
        return std::make_optional(std::pair<Scalar, Scalar> { u, v });
    }

    return std::nullopt;
}

static inline Scalar squared_distance_point_to_bbox(const Vec3& point, const std::array<Scalar, 6>& bounds) {
    Scalar dist_squared = 0;

    // For each dimension (x, y, z)
    for (size_t i = 0; i < 3; ++i) {
        Scalar min_bound = bounds[i * 2];
        Scalar max_bound = bounds[i * 2 + 1];

        // If the point is outside the bounds, add squared distance
        if (point[i] < min_bound) {
            Scalar diff = min_bound - point[i];
            dist_squared += diff * diff;
        } else if (point[i] > max_bound) {
            Scalar diff = point[i] - max_bound;
            dist_squared += diff * diff;
        }
    }

    return dist_squared;    
}

// return t0, t1
// todo: modify ray.tmin/tmax
static inline std::pair<Scalar, Scalar> sphere_node_intersect(const Ray& ray, const Node& node) {
    const Vec3& center = ray.org;
    Scalar radius = ray.tmax;

    Scalar dist_squared = squared_distance_point_to_bbox(center, node.bounds);

    // Check if the distance is less than or equal to the sphere radius squared
    // return dist_squared <= radius * radius;
    if (dist_squared <= radius) {
        Scalar t0 = dist_squared, t1 = t0 + 1.0f;
        return std::pair<Scalar, Scalar> { t0, t1 };
    }

    return std::pair<Scalar, Scalar> { 1.0f, 0.0f };
}

}

struct Accel {
    Bvh bvh;
    std::vector<PrecomputedTri> tris;
    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor;

    // config
    static constexpr bool should_permute = true;
    static constexpr size_t stack_size = 64;
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;

    Accel(): executor(thread_pool) {
    }

    void build_accel(const std::vector<Tri>& tris) {
        
        // build box and centers
        std::vector<BBox> bboxes(tris.size());
        std::vector<Vec3> centers(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i]  = tris[i].get_bbox();
                centers[i] = tris[i].get_center();
            }
        });

        // build bvh
        Accel& accel = *this;

        typename bvh::v2::DefaultBuilder<Node>::Config config;
        config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
        accel.bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);

        // Permuting the primitive data allows to remove indirections during traversal, which makes it faster.
        // This precomputes some data to speed up traversal further.
        accel.tris.resize(tris.size());
        if constexpr (should_permute) {
            executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                    accel.tris[i] = tris[accel.bvh.prim_ids[i]];
            });
        } else {
            executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                    accel.tris[i] = tris[i];
            });
        }
    }

    bool ray_intersect(Ray& ray) {
        constexpr bool isAnyHit = true;
    }

    // result is intersect result, true for intersect
    void batch_ray_intersect(std::vector<Vec3>& origins, std::vector<Vec3>& targets, std::vector<uint8_t>& result) {

        assert(origins.size() == targets.size());

        if (result.size() != origins.size()) {
            result.resize(origins.size());
        }

        // parallel for
        executor.for_each(0, origins.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto dir = targets[i] - origins[i];
                auto ray = Ray {
                    origins[i],         // Ray origin
                    dir,                // Ray direction
                    0.,                 // Minimum intersection distance
                    1.                  // Maximum intersection distance
                };
                auto ret = ray_intersect(ray);
                result[i] = ret ? 1 : 0;
            }
        });
    }

    size_t sphere_intersect(Ray& ray) {
        constexpr bool isAnyHit = true;

        static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
        auto prim_id = invalid_id;

        bvh.traverse_top_down<isAnyHit>(bvh.get_root().index, stack, 
            [&] (size_t begin, size_t end) {    // leaf
                for (size_t i = begin; i < end; ++i) {
                    size_t j = should_permute ? i : bvh.prim_ids[i];
                    if(auto hit = TraverseUtils::sphere_triangle_intersect(ray, tris[j])) {
                        prim_id = i;
                        return true;
                    }
                }
                return false;
            }, 
            [&] (const Node& left, const Node& right) {     // inner
                
                std::pair<Scalar, Scalar> intr_left, intr_right;
                intr_left = TraverseUtils::sphere_node_intersect(ray, left);
                intr_right = TraverseUtils::sphere_node_intersect(ray, right);

                // left hit, right hit, swap left right
                return std::make_tuple(
                    intr_left.first <= intr_left.second,
                    intr_right.first <= intr_right.second,
                    !isAnyHit && intr_left.first > intr_right.first);
            });

        return prim_id;
    }

    // result is intersect result, true for intersect
    void batch_sphere_intersect(std::vector<Vec3>& centers, Scalar radius, std::vector<uint8_t>& result) {

        if (result.size() != centers.size()) {
            result.resize(centers.size());
        }

        // parallel for
        executor.for_each(0, centers.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto ray = Ray {
                    centers[i],         // Ray origin
                    Vec3(0., 0., 1.),   // Ray direction
                    0.,                 // Minimum intersection distance
                    radius              // Maximum intersection distance
                };
                auto prim_id = sphere_intersect(ray);
                static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
                result[i] = prim_id != invalid_id ? 1 : 0;
            }
        });
    }
};

int main() {

    // load mesh
    RenderMesh mesh;
    std::vector<Tri> tris = mesh.loadBin("D:/data/dataset/obj/torus_knot_sparse_1k.bin").getTris();

    // build accel
    Accel accel;
    accel.build_accel(tris);

    // sphere test
    auto O = tris[0].p0 + 0.299999f * mesh.getNormal(0, 0);  // offset along normal
    auto ray = Ray {
        O,                // Ray origin
        Vec3(0., 0., 1.), // Ray direction
        0.,               // Minimum intersection distance
        0.3               // Maximum intersection distance
    };
    auto prim_id = accel.sphere_intersect(ray);
    
    // print result
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
    if (prim_id != invalid_id) {
        size_t j = prim_id;
        auto tri = accel.tris[j].convert_to_tri();
        std::cout
            << "Intersection found\n"
            << "  primitive: " << prim_id << "\n"
            << "  vertices  : " << tri.p0[0] << " " << tri.p0[1] << " " << tri.p0[2] << "\n"
            << "  should be : " << tris[0].p0[0] << " " << tris[0].p0[1] << " " << tris[0].p0[2] << "\n"
            << "  distance: " << ray.tmax << "\n";
            //<< "  barycentric coords.: " << u << ", " << v << std::endl;
    } else {
        std::cout << "No intersection found" << std::endl;
    }

    return 0;
}
