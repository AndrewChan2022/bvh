#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include <iostream>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

// Utility function to compute the closest point on a triangle
static Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    // Vectors from triangle vertices to the point
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;

    // Compute barycentric coordinates for point projection onto triangle plane
    Scalar d1 = bvh::v2::dot(ab, ap);
    Scalar d2 = bvh::v2::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a; // Closest to vertex A

    Vec3 bp = p - b;
    Scalar d3 = bvh::v2::dot(ab, bp);
    Scalar d4 = bvh::v2::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b; // Closest to vertex B

    Vec3 cp = p - c;
    Scalar d5 = bvh::v2::dot(ab, cp);
    Scalar d6 = bvh::v2::dot(ac, cp);
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
    Scalar v = vb * denom;
    Scalar w = vc * denom;
    return a + ab * v + ac * w;
}

static inline bool sphere_triangle_intersect(Vec3& center, float radius, Tri& triangle) {
    // Extract triangle vertices
    const Vec3& v0 = triangle.p0;
    const Vec3& v1 = triangle.p1;
    const Vec3& v2 = triangle.p2;

    // Find the closest point on the triangle to the sphere center
    Vec3 closest_point; // = bvh::v2::closest_point_on_triangle(center, v0, v1, v2);

    // Compute the distance between the sphere center and the closest point
    Vec3 diff = closest_point - center;
    float dist_squared = bvh::v2::dot(diff, diff);

    // Check if the distance is less than or equal to the sphere radius squared
    return dist_squared <= radius * radius;
}

static inline bool sphere_node_intersect(Vec3& center, float radius, Node& triangle) {

}

int main() {
    // This is the original data, which may come in some other data type/structure.
    std::vector<Tri> tris;
    tris.emplace_back(
        Vec3( 1.0, -1.0, 1.0),
        Vec3( 1.0,  1.0, 1.0),
        Vec3(-1.0,  1.0, 1.0)
    );
    tris.emplace_back(
        Vec3( 1.0, -1.0, 1.0),
        Vec3(-1.0, -1.0, 1.0),
        Vec3(-1.0,  1.0, 1.0)
    );

    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    // Get triangle centers and bounding boxes (required for BVH builder)
    std::vector<BBox> bboxes(tris.size());
    std::vector<Vec3> centers(tris.size());
    executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            bboxes[i]  = tris[i].get_bbox();
            centers[i] = tris[i].get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    auto bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);

    // Permuting the primitive data allows to remove indirections during traversal, which makes it faster.
    static constexpr bool should_permute = true;

    // This precomputes some data to speed up traversal further.
    std::vector<PrecomputedTri> precomputed_tris(tris.size());
    executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            auto j = should_permute ? bvh.prim_ids[i] : i;
            precomputed_tris[i] = tris[j];
        }
    });

    auto ray = Ray {
        Vec3(0., 0., 0.), // Ray origin
        Vec3(0., 0., 1.), // Ray direction
        0.,               // Minimum intersection distance
        100.              // Maximum intersection distance
    };

    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
    static constexpr size_t stack_size = 64;
    static constexpr bool use_robust_traversal = false;

    auto prim_id = invalid_id;
    Scalar u, v;

    // Traverse the BVH and get the u, v coordinates of the closest intersection.
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    bvh.traverse_top_down<true>(bvh.get_root().index, stack, 
        // leaf
        [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                size_t j = should_permute ? i : bvh.prim_ids[i];
                if (auto hit = precomputed_tris[j].intersect(ray)) {
                    prim_id = i;
                    std::tie(u, v) = *hit;
                }
            }
            return prim_id != invalid_id;
        }, 
        // inner
        [&] (const Node& left, const Node& right) {
            
            std::pair<Scalar, Scalar> intr_left, intr_right;
            if constexpr (IsRobust) {
                intr_left  = left .intersect_robust(ray, inv_dir, inv_dir_pad, octant);
                intr_right = right.intersect_robust(ray, inv_dir, inv_dir_pad, octant);
            } else {
                intr_left  = left .intersect_fast(ray, inv_dir, inv_org, octant);
                intr_right = right.intersect_fast(ray, inv_dir, inv_org, octant);
            }
            bool leftHit = true;
            bool rightHit = false;
            bool swapLeftRight = !IsAnyHit && intr_left.first > intr_right.first;

            return std::make_tuple(leftHit, rightHit, swapLeftRight);
        });

    if (prim_id != invalid_id) {
        std::cout
            << "Intersection found\n"
            << "  primitive: " << prim_id << "\n"
            << "  distance: " << ray.tmax << "\n"
            << "  barycentric coords.: " << u << ", " << v << std::endl;
        return 0;
    } else {
        std::cout << "No intersection found" << std::endl;
        return 1;
    }
}
