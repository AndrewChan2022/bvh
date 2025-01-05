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

namespace next_gsp {

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
static inline Vec3 ClosestPointOnTriangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, Scalar& u, Scalar& v) {
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
static inline std::optional<std::pair<Scalar, Scalar>> SphereTriangleIntersect(const Ray& ray, const bvh::v2::PrecomputedTri<Scalar>& precomputed_tri) {
    const Vec3& center = ray.org;
    Scalar radius = ray.tmax;

    // Extract triangle vertices from PrecomputedTri
    const Vec3& p0 = precomputed_tri.p0;
    const Vec3& p1 = precomputed_tri.p0 - precomputed_tri.e1;
    const Vec3& p2 = precomputed_tri.e2 + precomputed_tri.p0;

    // Find the closest point on the triangle to the sphere center
    Scalar u, v;
    Vec3 closest_point = ClosestPointOnTriangle(center, p0, p1, p2, u, v);

    // Compute the distance between the sphere center and the closest point
    Vec3 diff = closest_point - center;
    Scalar dist_squared = FastDot(diff, diff);

    // Check if the distance is less than or equal to the sphere radius squared
    if( dist_squared <= radius * radius) {
        return std::make_optional(std::pair<Scalar, Scalar> { u, v });
    }

    return std::nullopt;
}

static inline Scalar SquaredDistancePointToBbox(const Vec3& point, const std::array<Scalar, 6>& bounds) {
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
static inline std::pair<Scalar, Scalar> SphereNodeIntersect(const Ray& ray, const Node& node) {
    const Vec3& center = ray.org;
    Scalar radius = ray.tmax;

    Scalar dist_squared = SquaredDistancePointToBbox(center, node.bounds);

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

    // parallel
    bvh::v2::ThreadPool threadPool;
    bvh::v2::ParallelExecutor executor;

    // config
    static constexpr bool shouldPermute = true;
    static constexpr bool useRobustTraversal = true;
    static constexpr size_t stackSize = 64 * 8;        // 512 * sizeof(Bvh::Index) = 2k bytes
    bvh::v2::SmallStack<Bvh::Index, stackSize> sharedStack;

    // constructor, init order by declare order, init list overwrite declare
    Accel(): threadPool(0), executor(threadPool) {
    }

    // build
    void buildAccel(const std::vector<Tri>& tris) {
        
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
        accel.bvh = bvh::v2::DefaultBuilder<Node>::build(threadPool, bboxes, centers, config);

        // Permuting the primitive data allows to remove indirections during traversal, which makes it faster.
        // This precomputes some data to speed up traversal further.
        accel.tris.resize(tris.size());
        if constexpr (shouldPermute) {
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

    /// @brief first of ray bvh intersect
    /// @param ray ray origin and ray direction, direction maybe not normalized
    /// @return prim_id after permuted
    size_t RayIntersect(Ray& ray, bvh::v2::SmallStack<Bvh::Index, stackSize>& stack) {
        constexpr bool isAnyHit = true;

        static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
        auto prim_id = invalid_id;

        bvh.intersect<isAnyHit, useRobustTraversal>(ray, bvh.get_root().index, stack,
            [&] (size_t begin, size_t end) -> bool {
                for (size_t i = begin; i < end; ++i) {
                    size_t j = shouldPermute ? i : bvh.prim_ids[i];
                    if (auto hit = tris[j].intersect(ray)) {
                        prim_id = i;
                        return true;
                    }
                }
                return prim_id != invalid_id;
            });
        
        return prim_id;
    }

    /// @brief batch call RayIntersect with many rays  
    /// @param origins ray origins
    /// @param diretions ray diretions
    /// @param tmin tmin, so real min distance is (target - origin) * tmin
    /// @param tmax tmax, so real max distance is (target - origin) * tmax
    /// @param result intersect result, 1 of intersect else 0
    void BatchRayIntersect(const std::vector<Vec3>& origins, const std::vector<Vec3>& diretions, Scalar tmin, Scalar tmax, std::vector<uint8_t>& result) {

        assert(origins.size() == diretions.size());

        if (result.size() != origins.size()) {
            result.resize(origins.size());
        }

        // parallel for
        executor.for_each(0, origins.size(), [&] (size_t begin, size_t end) {
            bvh::v2::SmallStack<Bvh::Index, stackSize> stack;
            for (size_t i = begin; i < end; ++i) {
                auto ray = Ray {
                    origins[i],         // Ray origin
                    diretions[i],       // Ray direction
                    tmin,               // Minimum intersection distance
                    tmax                // Maximum intersection distance
                };
                stack.size = 0;
                auto prim_id = RayIntersect(ray, stack);
                static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
                result[i] = prim_id != invalid_id ? 1 : 0;
            }
        });
    }

    /// @brief first of sphere bvh intersect
    /// @param ray define sphere with ray.origin as center and ray.tmax as radius
    /// @return prim_id after permuted
    size_t SphereIntersect(Ray& ray, bvh::v2::SmallStack<Bvh::Index, stackSize>& stack) {
        constexpr bool isAnyHit = true;

        static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
        auto prim_id = invalid_id;

        bvh.traverse_top_down<isAnyHit>(bvh.get_root().index, stack, 
            [&] (size_t begin, size_t end) {    // leaf
                for (size_t i = begin; i < end; ++i) {
                    size_t j = shouldPermute ? i : bvh.prim_ids[i];
                    if(auto hit = TraverseUtils::SphereTriangleIntersect(ray, tris[j])) {
                        prim_id = i;
                        return true;
                    }
                }
                return false;
            }, 
            [&] (const Node& left, const Node& right) {     // inner
                
                std::pair<Scalar, Scalar> intr_left, intr_right;
                intr_left = TraverseUtils::SphereNodeIntersect(ray, left);
                intr_right = TraverseUtils::SphereNodeIntersect(ray, right);

                // left hit, right hit, swap left right
                return std::make_tuple(
                    intr_left.first <= intr_left.second,
                    intr_right.first <= intr_right.second,
                    !isAnyHit && intr_left.first > intr_right.first);
            });

        return prim_id;
    }

    /// @brief batch call SphereIntersect with many spheres
    /// @param centers sphere centers of many sphere
    /// @param radius sphere radius, all sphere has same radius
    /// @param result intersect result, 1 of intersect else 0
    void BatchSphereIntersect(const std::vector<Vec3>& centers, Scalar radius, std::vector<uint8_t>& result) {

        if (result.size() != centers.size()) {
            result.resize(centers.size());
        }

        // parallel for
        executor.for_each(0, centers.size(), [&] (size_t begin, size_t end) {
            bvh::v2::SmallStack<Bvh::Index, stackSize> stack;
            for (size_t i = begin; i < end; ++i) {
                auto ray = Ray {
                    centers[i],         // Ray origin
                    Vec3(0., 0., 1.),   // Ray direction
                    0.,                 // Minimum intersection distance
                    radius              // Maximum intersection distance
                };
                stack.size = 0;
                auto prim_id = SphereIntersect(ray, stack);
                static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
                result[i] = prim_id != invalid_id ? 1 : 0;
            }
        });
    }
};

}  // namespace next_gsp

int main() {

    using namespace next_gsp;

    // load mesh
    RenderMesh mesh;
    std::vector<Tri> tris = mesh.loadBin("D:/data/dataset/obj/torus_knot_sparse_1m.bin").getTris();

    // build accel
    Accel accel;
    accel.buildAccel(tris);

    // sphere test, 1.0 always not found, 1.00001 always found
    auto O = tris[0].p0 + 0.299999f * mesh.getNormal(0, 0);  // offset along normal
    auto ray = Ray {
        O,                // Ray origin
        Vec3(0., 0., 1.), // Ray direction
        0.,               // Minimum intersection distance
        0.3               // Maximum intersection distance
    };
    auto prim_id = accel.SphereIntersect(ray, accel.sharedStack);
    
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


    {
        // ray test,  1.0 always found, 0.999 not found
        O = tris[0].p0 - 0.299999f * mesh.getNormal(0, 0);  // offset along normal
        auto Target = tris[0].p0 + 0.0f * mesh.getNormal(0, 0);
        ray = Ray {
            O,                // Ray origin
            Target - O,       // Ray direction
            0.,               // Minimum intersection distance
            0.999f            // Maximum intersection distance
        };
        prim_id = accel.RayIntersect(ray, accel.sharedStack);
        
        // print result
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
    }

    {
        // ray test,  1.0 always found, 0.999 not found
        O = tris[0].p0 - 0.299999f * mesh.getNormal(0, 0);  // offset along normal
        auto Target = tris[0].p0 + 0.0f * mesh.getNormal(0, 0);
        ray = Ray {
            O,                // Ray origin
            Target - O,       // Ray direction
            0.,               // Minimum intersection distance
            1.001f            // Maximum intersection distance
        };
        prim_id = accel.RayIntersect(ray, accel.sharedStack);
        
        // print result
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
    }



    // batch ray test
    {   
        constexpr float offset = 100.0;

        size_t nnode = mesh.vertices.size() / 3;
        std::vector<Vec3> origins(nnode);
        std::vector<Vec3> directions(nnode);
        std::vector<Vec3> targets(nnode);
        for (size_t i = 0; i < nnode; i++){
            Vec3 point = Vec3(mesh.vertices[i * 3 + 0], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2]);
            Vec3 normal = Vec3(mesh.normals[i * 3 + 0], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]);
            normal = bvh::v2::normalize(normal);
            origins[i] = point;
            directions[i] = normal;
            targets[i] = point + offset * normal;
        }
        std::vector<uint8_t> ret(nnode);
        auto start = std::chrono::high_resolution_clock::now();
        accel.BatchRayIntersect(origins, directions, 0.01, 100.0, ret);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "\n\nray hit time:  " << duration / 1000.0 << " ms\n";


        std::vector<float> lines_nohit;
        std::vector<float> lines_hit;
        for (size_t i = 0; i < std::min((size_t)1000, ret.size()); i++) {
            if (ret[i] < 0.5) {
                lines_nohit.push_back( origins[i][0] );
                lines_nohit.push_back( origins[i][1] );
                lines_nohit.push_back( origins[i][2] );

                lines_nohit.push_back( targets[i][0] );
                lines_nohit.push_back( targets[i][1] );
                lines_nohit.push_back( targets[i][2] );
            } else {
                lines_hit.push_back( origins[i][0] );
                lines_hit.push_back( origins[i][1] );
                lines_hit.push_back( origins[i][2] );

                lines_hit.push_back( targets[i][0] );
                lines_hit.push_back( targets[i][1] );
                lines_hit.push_back( targets[i][2] );
            }
        }
        RenderMesh::saveToLine("./line_nohit2.obj", lines_nohit);
        RenderMesh::saveToLine("./line_hit2.obj", lines_hit);
    }

    // batch knn test
    {   
        constexpr float offset = 0.1;
        constexpr float threshold = 0.101;

        size_t nnode = mesh.vertices.size() / 3;
        std::vector<Vec3> origins(nnode);
        std::vector<Vec3> directions(nnode);
        std::vector<Vec3> targets(nnode);
        for (size_t i = 0; i < nnode; i++){
            Vec3 point = Vec3(mesh.vertices[i * 3 + 0], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2]);
            Vec3 normal = Vec3(mesh.normals[i * 3 + 0], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]);
            normal = bvh::v2::normalize(normal);
            origins[i] = point;
            directions[i] = normal;
            targets[i] = point + offset * normal;
        }
        std::vector<uint8_t> ret(nnode);
        auto start = std::chrono::high_resolution_clock::now();
        accel.BatchSphereIntersect(targets, threshold, ret);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "\n\nknn time:  " << duration / 1000.0 << " ms\n";


        size_t hit_count = 0;
        size_t nohit_count = 0;
        for (size_t i = 0; i < ret.size(); i++) {
            if (ret[i] < 0.5) {
                nohit_count++;
            } else {
                hit_count++;
            }
        }
        std::cout << "should all knn\n";
        std::cout << "no knn count: " << nohit_count << "\n";
        std::cout << "has knn count: " << hit_count << "\n";
    }

    // batch knn test
    {   
        constexpr float offset = 0.1;
        constexpr float threshold = 0.0999;

        size_t nnode = mesh.vertices.size() / 3;
        std::vector<Vec3> origins(nnode);
        std::vector<Vec3> directions(nnode);
        std::vector<Vec3> targets(nnode);
        for (size_t i = 0; i < nnode; i++){
            Vec3 point = Vec3(mesh.vertices[i * 3 + 0], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2]);
            Vec3 normal = Vec3(mesh.normals[i * 3 + 0], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]);
            normal = bvh::v2::normalize(normal);
            origins[i] = point;
            directions[i] = normal;
            targets[i] = point + offset * normal;
        }
        std::vector<uint8_t> ret(nnode);
        auto start = std::chrono::high_resolution_clock::now();
        accel.BatchSphereIntersect(targets, threshold, ret);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "\n\nknn time:  " << duration / 1000.0 << " ms\n";


        size_t hit_count = 0;
        size_t nohit_count = 0;
        for (size_t i = 0; i < ret.size(); i++) {
            if (ret[i] < 0.5) {
                nohit_count++;
            } else {
                hit_count++;
            }
        }
        std::cout << "should no knn\n";
        std::cout << "no knn count: " << nohit_count << "\n";
        std::cout << "has knn count: " << hit_count << "\n";
    }

    return 0;
}
