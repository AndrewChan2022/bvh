
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <bvh/v2/vec.h>
#include <bvh/v2/tri.h>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;

struct RenderMesh {

    std::vector<float> vertices;   // xyz
    std::vector<float> normals;    // xyz grad direction, normalized

    std::vector<uint32_t> indices;

    RenderMesh& loadBin(const std::string& filename) {
        RenderMesh& mesh = *this;

        // Open file in binary mode
        std::ifstream inFile(filename, std::ios::binary);
        if (!inFile.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        // Read vertex count and index count
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        inFile.read(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
        inFile.read(reinterpret_cast<char*>(&indexCount), sizeof(indexCount));

        // Resize vectors
        mesh.vertices.resize(vertexCount * 3); // 3 floats per vertex
        mesh.normals.resize(vertexCount * 3);  // 3 floats per normal
        mesh.indices.resize(indexCount);       // 1 uint32_t per index

        // Read vertex data
        inFile.read(reinterpret_cast<char*>(mesh.vertices.data()), mesh.vertices.size() * sizeof(float));

        // Read normal data
        inFile.read(reinterpret_cast<char*>(mesh.normals.data()), mesh.normals.size() * sizeof(float));

        // Read index data
        inFile.read(reinterpret_cast<char*>(mesh.indices.data()), mesh.indices.size() * sizeof(uint32_t));

        inFile.close();

        return *this;
    }

    std::vector<Tri> getTris() {

        size_t ntriangle = indices.size() / 3;

        std::vector<Tri> tris(ntriangle);

        for (size_t i = 0; i < ntriangle; i++) {
            uint32_t index = indices[i * 3 + 0];
            tris[i].p0 = Vec3(vertices[index * 3 + 0], vertices[index * 3 + 1], vertices[index * 3 + 2]);

            index = indices[i * 3 + 1];
            tris[i].p1 = Vec3(vertices[index * 3 + 0], vertices[index * 3 + 1], vertices[index * 3 + 2]);

            index = indices[i * 3 + 2];
            tris[i].p2 = Vec3(vertices[index * 3 + 0], vertices[index * 3 + 1], vertices[index * 3 + 2]);
        }

        return tris;
    }

    Vec3 getNormal(uint32_t triangle, uint32_t point) {
        uint32_t index = indices[triangle * 3 + point];
        return Vec3(normals[index * 3 + 0], normals[index * 3 + 1], normals[index * 3 + 2]);
    }
};
