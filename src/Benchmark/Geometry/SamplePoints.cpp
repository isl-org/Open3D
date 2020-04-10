#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "benchmark/benchmark.h"

class SamplePointsFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) {
        trimesh = open3d::io::CreateMeshFromFile(TEST_DATA_DIR "/knot.ply");
    }

    void TearDown(const benchmark::State& state) {
        // empty
    }
    std::shared_ptr<open3d::geometry::TriangleMesh> trimesh;
};

BENCHMARK_DEFINE_F(SamplePointsFixture, Poisson)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->SamplePointsPoissonDisk(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Poisson)->Args({123})->Arg({1000});

BENCHMARK_DEFINE_F(SamplePointsFixture, Uniform)(benchmark::State& state) {
    for (auto _ : state) {
        trimesh->SamplePointsUniformly(state.range(0));
    }
}

BENCHMARK_REGISTER_F(SamplePointsFixture, Uniform)->Args({123})->Args({1000});
