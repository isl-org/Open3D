#include <gtest/gtest.h>

#include "Open3D/Registry.h"

// PointCloud.h
class PointCloud {
public:
    PointCloud();
    virtual ~PointCloud() {}
};
OPEN3D_DECLARE_REGISTRY_FOR_CLASS(PointCloudRegistry, PointCloud);

// PointCloud.cc
PointCloud::PointCloud() {}
OPEN3D_DEFINE_REGISTRY_FOR_CLASS(PointCloudRegistry, PointCloud);

// PointCloudCPU.h
class PointCloudCPU : public PointCloud {
public:
    PointCloudCPU();
    ~PointCloudCPU() {}
};

// PointCloudCPU.cc
PointCloudCPU::PointCloudCPU() : PointCloud() {}
OPEN3D_REGISTER_CLASS(PointCloudRegistry, "cpu", PointCloudCPU);

// PointCloudGPU.h, conditionally included
class PointCloudGPU : public PointCloud {
public:
    PointCloudGPU();
    ~PointCloudGPU() {}
};

// PointCloudGPU.cc, conditionally compiled
PointCloudGPU::PointCloudGPU() : PointCloud() {}
OPEN3D_REGISTER_CLASS(PointCloudRegistry, "gpu", PointCloudGPU);

TEST(RegistryForClass, CanRunCreator) {
    std::shared_ptr<PointCloud> pcd_cpu =
            PointCloudRegistry()->GetFactory("cpu")();
    EXPECT_TRUE(pcd_cpu != nullptr);

    std::shared_ptr<PointCloud> pcd_gpu =
            PointCloudRegistry()->GetFactory("gpu")();
    EXPECT_TRUE(pcd_cpu != nullptr);

    EXPECT_TRUE(PointCloudRegistry()->GetFactory("tpu") == nullptr);
}

class Foo {
public:
    explicit Foo(int x) {}
    explicit Foo(int x, int y) {}
    virtual ~Foo() {}
};

OPEN3D_DECLARE_REGISTRY_FOR_CLASS(FooRegistry, Foo, int);
OPEN3D_DEFINE_REGISTRY_FOR_CLASS(FooRegistry, Foo, int);

OPEN3D_DECLARE_REGISTRY_FOR_CLASS(FooRegistryTwoArg, Foo, int, int);
OPEN3D_DEFINE_REGISTRY_FOR_CLASS(FooRegistryTwoArg, Foo, int, int);
OPEN3D_REGISTER_CLASS(FooRegistryTwoArg, "Foo", Foo);

class Bar : public Foo {
public:
    explicit Bar(int x) : Foo(x) {}
};
OPEN3D_REGISTER_CLASS(FooRegistry, "Bar", Bar)

class AnotherBar : public Foo {
public:
    explicit AnotherBar(int x) : Foo(x) {}
};
OPEN3D_REGISTER_CLASS(FooRegistry, "AnotherBar", AnotherBar);

TEST(RegistryForClass, CanRunCreatorWithArgs) {
    std::shared_ptr<Foo> foo_two_args =
            FooRegistryTwoArg()->GetFactory("Foo")(123, 456);
    EXPECT_TRUE(foo_two_args != nullptr) << "Cannot create Foo with two args.";

    std::shared_ptr<Foo> bar = FooRegistry()->GetFactory("Bar")(123);
    EXPECT_TRUE(bar != nullptr) << "Cannot create bar.";

    std::shared_ptr<Foo> another_bar =
            FooRegistry()->GetFactory("AnotherBar")(456);
    EXPECT_TRUE(another_bar != nullptr) << "Cannot create another_bar.";
}

TEST(RegistryForClass, ReturnNullOnNonExistingCreatorWithArgs) {
    EXPECT_EQ(FooRegistry()->GetFactory("Non-existing bar"), nullptr);
}

// registration.h
int RegistrationICP(int i, int j, const std::string& device);
OPEN3D_DECLARE_REGISTRY_FOR_FUNCTION(RegistrationICPRegistry,
                                     std::function<int(int, int)>);

// registration.cc
int RegistrationICP(int i, int j, const std::string& device) {
    return RegistrationICPRegistry()->GetFunction(device)(i, j);
}
OPEN3D_DEFINE_REGISTRY_FOR_FUNCTION(RegistrationICPRegistry,
                                    std::function<int(int, int)>);

// registration_cpu.h
int RegistrationICPCPU(int i, int j);

// registration_cpu.cc
int RegistrationICPCPU(int i, int j) { return i + j; }
OPEN3D_REGISTER_FUNCTION(RegistrationICPRegistry, "cpu", RegistrationICPCPU);

// registration_gpu.h, conditionally included
int RegistrationICPGPU(int i, int j);

// registration_gpu.cc, conditionally compiled
int RegistrationICPGPU(int i, int j) { return i + j + 1; }
OPEN3D_REGISTER_FUNCTION(RegistrationICPRegistry, "gpu", RegistrationICPGPU);

TEST(RegistryForFunction, CanRunCreator) {
    EXPECT_EQ(RegistrationICPRegistry()->GetFunction("cpu")(1, 2), 3);
    EXPECT_EQ(RegistrationICPRegistry()->GetFunction("gpu")(1, 2), 4);
    EXPECT_EQ(RegistrationICPRegistry()->GetFunction("tpu"), nullptr);
}
