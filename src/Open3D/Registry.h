#pragma once

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "Open3D/Macro.h"

namespace open3d {

// Registry for classes that have both CPU and GPU version
// Ref: https://github.com/pytorch/pytorch/blob/master/c10/util/Registry.h
template <class ObjectPtrType, class... Args>
class RegistryForClass {
public:
    typedef std::function<ObjectPtrType(Args...)> Factory;

    RegistryForClass() : registry_() {}

    void Register(const std::string& key, Factory factory) {
        std::lock_guard<std::mutex> lock(register_mutex_);
        if (registry_.count(key) == 0) {
            registry_[key] = factory;
        }
    }

    inline bool Has(const std::string& key) {
        return (registry_.count(key) != 0);
    }

    Factory GetFactory(const std::string& key) {
        if (registry_.count(key) == 0) {
            // Returns nullptr if the key is not registered.
            return nullptr;
        }
        return registry_[key];
    }

    // Returns the keys currently registered as a std::vector.
    std::vector<std::string> Keys() const {
        std::vector<std::string> keys;
        for (const auto& it : registry_) {
            keys.push_back(it.first);
        }
        return keys;
    }

private:
    std::unordered_map<std::string, Factory> registry_;
    std::mutex register_mutex_;

    DISABLE_COPY_AND_ASSIGN(RegistryForClass);
};

template <class ObjectPtrType, class... Args>
class RegistererForClass {
public:
    explicit RegistererForClass(
            const std::string& key,
            const std::shared_ptr<RegistryForClass<ObjectPtrType, Args...>>&
                    registry,
            typename RegistryForClass<ObjectPtrType, Args...>::Factory
                    factory) {
        registry->Register(key, factory);
    }

    template <class DerivedType>
    static ObjectPtrType DefaultCreator(Args... args) {
        return ObjectPtrType(new DerivedType(args...));
    }
};

// Registry for functions that have both CPU and GPU version
template <class FunctionType>
class RegistryForFunction {
public:
    RegistryForFunction() : registry_() {}

    void Register(const std::string& key, FunctionType func) {
        std::lock_guard<std::mutex> lock(register_mutex_);
        if (registry_.count(key) == 0) {
            registry_[key] = func;
        }
    }

    inline bool Has(const std::string& key) {
        return (registry_.count(key) != 0);
    }

    FunctionType GetFunction(const std::string& key) {
        if (registry_.count(key) == 0) {
            // Returns nullptr if the key is not registered.
            return nullptr;
        }
        return registry_[key];
    }

    // Returns the keys currently registered as a std::vector.
    std::vector<std::string> Keys() const {
        std::vector<std::string> keys;
        for (const auto& it : registry_) {
            keys.push_back(it.first);
        }
        return keys;
    }

private:
    std::unordered_map<std::string, FunctionType> registry_;
    std::mutex register_mutex_;

    DISABLE_COPY_AND_ASSIGN(RegistryForFunction);
};

template <class FunctionType>
class RegistererForFunction {
public:
    explicit RegistererForFunction(
            const std::string& key,
            const std::shared_ptr<RegistryForFunction<FunctionType>>& registry,
            FunctionType func) {
        registry->Register(key, func);
    }
};

// Registry for singleton objects referenced by shared_ptr
// Every registry can register multiple singleton objects index by name
template <class ObjectPtrType>
class RegistryForSingleton {
public:
    RegistryForSingleton() : registry_() {}

    void Register(const std::string& key, ObjectPtrType obj_ptr) {
        std::lock_guard<std::mutex> lock(register_mutex_);
        if (registry_.count(key) == 0) {
            registry_[key] = obj_ptr;
        }
    }

    inline bool Has(const std::string& key) {
        return (registry_.count(key) != 0);
    }

    ObjectPtrType GetSingletonObject(const std::string& key) {
        if (registry_.count(key) == 0) {
            // Returns nullptr if the key is not registered.
            return nullptr;
        }
        return registry_[key];
    }

    // Returns the keys currently registered as a std::vector.
    std::vector<std::string> Keys() const {
        std::vector<std::string> keys;
        for (const auto& it : registry_) {
            keys.push_back(it.first);
        }
        return keys;
    }

private:
    std::unordered_map<std::string, ObjectPtrType> registry_;
    std::mutex register_mutex_;

    DISABLE_COPY_AND_ASSIGN(RegistryForSingleton);
};

template <class ObjectPtrType>
class RegistererForSingleton {
public:
    explicit RegistererForSingleton(
            const std::string& key,
            const std::shared_ptr<RegistryForSingleton<ObjectPtrType>>&
                    registry,
            ObjectPtrType obj_ptr) {
        registry->Register(key, obj_ptr);
    }
};
}  // namespace open3d

#define OPEN3D_DECLARE_REGISTRY_FOR_CLASS(RegistryName, ObjectType, ...) \
    OPEN3D_IMPORT std::shared_ptr<::open3d::RegistryForClass<            \
            std::shared_ptr<ObjectType>, ##__VA_ARGS__>>                 \
    RegistryName();                                                      \
    typedef ::open3d::RegistererForClass<std::shared_ptr<ObjectType>,    \
                                         ##__VA_ARGS__>                  \
            RegistererForClass##RegistryName

#define OPEN3D_DEFINE_REGISTRY_FOR_CLASS(RegistryName, ObjectType, ...)     \
    OPEN3D_EXPORT std::shared_ptr<::open3d::RegistryForClass<               \
            std::shared_ptr<ObjectType>, ##__VA_ARGS__>>                    \
    RegistryName() {                                                        \
        static auto registry = std::make_shared<::open3d::RegistryForClass< \
                std::shared_ptr<ObjectType>, ##__VA_ARGS__>>();             \
        return registry;                                                    \
    }

#define OPEN3D_REGISTER_CLASS(RegistryName, key, ...)                  \
    static RegistererForClass##RegistryName OPEN3D_ANONYMOUS_VARIABLE( \
            g_##RegistryName)(                                         \
            key, RegistryName(),                                       \
            RegistererForClass##RegistryName::DefaultCreator<__VA_ARGS__>);

#define OPEN3D_DECLARE_REGISTRY_FOR_FUNCTION(RegistryName, FunctionType)       \
    OPEN3D_IMPORT std::shared_ptr<::open3d::RegistryForFunction<FunctionType>> \
    RegistryName();                                                            \
    typedef ::open3d::RegistererForFunction<FunctionType>                      \
            RegistererForFunction##RegistryName

#define OPEN3D_DEFINE_REGISTRY_FOR_FUNCTION(RegistryName, FunctionType)        \
    OPEN3D_EXPORT std::shared_ptr<::open3d::RegistryForFunction<FunctionType>> \
    RegistryName() {                                                           \
        static auto registry = std::make_shared<                               \
                ::open3d::RegistryForFunction<FunctionType>>();                \
        return registry;                                                       \
    }

#define OPEN3D_REGISTER_FUNCTION(RegistryName, key, func)                 \
    static RegistererForFunction##RegistryName OPEN3D_ANONYMOUS_VARIABLE( \
            g_##RegistryName)(key, RegistryName(), func);

#define OPEN3D_DECLARE_REGISTRY_FOR_SINGLETON(RegistryName, ObjectPtrType) \
    OPEN3D_IMPORT                                                          \
    std::shared_ptr<::open3d::RegistryForSingleton<ObjectPtrType>>         \
    RegistryName();                                                        \
    typedef ::open3d::RegistererForSingleton<ObjectPtrType>                \
            RegistererForSingleton##RegistryName

#define OPEN3D_DEFINE_REGISTRY_FOR_SINGLETON(RegistryName, ObjectPtrType) \
    OPEN3D_EXPORT                                                         \
    std::shared_ptr<::open3d::RegistryForSingleton<ObjectPtrType>>        \
    RegistryName() {                                                      \
        static auto registry = std::make_shared<                          \
                ::open3d::RegistryForSingleton<ObjectPtrType>>();         \
        return registry;                                                  \
    }

#define OPEN3D_REGISTER_SINGLETON_OBJECT(RegistryName, key, obj_ptr)       \
    static RegistererForSingleton##RegistryName OPEN3D_ANONYMOUS_VARIABLE( \
            g_##RegistryName)(key, RegistryName(), obj_ptr);
