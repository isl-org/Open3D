#include <open3d/core/Scalar.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

namespace open3d {
namespace core {

/***************************
 * nested_initializer_list *
 ***************************/

template <class T, std::size_t I>
struct nested_initializer_list {
    using type = std::initializer_list<
            typename nested_initializer_list<T, I - 1>::type>;
};

template <class T>
struct nested_initializer_list<T, 0> {
    using type = T;
};

template <class T, std::size_t I>
using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;

template <class T, class S>
void nested_copy(T&& iter, const S& s) {
    *iter++ = s;
}

template <class T, class S>
void nested_copy(T&& iter, std::initializer_list<S> s) {
    for (auto it = s.begin(); it != s.end(); ++it) {
        nested_copy(std::forward<T>(iter), *it);
    }
}

template <class U>
struct initializer_depth_impl {
    static constexpr std::size_t value = 0;
};

template <class T>
struct initializer_depth_impl<std::initializer_list<T>> {
    static constexpr std::size_t value = 1 + initializer_depth_impl<T>::value;
};

template <class U>
struct initializer_dimension {
    static constexpr std::size_t value = initializer_depth_impl<U>::value;
};

template <std::size_t I>
struct initializer_shape_impl {
    template <class T>
    static constexpr std::size_t value(T t) {
        if (t.size() == 0) {
            return 0;
        }
        size_t dim = initializer_shape_impl<I - 1>::value(*t.begin());
        for (auto it = t.begin(); it != t.end(); ++it) {
            if (dim != initializer_shape_impl<I - 1>::value(*it)) {
                utility::LogError(
                        "Input contains ragged nested sequences"
                        "(nested lists with unequal sizes or shapes).");
            }
        }
        return dim;
    }
};

template <>
struct initializer_shape_impl<0> {
    template <class T>
    static constexpr std::size_t value(T t) {
        return t.size();
    }
};

template <class R, class U, std::size_t... I>
constexpr R initializer_shape(U t, std::index_sequence<I...>) {
    using size_type = typename R::value_type;
    return {size_type(initializer_shape_impl<I>::value(t))...};
}

template <class R, class T>
constexpr R shape(T t) {
    return initializer_shape<R, decltype(t)>(
            t, std::make_index_sequence<
                       initializer_dimension<decltype(t)>::value>());
}

}  // namespace core
}  // namespace open3d
