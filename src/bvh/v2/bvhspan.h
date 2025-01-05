#pragma once

namespace bvh::v2 {

template <typename T>
struct Span {
    T* data;  // Pointer to the data
    size_t size_;  // Size of the data

    // Constructor
    Span(T* data, size_t size) : data(data), size_(size) {}
    Span(std::vector<T>& vec) : data(vec.data()), size_(vec.size()) {}
    
    // Constructor for std::vector<T> when T is not const
    template <typename U = T, typename = std::enable_if_t<!std::is_const<U>::value>>
    Span(const std::vector<U>& vec)
        : data(vec.data()), size_(vec.size()) {}

    // SFINAE: Constructor for Span<const U> from std::vector<U>
    // This constructor is used when T is const U
    template <typename U = typename std::remove_const<T>::type, typename = std::enable_if_t<std::is_const<T>::value>>
    Span(std::vector<U>& vec, typename std::enable_if<std::is_const<T>::value>::type* = nullptr)
        : data(vec.data()), size_(vec.size()) {}


    // Accessor for size_
    size_t length() const { return size_; }

    rsize_t size() const { return size_;  }

    // Accessor for data
    T* begin() { return data; }
    const T* begin() const { return data; }

    T* end() { return data + size_; }
    const T* end() const { return data + size_; }

    // Index operator for accessing elements
    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    // Utility functions (optional, for convenience)
    bool empty() const { return size_ == 0; }

    // Example method: Copy the span into a new array
    void copy_to(T* destination) const {
        std::copy(data, data + size_, destination);
    }
};

}