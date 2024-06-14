#include <cassert>
#include <cstdint>
#include <iterator>
#include <type_traits>

template<typename T> class random_access_iterator final {
    public:
        using value_type        = typename std::remove_cv<T>::type;
        using pointer           = value_type*;
        using const_pointer     = const value_type*;
        using reference         = value_type&;
        using const_reference   = const value_type&;
        using difference_type   = signed long long;   // aka ptrdiff_t
        using size_type         = unsigned long long; // aka size_t
        using iterator_category = std::random_access_iterator_tag;

    private:
        pointer   _rsrc;   // pointer to the iterable resource
        size_type _length; // number of elements in the iterable
        size_type _offset; // current position in the iterable

    public:
        // will require an explicit template type specification
        constexpr inline random_access_iterator() noexcept : _rsrc(), _length(), _offset() { }

        // resource pointer and length of the iterable
        constexpr inline random_access_iterator(_In_ const T& _res, _In_ const size_t& _sz) noexcept :
            _rsrc(_res), _length(_sz), _offset() { }

        // resource pointer, length of the iterable and the current iterator position
        constexpr inline random_access_iterator(_In_ const T& _res, _In_ const size_t& _sz, _In_ const size_t& _pos) noexcept :
            _rsrc(_res), _length(_sz), _offset(_pos) {
            assert(_sz >= _pos);
        }

        constexpr inline random_access_iterator(_In_ const random_access_iterator& _other) noexcept :
            _rsrc(_other._rsrc), _length(_other._length), _offset(_other._offset) { }

        constexpr inline random_access_iterator(_In_ random_access_iterator&& _other) noexcept :
            _rsrc(_other._rsrc), _length(_other._length), _offset(_other._offset) {
            // cleanup the stolen from resource
            _other._rsrc   = nullptr;
            _other._length = _other._offset = 0;
        }
};

namespace idxio { // we will not be using exceptions here! caller will have to manually examine the returned class type for errors

    class idx1 {
        public:

        private:
            uint32_t _idxmagic;
            uint32_t _nlabels;
            uint8_t* _labels;

        public:
            constexpr inline idx1() noexcept; // default ctor

            constexpr inline explicit idx1(_In_ const wchar_t* const _path) noexcept; // construct from a file path

            constexpr inline explicit idx1(
                _In_ const uint8_t* const _buffer, _In_ const size_t& _sz
            ) noexcept; // construct from a byte buffer

            constexpr inline idx1(_In_ const idx1& _other) noexcept; // copy ctor

            constexpr inline idx1(_In_ idx1&& _other) noexcept; // move ctor

            constexpr inline idx1& operator=(const idx1& _other) noexcept; // copy =

            constexpr inline idx1& operator=(idx1&& _other) noexcept; // move =

            constexpr inline ~idx1() noexcept; // dtor

            constexpr size_t count() const noexcept; // label count
    };

    class idx3 { };
} // namespace idxio
