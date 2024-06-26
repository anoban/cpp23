// templated copy ctors and move ctors are called universal constructors

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numbers>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>

template<typename scalar_t> class random_access_iterator final { // unchecked iterator!
    private:
        scalar_t* _start;
        size_t    _offset;
        size_t    _length;

    public:
        using value_type        = scalar_t;
        using size_type         = size_t;
        using pointer           = scalar_t*;
        using const_pointer     = const scalar_t*;
        using reference         = scalar_t&;
        using const_reference   = const scalar_t&;
        using iterator_category = std::random_access_iterator_tag;
        using difference_type   = ptrdiff_t;

        constexpr random_access_iterator() noexcept : _start(), _offset(), _length() { }

        constexpr random_access_iterator(const pointer& _ptr, const size_type& _len) noexcept : _start(_ptr), _offset(), _length(_len) { }

        constexpr random_access_iterator(const pointer& _ptr, const size_type& _len, const size_type& _off) noexcept :
            _start(_ptr), _offset(_off), _length(_len) { }

        constexpr random_access_iterator(const random_access_iterator& other) noexcept :
            _start(other._start), _offset(other._offset), _length(other._length) { // copy ctor
        }

        constexpr random_access_iterator(random_access_iterator&& other) noexcept :
            _start(other._start), _offset(other._offset), _length(other._length) { // move ctor
        }

        constexpr ~random_access_iterator() noexcept {
            _start  = nullptr;
            _offset = _length = 0;
        }

        constexpr random_access_iterator& operator=(const random_access_iterator& other) noexcept { // copy assignment
            if (this == &other) return *this;
            _start  = other._start;
            _offset = other._offset;
            _length = other._length;
            return *this;
        }

        constexpr random_access_iterator& operator=(random_access_iterator&& other) noexcept { // move assignment
            if (this == &other) return *this;
            _start  = other._start;
            _offset = other._offset;
            _length = other._length;
            return *this;
        }

        constexpr random_access_iterator& operator++() noexcept { // prefix increment
            _offset++;
            return *this;
        }

        constexpr random_access_iterator& operator--() noexcept { // prefix decrement
            _offset--;
            return *this;
        }

        constexpr random_access_iterator operator++(int) noexcept { // postfix increment
            _offset++;
            return { _start, _offset - 1, _length };
        }

        constexpr random_access_iterator operator--(int) noexcept { // postfix decrement
            _offset--;
            return { _start, _offset + 1, _length };
        }

        constexpr bool operator==(const random_access_iterator& other) noexcept {
            return _start == other._start && _offset == other._offset;
        }

        constexpr bool operator!=(const random_access_iterator& other) noexcept {
            return _start != other._start || _offset != other._offset;
        }

        constexpr reference operator*() noexcept { return _start[_offset]; }

        constexpr const_reference operator*() const noexcept { return _start[_offset]; }
};

template<typename scalar_t> requires std::is_arithmetic_v<scalar_t> class vector final {
    private:
        scalar_t* _buffer;
        size_t    _size;

    public:
        // even typedefs and using aliases inside class types obey the private, public and protected accessibility restrictions
        using value_type      = scalar_t;
        using size_type       = size_t;
        using pointer         = scalar_t*;
        using const_pointer   = const scalar_t*;
        using reference       = scalar_t&;
        using const_reference = const scalar_t&;
        using iterator        = ::random_access_iterator<scalar_t>;
        using const_iterator  = ::random_access_iterator<const scalar_t>;

        constexpr static size_t default_size { 100 };

        inline vector() noexcept : _buffer(new (std::nothrow) scalar_t[default_size]), _size(default_size) { } // default ctor

        inline explicit vector(const size_t& _len) noexcept : _buffer(new (std::nothrow) scalar_t[_len]), _size(_len) { } // ctor

        ~vector() noexcept { // dtor
            delete[] _buffer;
            _size = 0;
        }

        inline vector(const vector& other) noexcept : _buffer(new (std::nothrow) scalar_t[other._size]), _size(other._size) { // copy ctor
            std::copy(other._buffer, other._buffer + _size, _buffer);
        }

        inline vector(vector&& other) noexcept : _buffer(std::move(other._buffer)), _size(other._size) {
            other._buffer = nullptr;
            other._size   = 0;
        } // move ctor

        inline vector& operator=(const vector& other) noexcept { // copy assignment
            if (&other == this) return *this;
            // if the existing buffer is long enough
            if (_size >= other._size) {
                std::fill(_buffer, _buffer + _size, 0); // memset 0
                std::copy(other._buffer, other._buffer + other._size, _buffer);
            } else {
                delete[] _buffer;
                _buffer = new (std::nothrow) scalar_t[other._size];
                std::copy(other._buffer, other._buffer + other._size, _buffer);
                _size = other._size;
            }
            return *this;
        }

        inline vector& operator=(vector&& other) noexcept { // move assignment
            if (&other == this) return *this;
            delete[] _buffer;
            _buffer       = other._buffer;
            _size         = other._size;
            other._buffer = nullptr;
            other._size   = 0;
            return *this;
        }

        inline size_type size() const noexcept { return _size; }

        inline pointer data() noexcept { return _buffer; }

        inline const_pointer data() const noexcept { return _buffer; }

        inline iterator begin() noexcept { return { _buffer, _size }; }

        inline const_iterator begin() const noexcept { return { _buffer, _size, 0 }; }

        inline iterator end() noexcept { return { _buffer, _size, _size }; }

        inline const_iterator end() const noexcept { return { _buffer, _size, _size }; }

        inline const_iterator cbegin() const noexcept { return { _buffer, _size, 0 }; }

        inline const_iterator cend() const noexcept { return { _buffer, _size, _size }; }
};

// starting to love C++ :)

auto wmain() -> int {
    auto rdev { std::random_device {} };
    auto rngine { std::knuth_b { rdev() } };

    auto numbers { ::vector<int> { 500'000 } };
    std::generate(numbers.begin(), numbers.end(), rngine);
    const auto sum { std::accumulate(numbers.cbegin(), numbers.cend(), 0.0L) };

    std::wcout << numbers.size() << std::endl;

    const auto randoms { std::move(numbers) };
    const auto rsum { std::accumulate(randoms.cbegin(), randoms.cend(), 0.0L) };

    std::wcout << randoms.size() << std::endl;
    std::wcout << numbers.size() << std::endl;
    std::wcout << sum << L' ' << rsum << std::endl;

    if (!numbers.data()) std::wcout << L"Yes! nullptr\n";

    constexpr ::vector<float>::value_type x { std::numbers::pi_v<decltype(x)> };

    ::vector<double> nums(100);
    std::iota(nums.begin(), nums.end(), 1);
    for (unsigned i {}; const decltype(nums)::value_type& e : nums) std::wcout << L'(' << i++ << L") " << e << L' ';
    for (decltype(nums)::const_iterator it = nums.cbegin(), end = nums.cend(); it != end; ++it) std::wcout << *it << L'\n';

    return EXIT_SUCCESS;
}
