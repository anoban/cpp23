#include <memory>

class buffer {
    private:
        uint8_t* _resource;
        size_t   _size;
        size_t   _capacity;

    public:
        inline buffer() throw() : _resource(), _size(), _capacity() { }

        inline explicit buffer(const size_t& size) throw() : _resource(new (std::nothrow) uint8_t[size]), _size(size), _capacity(size) {
            if (!_resource) { // if the allocation failed,
                _capacity = _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
            }
        }

        inline buffer(const buffer& other) throw() :
            _resource(new (std::nothrow) uint8_t[other._size]), _size(other._size), _capacity(other._capacity) {
            if (!_resource) { // if the allocation failed,
                _capacity = _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
                return;
            }

            ::memcpy_s(_resource, _size, other._resource, other._size);
        }

        inline buffer& operator=(const buffer& other) throw() {
            if (this == &other) return *this;

            if (_size == other._size) { // no need for new allocations
                ::memcpy_s(_resource, _size, other._resource, other._size);
                return *this;
            }

            if (_size > other._size) {          // no need for new allocations
                ::memset(_resource, 0U, _size); // since we'll have trailing garbage bytes
                ::memcpy_s(_resource, _size, other._resource, other._size);
                return *this;
            }

            delete[] _resource;                                  // give up the old buffer
            _resource = new (std::nothrow) uint8_t[other._size]; // allocate a new buffer
            _size     = other._size;

            if (!_resource) { // if the allocation failed,
                _size = 0;
                ::fputws(L"allocation failure!\n", stderr);
                return *this;
            }
            ::memcpy_s(_resource, _size, other._resource, other._size);
            return *this;
        }

#if __cplusplus >= 201103L
        inline buffer(buffer&& other) throw() : _resource(other._resource), _size(other._size) {
            other._resource = NULL;
            other._size     = 0;
        }
#endif

        inline uint8_t* data() throw() { return _resource; }

        inline const uint8_t* data() const throw() { return _resource; }

        inline size_t length() const throw() { return _size; }
};

int wmain() {
    //
    return EXIT_SUCCESS;
}
