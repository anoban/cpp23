#include <type_traits>

template<class _Ty, typename std::enable_if<std::is_arithmetic<_Ty>::value, bool>::type is_valid = true> class rational final {
    public:
        typedef _Ty value_type; // NOLINT(modernize-use-using) because I'm feeling nostalgic
        // typedef _Ty*       pointer;
        // typedef const _Ty* const_pointer;
        // typedef _Ty&       reference;
        // typedef const _Ty& const_reference;

    private:
        _Ty numerator, denominator;

    public:
};
