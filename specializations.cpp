namespace type_traits {

    // THESE ARE PARTIAL SPECIALIZATIONS NOT EXPLICIT SPECIALIZATIONS!
    // instead of specializing for concrete types we specialize on variants of the template parameter _Ty

    template<class _Ty> struct is_reference final {
            static constexpr bool value = false;
    };

    template<class _Ty> struct is_reference<_Ty&> final {
            static constexpr bool value = true;
    };

    template<class _Ty> struct is_reference<_Ty&&> final {
            static constexpr bool value = true;
    };

} // namespace type_traits

template<class _Ty> inline constexpr bool is_reference_v = type_traits::is_reference<_Ty>::value;
