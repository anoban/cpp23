// IMPOSING TYPE RESTRICTIONS BY SFINAE DIRECTLY WILL LEAD TO INELEGANT COMPILE TIME ERRORS
// E.G.

template<typename _TyLeft, typename _TyRight> struct is_assignable final {
        static constexpr bool value { false };
};
