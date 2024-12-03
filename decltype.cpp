#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>

[[maybe_unused]] static constexpr double             global { M_PI };

[[maybe_unused]] constexpr decltype(global)          pi { global };
// decltype() does not accept types only identifiers
// eg. decltype(float) is invalid

// we can useb function type casts (in fact, any valid C++ expression will be accepted) inside decltype()

[[maybe_unused]] const decltype(void(global))* const _ptr      = nullptr;

constexpr auto                                       is_int    = std::is_same_v<decltype(12 + 7 * 97), int>;
constexpr auto                                       is_double = std::is_same_v<decltype(M_PI + 7 * 97), double>;

constexpr decltype(float(), short())                 what {};

// this is a new garbage added in C++17 for what?
template<typename...> using erased_t                                                                                  = void;
// using this we can define an single alias for a template parameter pack
// but WHY? because it's fucking C++

[[maybe_unused]] const ::erased_t<float, short, char&, float*, const int&&, volatile double*>* const ptr              = nullptr;

static bool                                                                                          are_you_okay_bro = false;

static_assert(std::is_same_v<decltype(are_you_okay_bro = true), bool&>);
// an assignment operation's return type is the left operand's type's reference
static_assert(std::is_same_v<decltype(are_you_okay_bro), bool>); // that's just bool, see!

static_assert(std::is_same_v<decltype((global)), const double&>);
// using a parenthesis inside decltype() turns the identifier into an expression
// thus it evaluates to a lvalue reference
