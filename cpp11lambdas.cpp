// C++11 lambda syntax is
// [closure captures](arguments) specifiers -> return type { function body };

// the simplest form of lambda
const auto bare_minimum    = [] {};                        // argument paranthesis is not mandatory

const auto add_10          = [](int x) { return x + 10; }; // remember the trailing return type is optional
// the above is equivalent to
const auto add_10_explicit = [](int x) -> int { return x + 10; };

// noexcept qualified lambda
const auto increment       = [](int& x) noexcept -> void { x++; };

// constexpr lambdas
const auto decrement       = [](int& x) constexpr noexcept -> void { --x; };
const auto decrement2      = [](int& x) consteval noexcept -> void { x -= 2; };
