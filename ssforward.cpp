#define __SSTRING_PRINT_METHOD_SIGNATURES_ON_CALL__
#include <sstring>
#include <utility>
#include <vector>

class member final {
    private:
        ::sstring first_name;
        ::sstring last_name;
        unsigned  age;
        bool      marital_status;
        double    annual_income;

    public:
        template<typename TString_0, typename TString_1> inline member(
            TString_0&& fname, TString_1&& lname, const unsigned& _age, const bool& _is_married, const double& _salary
        ) noexcept(std::is_nothrow_move_constructible_v<std::remove_cvref_t<TString_0>> && std::is_nothrow_copy_constructible_v<std::remove_cvref_t<TString_0>>) :
            first_name(std::forward<TString_0>(fname)),
            last_name(std::forward<TString_1>(lname)),
            age(_age),
            marital_status(_is_married),
            annual_income(_salary) { }

        //   template<typename... TList> inline member(TList&&... args) noexcept : member(std::forward<TList...>(args...)) { }
};

auto main() -> int {
    ::sstring             Jennifer { "Jennifer" };
    const member          Julia { "Julia", "Smithson", 32, false, 450000 };
    const member          Roberts { std::move(Jennifer), "Ivanovick", 29, true, 589200 };

    std::vector<::member> members {};
    members.emplace_back("Nathan", "Williams", 47, false, 78897.56497);
    members.emplace_back(false, "Williams", 47, false, 78897.56497); // should err

    return EXIT_SUCCESS;
}
