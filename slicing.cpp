#include <string>
#include <type_traits>

class user {
        std::wstring _name;
        unsigned     _age;
        unsigned     _total_purchases;

    public:
        inline user(std::wstring&& name, unsigned age, unsigned purchases) noexcept :
            _name(std::move(name)), _age(age), _total_purchases(purchases) { }
};

class privileged_user : public user {
        float _loyalty_rewards;
        float _base_discount;

    public:
        inline privileged_user(std::wstring name, unsigned age, unsigned purchases) noexcept :
            user(std::move(name), age, purchases), _loyalty_rewards(12.75), _base_discount(3.89) { }
};

static_assert(std::is_standard_layout_v<user>);
static_assert(!std::is_standard_layout_v<privileged_user>);

auto wmain() -> int {
    const auto james {
        user { L"James", 37, 643 }
    };

    const auto natalie {
        privileged_user { L"Natalie", 29, 741 }
    };

    // slicing
    [[maybe_unused]] const user sliced_user { natalie }; // slicing object from type 'privileged_user' to 'user' discards 8 bytes of state
}
