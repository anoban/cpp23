// practicing agent based modelling

#include <iostream>
#include <random>
#include <ranges>
#include <vector>

// let's model how a rumour/gossip might spread in a population

class person final {
    private:
        std::wstring _rumour;
        bool         _has_rumour;

    public:
        person() noexcept : _rumour {}, _has_rumour {} { }

        explicit person(const wchar_t* const _string) noexcept : _rumour { _string }, _has_rumour { true } { }

        explicit person(const std::wstring& _string) noexcept : _rumour { _string }, _has_rumour { true } { }

        void listen(const person& _other) noexcept {
            if (_other._has_rumour) { // if the other person has a rumour, listen to it
                _rumour     = _other._rumour;
                _has_rumour = true;
            }
        }

        bool has_rumour() const noexcept { return _has_rumour; }

        person(const person&)            = default;
        person(person&&)                 = default;
        person& operator=(const person&) = default;
        person& operator=(person&&)      = default;
        ~person() noexcept               = default;
};

static constexpr unsigned long population_size { 8'000'000 }, max_iterations { 30'000 };

auto wmain() -> int {
    std::mt19937_64                         rengine { std::random_device {}() };
    std::uniform_int_distribution<unsigned> randint { 0, population_size - 1 };
    const person                            dumbass { L"There are aliens in area 51, my brother's friend in CIA told me!!" };

    std::vector<person> population(population_size);
    population.at(0).listen(dumbass); // the first point of contact

    // simulate subsequent contacts
    unsigned random_selection {};
    for (const auto& _ : std::ranges::views::iota(0U, max_iterations)) {
        //
        random_selection = randint(rengine);
    }

    return EXIT_SUCCESS;
}
