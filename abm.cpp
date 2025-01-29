// practicing agent based modelling https://caam37830.github.io/book/09_computing/agent_based_models.html

#include <iostream>
#include <numeric>
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

        void converse(const person& _other) noexcept {
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

        // person + person
        unsigned long long operator+(const person& _other) const noexcept { return _has_rumour + _other._has_rumour; }

        // person + value
        unsigned long long operator+(const unsigned long long& _sum) const noexcept { return _has_rumour + _sum; }

        // value + person
        friend constexpr unsigned long long operator+(const unsigned long long _sum, const person& _other) noexcept {
            return _sum + _other._has_rumour;
        }
};

static constexpr unsigned long population_size { 8'000 }, max_iterations { 30'000 }, max_days { 1'000 }, max_contacts { 18 };

auto wmain() -> int {
    std::mt19937_64                         rengine { std::random_device {}() };
    std::uniform_int_distribution<unsigned> randint { 0, population_size - 1 };
    const person                            dumbass { L"There are aliens in area 51, my brother's friend in CIA told me!!" };

    std::vector<person>        population(population_size);
    std::vector<unsigned char> daily_changes(max_days);

    population.at(0).converse(dumbass); // the first point of contact

    // simulate subsequent contacts
    unsigned random_selection {}, contacts {}; // NOLINT(readability-isolate-declaration)
    std::wcout << L"population size :: " << population_size << L'\n';
    for (const auto& d : std::ranges::views::iota(0U, max_days)) {
        for (const auto& _ : std::ranges::views::iota(0U, max_iterations)) {
            random_selection = randint(rengine);
            for (const auto& _ : std::ranges::views::iota(0U, max_contacts)) {
                // make contact
                contacts = randint(rengine);
                population.at(random_selection).converse(population.at(contacts));
            }
        }

        daily_changes.at(d) = std::accumulate(population.cbegin(), population.cend(), 0LU);

        std::wcout << L"fraction of people who knew the rumour at day " << d + 1 << L" is "
                   << daily_changes.at(d) / static_cast<double>(population_size) << L'\n';
    }

    return EXIT_SUCCESS;
}
