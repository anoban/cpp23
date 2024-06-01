#include <algorithm>
#include <iostream>
#include <numbers>
#include <random>
#include <ranges>
#include <vector>

constexpr auto nelems { 100u };

template<typename scalar_t, typename char_t, typename = std::enable_if<std::is_arithmetic<scalar_t>::value, bool>::type>
typename std::enable_if<std::is_same_v<char, char_t> || std::is_same_v<wchar_t, char_t>, std::basic_ostream<char_t>&>::type operator<<(
    std::basic_ostream<char_t>& ostream, const std::vector<scalar_t>& vector
) {
    ostream << char_t('{') << char_t(' ');
    for (typename std::vector<scalar_t>::const_iterator it = vector.cbegin(), end = vector.cend(); it != end; ++it)
        ostream << *it << char_t(',') << char_t(' ');
    ostream << char_t('\b') << char_t('\b') << char_t(' ') << char_t('}') << char_t('\n');
    return ostream;
}

int wmain() {
    auto rdevice { std::random_device {} };
    auto rngine { std::knuth_b { rdevice() } };

    auto empty { std::vector<int> {} };
    auto randoms { std::vector<int>(nelems) };
    std::generate(randoms.begin(), randoms.end(), rngine);

    std::wcout << L"empty :: size = " << empty.size() << L" capacity = " << empty.capacity() << L'\n';
    std::wcout << L"randoms :: size = " << randoms.size() << L" capacity = " << randoms.capacity() << L'\n';

    const auto* ptr { randoms.begin()._Unwrapped() };
    const auto* _ptr { randoms.data() };
    const auto* _eptr { empty.data() };

    empty.swap(randoms);
    std::wcout << L"empty :: size = " << empty.size() << L" capacity = " << empty.capacity() << L'\n';
    std::wcout << L"randoms :: size = " << randoms.size() << L" capacity = " << randoms.capacity() << L'\n';

    const auto* postswap_ptr { empty.begin()._Unwrapped() };
    std::wcout << std::hex << std::uppercase << L" ptr = " << ptr << L" _ptr = " << _ptr << L" postswap_ptr = " << postswap_ptr
               << std::endl;
    std::wcout << L" _eptr = " << _eptr << L" post-swap randoms.data() = " << randoms.data() << L'\n';

    std::vector<float> fvec(200);

    std::wcout << fvec << L"fvec.size = " << fvec.size() << L'\n';

    unsigned     nreallocs {};
    const float* _fptr {};
    for (const auto& _ : std::ranges::views::iota(0, 100)) {
        _fptr = fvec.cbegin()._Unwrapped();
        fvec.insert(fvec.begin() + 1, { std::numbers::pi_v<float> });
        if (_fptr != fvec.data()) nreallocs++;
    }

    std::wcout << nreallocs << L" reallocations has happened!\n";
    std::wcout << fvec << L"fvec.size = " << std::dec << fvec.size() << L'\n';
    std::wcout << empty;

    std::vector<float> cpy { fvec.cbegin() + 101, fvec.cend() };
    std::wcerr << cpy;

    return EXIT_SUCCESS;
}
