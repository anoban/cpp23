// clang .\motiv03.cpp -Wall -O3 -std=c++20 -Wextra -pedantic

// move semantics before C++11 i.e C++03
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

class object { // let's create an expensive to copy class
    public:
        explicit object(float f32, int64_t i64, uint8_t u8) : nums_f32(100, f32), nums_i64(100, i64), nums_u8(100, u8) { }

        object() : nums_f32(100), nums_i64(100), nums_u8(100) { }

        // copy ctor
        object(const object& other) : nums_f32 { other.nums_f32 }, nums_i64 { other.nums_i64 }, nums_u8 { other.nums_u8 } {
            std::wcout << L"object is being copied!\n";
        }

        // move ctor
        object(const object&& xvalue) noexcept : nums_f32 { xvalue.nums_f32 }, nums_i64 { xvalue.nums_i64 }, nums_u8 { xvalue.nums_u8 } {
            std::wcout << L"object is being moved!\n";
        }

        // defaulted dtor
        ~object() = default;

        [[nodiscard]] object operator+(const object& other) const {
            std::wcout << L"new object is being created!\n";
            auto result { object {} };
            std::transform(nums_f32.cbegin(), nums_f32.cend(), other.nums_f32.cbegin(), result.nums_f32.begin(), std::plus<float> {});
            std::transform(nums_i64.cbegin(), nums_i64.cend(), other.nums_i64.cbegin(), result.nums_i64.begin(), std::plus<int64_t> {});
            std::transform(nums_u8.cbegin(), nums_u8.cend(), other.nums_u8.cbegin(), result.nums_u8.begin(), std::plus<uint8_t> {});
            return result;
        }

        // copy assignment
        object& operator=(const object& other) {
            std::wcout << L"object is being copy assigned!\n";
            nums_f32 = other.nums_f32;
            nums_i64 = other.nums_i64;
            nums_u8  = other.nums_u8;
            return *this;
        }

        friend std::wostream& operator<<(std::wostream& wistr, const object& obj) {
            for (const auto& f32 : obj.nums_f32) wistr << f32 << L' ';
            wistr << L'\n';
            for (const auto& i64 : obj.nums_i64) wistr << i64 << L' ';
            wistr << L'\n';
            for (const auto& u8 : obj.nums_u8) wistr << u8 << L' ';
            wistr << L'\n';
            return wistr;
        }

    private:
        std::vector<float>   nums_f32;
        std::vector<int64_t> nums_i64;
        std::vector<uint8_t> nums_u8;
};

int main() {
    std::vector<object> objvector {};

    auto obj_0 {
        object { 5.5F, 78, 121 }
    },
        obj_1 { object { 4.5F, 12, 134 } };

    const auto cpy = obj_0;

    objvector.push_back(obj_0);            // copy
    objvector.push_back(obj_1);            // copy
    objvector.push_back(obj_0 + obj_1);    // this creates a temporary, which gets moved into the vector
    objvector.push_back(std::move(obj_1)); // move
    objvector.push_back(std::move(obj_0)); // move

    std::wcout << objvector.at(2);

    return EXIT_SUCCESS;
}
