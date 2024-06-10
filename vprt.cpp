#include <iostream>
#include <string>

class object {
    public:
        unsigned long count;
        std::wstring  name;

        object() noexcept : count() { }
        ~object() = default;
        virtual void greet() const { std::wcout << L"Hello!\n"; }
};

auto wmain() -> int {
    //
    constexpr auto sizeof_basic_string { sizeof(std::wstring) }; // 40 bytes
    auto           obj { object {} };                            // data members take up 40 + 4 bytes in size
    // factoring in alignment costs, may be 48 + 8 bytes in size, the last 8 byte for the pointer to virtual function table
    constexpr auto size { sizeof(object) };

    std::wcout << L"address of the object is " << std::addressof(obj) << L'\n';
    std::wcout << L"address of the data member (unsigned long) count is " << std::addressof(obj.count) << L'\n';
    std::wcout << L"address of the data member (std::wstring) name is " << std::addressof(obj.name) << L'\n';

    // there's a 8 byte gap between the address of the object and the count data member
    // perhaps that's where the pointer to the vtable is!
    uint8_t* const vtableptr = reinterpret_cast<uint8_t*>(&obj) + 8; // this is the pointer to the virtual function table
    // generally the first member of this table is a std::type_info object stored for RTTI
    constexpr auto typeinfo_size { sizeof(std::type_info) }; // 24 bytes

    // dereferencing the pointer to virtual table will give us the virtual table
    // and we offset by 24 bytes to get the tentative virtual function pointer
    void (*ptrgreet)(void*) = reinterpret_cast<void (*)(void*)>(vtableptr + sizeof(std::type_info));
    (*ptrgreet)(std::addressof(obj));

    return EXIT_SUCCESS;
}
