#include <iostream>
#include <string>
#include <type_traits>

struct object {
        // imagine an 8 byte vptr here
        unsigned long count; // 4 bytes
        // 4 bytes padding goes here
        std::wstring  name; // 32 bytes

        virtual ~object() = default;

        virtual void greet() const { std::wcout << L"Hello!\n"; }
};

static_assert(!std::is_standard_layout<object>::value); // offsetof use is illformed with non-standard layout types
static_assert(offsetof(object, count) == 8);            // the first 8 bytes are  taken up by the pointer to virtual table aka vptr
static_assert(offsetof(object, name) == 16);
static_assert(sizeof(object) == 48);

struct cstyle {
        char     initial;
        char     name[20];
        unsigned age;
};

static_assert(std::is_standard_layout<cstyle>::value); // yes!
static_assert(offsetof(cstyle, initial) == 0);         // see, no fucking vptrs

namespace dummies {
    extern "C" {
        struct __std_type_info_data {
                const char* _UndecoratedName;
                const char  _DecoratedName[1];

                __std_type_info_data() noexcept : _UndecoratedName(nullptr), _DecoratedName() { };
                __std_type_info_data(const __std_type_info_data&)            = default;
                __std_type_info_data(__std_type_info_data&&)                 = default;

                __std_type_info_data& operator=(const __std_type_info_data&) = default;
                __std_type_info_data& operator=(__std_type_info_data&&)      = default;
                ~__std_type_info_data()                                      = default;
        };
    }

} // namespace dummies

static_assert(std::is_standard_layout_v<dummies::__std_type_info_data>);

auto wmain() -> int {
    object dummy;

    std::wcout << std::hex << std::uppercase;

    std::wcout << L"address of the object is " << std::addressof(dummy) << L'\n';
    std::wcout << L"address of the object is " << &dummy << L'\n';
    std::wcout << L"address of the data member (unsigned long) count is " << std::addressof(dummy.count) << L'\n';
    std::wcout << L"address of the data member (std::wstring) name is " << std::addressof(dummy.name) << L'\n';

    // there's a 8 byte gap between the address of the object and the count data member, that's the vptr
    // &object gives us the pointer to the vptr but WE NEED THE VPTR, HENCE THE TWO LEVELS OF INDIRECTION
    const uintptr_t* const vptr = *reinterpret_cast<const uintptr_t**>(&dummy);

    std::wcout << L"vptr is " << vptr << L'\n';

    // generally the first member of this table is a std::type_info object stored for RTTI
    static_assert(sizeof(std::type_info) == 24); // std::type_info is 24 bytes

    // const std::type_info dummy_typeinfo = *reinterpret_cast<const std::type_info*>(vptr);
    // the above will not work because the copy ctor of std::type_info has been explicitly deleted

    // std::type_info has a virtual dtor, so there will be a vptr inside a std::type_info object
    // the only private data member of std::type_info is a struct __std_type_info_data

    static_assert(std::is_standard_layout<__std_type_info_data>::value); // woohoo :)

    // __std_type_info_data has all its ctors an assignment operators deleted explicitly

    struct __std_type_info_data {
            const char* _UndecoratedName;
            const char  _DecoratedName[1];
    };

    // we could emulate std::type_info very easily using a POD type
    struct type_info {
            uintptr_t*           _vptr; // a dummy to remove the first 8 bytes
            __std_type_info_data _data;
    };

    static_assert(sizeof(std::type_info) == sizeof(type_info));     // there we go :)
    static_assert(std::is_standard_layout<type_info>::value);       // yes
    static_assert(!std::is_standard_layout<std::type_info>::value); // no

    // we should be able to get this by dereferencing vptr + 1 (vptr offsetted by 8 bytes)
    type_info meta = *reinterpret_cast<const type_info*>(vptr);

    //::puts(__std_type_info_name(&meta._data, &__type_info_root_node));
    ::puts(meta._data._DecoratedName);

    return EXIT_SUCCESS;
}
