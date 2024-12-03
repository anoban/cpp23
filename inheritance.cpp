#include <iostream>
#include <numbers>

// access privilege levels can be imagined as public < protected < private
// when we specify a certain type of inheritance, the members of the base class whose privilege is lower than the specified privilege will be
// promoted to have the new privilege level

// a derived class will have access to all the base class's protected and public members
// but cannot access base class's private members, regardless of the type of inheritance

class base {
    private:
        long _private;

    protected:
        long _protected;

    public:
        long _public;

        consteval base() noexcept : _private { 1 }, _protected { 10 }, _public { 100 } { }
};

#ifdef __TEST__
static_assert(base {}._private);   // inaccessible
static_assert(base {}._protected); // inaccessible
static_assert(base {}._public);
#endif

class publicly_derived : public base {
    public:
        consteval publicly_derived() noexcept : base {} { }

        // upon construction of the derived class type, call the constructor of the base class.
        // private member of base class is not accessible through inheritance
};

#ifdef __TEST__
static_assert(publicly_derived {}._private);   // inaccessible, public
static_assert(publicly_derived {}._protected); // inaccessible, protected
static_assert(publicly_derived {}._public);
#endif

class privately_derived : private base {
    public:
        consteval privately_derived() noexcept : base {} { }
};

#ifdef __TEST__
static_assert(privately_derived {}._private); // inaccessible, private
static_assert(privately_derived {}._protected
); // inaccessible, private inheritance promotes protected members of base class to private privilege level in the derived class
static_assert(privately_derived {}._public
); // inaccessible, private inheritance promotes public members of base class to private privilege level in the derived class
#endif

class protectedly_derived : protected base {
    public:
        consteval protectedly_derived() noexcept : base {} { }
};
#ifdef __TEST__
static_assert(protectedly_derived {}._private); // inaccessible, private
static_assert(protectedly_derived {}._protected);
static_assert(protectedly_derived {}._public
); // inaccessible, protected inheritance promotes public members of base class to protected privilege level in the derived class
#endif

// for structs the default inheritance mode is public inheritance
// for classes, the default inheritance mode is private inheritance
// the default mode of inheritance is dictated by the deriving type, not the type that is being derived from

class derived_class : base { }; // private inheritance

struct derived_struct : base { }; // public inheritance

// public inheritance is essentially an as-is inheritance
// all members of the base class are reintroduced into the derived class with the same access specifiers
// because an inheritance mode can only escalate the privilege of members with lesser privilege than itself and
// there are no privilege levels below public accessibility

class material {
    protected:
        unsigned quantity;
        float    unit_price;
        float    discount_prcnt;

    public:
        material() = delete;

        constexpr material(unsigned _quant, float _uprice, float _discnt) noexcept :
            quantity { _quant }, unit_price { _uprice }, discount_prcnt { _discnt } { }
};

class book {
    protected:
        std::wstring author;
        std::wstring title;
        unsigned     page_count;

    public:
        book() = delete;

        book(std::wstring auth, std::wstring titl, const unsigned& pc) noexcept :
            author { std::move(auth) }, title { std::move(titl) }, page_count { pc } { }
};

// multiple inheritance from two base classes
class novel : public material, public book {
    public:
        novel() = delete;

        novel(unsigned quant, float uprice, float disprcnt, std::wstring auth, std::wstring titl, unsigned pcount) noexcept :
            material { quant, uprice, disprcnt }, book { auth, titl, pcount } { }

        const std::wstring& author() const noexcept { return book::author; }

        const std::wstring& title() const noexcept { return book::title; }

        unsigned            pages() const noexcept { return book::page_count; }

        unsigned            stock() const noexcept { return material::quantity; }

        float               price() const noexcept { return unit_price; }

        float               discount() const noexcept { return material::discount_prcnt; }

        ~novel() = default;
};

template<typename T> requires std::integral<T> class something {
    public:
        typedef std::remove_reference<T>::type value_type;

    protected:
        value_type foo;
        value_type bar;

    public:
        constexpr something(const T& f, const T& b) noexcept : foo(f), bar(b) { }
};

template<typename T> class otherthing final : public something<T> {
        using typename something<T>::value_type;

    private:
        value_type bazz;

    public:
        constexpr otherthing(const T& f, const T& b, const T& z) noexcept : otherthing::something { f, b }, bazz { z } { }

        constexpr ~otherthing() noexcept { otherthing::foo = otherthing::bar = bazz = 0; }

        constexpr value_type get_foo() const noexcept { return otherthing::something::foo; }
};

auto wmain() -> int {
    const novel TheDavinciCode { 120, 89.54, 3.65, L"Dan Brown", L"The Da Vinci Code", 689 };

    std::wcout << TheDavinciCode.author() << std::endl;
    std::wcout << TheDavinciCode.title() << std::endl;
    std::wcout << TheDavinciCode.pages() << std::endl;
    std::wcout << TheDavinciCode.stock() << std::endl;
    std::wcout << TheDavinciCode.price() << std::endl;
    std::wcout << TheDavinciCode.discount() << std::endl;

    otherthing<short> other { 34, 567, 5798 };

    other.~otherthing();

    return EXIT_SUCCESS;
}
