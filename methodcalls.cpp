#include <cstdlib>
#include <iostream>
#include <new>

class foo {
    private:
        float _bar;
        short _bazz;

    public:
        foo() = delete;

        foo(const float& bar, const short& bazz) noexcept : _bar { bar }, _bazz { bazz } { }

        virtual float bar() const noexcept { return _bar; }
};

static_assert(!std::is_standard_layout<foo>::value); // we have a virtual function

auto wmain() -> int {
    auto object {
        foo { 23.0354, 475 }
    };
    auto* objptr {
        new (std::nothrow) foo { 87.54, 765 }
    };

    std::wcout << object.bar() << L"  " << objptr->bar() << std::endl;

    return EXIT_SUCCESS;
}

namespace hypothetical {

    // what may have happened internally when
    // auto object { foo { 23.0354, 475 } };
    // auto* objptr { new (std::nothrow) foo { 87.54, 765 }};
    // object.bar();
    // objptr->bar();
    // statements were evaluated???

    struct vtable; // forward declaration

    extern "C" struct foo {
            hypothetical::vtable* vptr;
            float                 _bar;
            short                 _bazz;
    };

    extern "C" struct vtable {
            std::type_info info;
            float          (*fnptr)(const hypothetical::foo&) noexcept;
    };

    extern "C" float virtual_bar(const foo& object) noexcept { return object._bar; }

    extern "C" void  init(foo& object, const float& bar, const short& bazz) noexcept {
        // initialize the member variables with the provided arguments
        object._bar  = bar;
        object._bazz = bazz;
    }

    void demo() noexcept {
        const vtable vtable_for_foo { typeid(foo), virtual_bar };

        // compiler creates and constructs a POD class type foo on stack (foo is not a POD type)
        foo          onstack {};
        init(onstack, 23.0354, 475);
        bar(onstack);

        // construction of foo on heap
        foo* onheap { nullptr };
        onheap = reinterpret_cast<foo*>(::operator new(sizeof(foo)));
        init(*onheap, 87.54, 765);

        ::operator delete(onheap);
    }

} // namespace hypothetical
