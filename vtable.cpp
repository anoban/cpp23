#include <iostream>

struct base {
        // vtable ptr
        unsigned long  _fourbytes_taken;
        unsigned short _fillthe_gap;

        virtual void hello() const noexcept(true) { ::puts("Hello from base!"); }

        void hi() const noexcept { ::puts("Hi from base!"); } // NOLINT(readability-convert-member-functions-to-static)
};

struct derived : public base {
        // base subobject, which is practically empty here since the vtable ptr is not inherited
        // vtable ptr
        void hello() const noexcept override { ::puts("Hello from derived!"); }

        void hi() const noexcept { ::puts("Hi from derived!"); } // NOLINT(readability-convert-member-functions-to-static)
};

static_assert(sizeof(base) == 16); // woohooo
static_assert(sizeof(derived) == 16);

int main() {
    const derived     dobject {};
    const base        bobject {};
    const auto* const dptr { &dobject };
    const auto* const bptr { &bobject };

    // upcast
    reinterpret_cast<const base*>(dptr)->hello(); // Hello from derived!

    // downcast
    reinterpret_cast<const derived* const>(bptr)->hello(); // Hello from base!

    derived _derived {};
    base    _base {};
    // replace the _derived object's vtable pointer with class base's vtable pointer and see what happens when Hello() is called!
    *reinterpret_cast<uintptr_t*>(&_derived) = *reinterpret_cast<const uintptr_t*>(&bobject);
    *reinterpret_cast<uintptr_t*>(&_base)    = *reinterpret_cast<const uintptr_t*>(&dobject);

    _derived.hello(); // Hello from base!
    _base.hello();    // Hello from derived!

    return EXIT_SUCCESS;
}
