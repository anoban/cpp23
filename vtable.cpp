#include <iostream>

struct base {
        // vtable ptr
        virtual void hello() const noexcept(true) { ::puts("Hello from base!"); }

        void hi() const noexcept { ::puts("Hi from base!"); }
};

struct derived : public base {
        // base subobject, which is practically empty here since the vtable ptr is not inherited
        // vtable ptr

        void hello() const noexcept override { ::puts("Hello from derived!"); }

        void hi() const noexcept { ::puts("Hi from derived!"); }
};

static_assert(sizeof(base) == 8); // woohooo
static_assert(sizeof(derived) == 8);

int main() {
    //
    return EXIT_SUCCESS;
}
