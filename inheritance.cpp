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

static_assert(base {}._private);   // inaccessible
static_assert(base {}._protected); // inaccessible
static_assert(base {}._public);

class publicly_derived : public base {
    public:
        consteval publicly_derived() noexcept : base {} { }
        // upon construction of the derived class type, call the constructor of the base class.
        // private member of base class is not accessible through inheritance
};

static_assert(publicly_derived {}._private);   // inaccessible, public
static_assert(publicly_derived {}._protected); // inaccessible, protected
static_assert(publicly_derived {}._public);

class privately_derived : private base {
    public:
        consteval privately_derived() noexcept : base {} { }
};

static_assert(privately_derived {}._private); // inaccessible, private
static_assert(privately_derived {}._protected
); // inaccessible, private inheritance promotes protected members of base class to private privilege level in the derived class
static_assert(privately_derived {}._public
); // inaccessible, private inheritance promotes public members of base class to private privilege level in the derived class

class protectedly_derived : protected base {
    public:
        consteval protectedly_derived() noexcept : base {} { }
};

static_assert(protectedly_derived {}._private); // inaccessible, private
static_assert(protectedly_derived {}._protected);
static_assert(protectedly_derived {}._public
); // inaccessible, protected inheritance promotes public members of base class to protected privilege level in the derived class

// for structs the default inheritance mode is public inheritance
// for classes, the default inheritance mode is private inheritance
// the default mode of inheritance is dictated by the deriving type, not the type that is being derived from

class derived_class : base { };   // private inheritance
struct derived_struct : base { }; // public inheritance

// public inheritance is essentially an as-is inheritance
// all members of the base class are reintroduced into the derived class with the same access specifiers
// because an inheritance mode can only escalate the privilege of members with lesser privilege than itself and
// there are no privilege levels below public accessibility
