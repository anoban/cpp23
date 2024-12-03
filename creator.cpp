#include <memory>

template<class T> struct NewCreator { // policy class 1 based on the C++ operator new
        typedef T           value_type;
        typedef T*          pointer_type;

        static pointer_type create() throw() { return new T; } // allocate memory for one element of type T in heap and return the address
};

template<class T> struct MallocCreator { // policy class 2 using UCRT's malloc()
        typedef T           value_type;
        typedef T*          pointer_type;

        // allocate memory for one element of type T in heap and return the address
        static pointer_type create() throw() { return malloc(sizeof(T)); }
};

template<class T> struct PrototypeCreator { // policy class 3
        typedef T        value_type;
        typedef T*       pointer_type;
        typedef const T* const_pointer_type;

    public:
        // default ctor
        explicit constexpr PrototypeCreator(pointer_type Object = nullptr) throw() : _prototype { Object } { }

        // the logic here is that if the inner pointer is not nullptr, return a pointer to a copy of the underlying type T else return nullptr
        pointer_type create() const throw() { return _prototype ? _prototype->clone() : nullptr; }

        // type T must have a clone() method defined

        pointer_type get_prototype() const throw() { return _prototype; }

        void         set_prototype(const_pointer_type Object) throw() { _prototype = Object; }

    private:
        pointer_type _prototype;
};

// policies are syntax oriented not signature oriented
// Creator policy specifies which syntactic features must be valid for conforming classes
// the coupling between a policy and a clas that conforms to that policy is rather loose

// e.g.
// Creator policy does not specify that the create() member function of conforming classes must be static or virtual
// it just requires that it returns a pointer_type object

// the three policy classes defined above have different implementations and different interfaces
// the create() method in class NewCreator and class MallocCreator are static but in PrototypeCreator it is not!
// PrototypeCreator has few more methods available in its public interface compared to the other two policy classes!

// however, all three policy classes define a create() method that returns a pointer_type objcet
// hence they adhere to the Creator policy  !!! NOTE THIS CREATOR POLICY IS AN IMAGINARY POLICY W/O ANY EXPLICIT DEFINITION IN CODE

// here's a class that consumes a creation policy
template<class creation_policy> class ObjectManager : public creation_policy { };
