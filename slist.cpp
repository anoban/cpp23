// a singly linked list implementation

#include <cstddef>
#include <memory>

// an iterator class to be used with the singly linked list, only supports forward traversal
template<typename T /* template parameter to specify the node type */> class list_iterator {
    public:
        using reference       = T&;
        using const_reference = const T&;
        using pointer         = T*;
        using const_pointer   = const T*;
        using size_type       = size_t;
        using difference_type = ptrdiff_t;
        using value_type      = T;

        constexpr list_iterator() noexcept { }

        constexpr explicit list_iterator(_In_ const_pointer _start_node) noexcept : _node { _start_node } { }

        [[nodiscard]] constexpr list_iterator& operator++() noexcept {
            _node = _node->_next;
            return *this;
        }

        [[nodiscard]] constexpr list_iterator operator++(int) noexcept {
            const auto temp { *_node };
            _node = _node->next;
            return list_iterator(_node);
        }

        constexpr list_iterator& operator--()   = delete; // cannot move backwards

        constexpr list_iterator operator--(int) = delete; // cannot move backwards

    private:
        pointer _node {};
};

template<typename T, typename allocator = std::allocator<T> /* default allocator */> class list {
    public:
        using reference       = T&;
        using const_reference = const T&;
        using pointer         = T*;
        using const_pointer   = const T*;
        using size_type       = size_t;
        using difference_type = ptrdiff_t;
        using value_type      = T;
        using allocator_type  = allocator;

    private:
        // a plain C style aggregate type for the linked list nodes
        template<typename U> struct link {
                link<U>* _next {}; // data stored in the all nodes of the linked list must be of the same type
                U        _value {};
        };

        using link_type = link<T>;
        using link_allocator_type =
            typename allocator_type::template rebind<link_type>::other; // rebinding std::allocator<T> for allocations of link_type

        link_type*     _head {};   // first node of the linked list
        link_type*     _tail {};   // last node in the linked list
        size_type      _nlinks {}; // number of nodes in the linked list
        allocator_type _allocator; // allocator
};
