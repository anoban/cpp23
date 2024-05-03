types: tuple[str] = (
    "char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "float",
    "double",
    "long double",
)

# use double curly braces to escape { or } in format strings
for type in types:
    print(
        """template<> struct is_arithmetic<{}> {{
        typedef {} type;
        static constexpr bool value {{ true }};
}};""".format(
            type, type
        )
    )
