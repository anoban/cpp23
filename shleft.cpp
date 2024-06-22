constexpr unsigned char x = 0b1011'0000;
constexpr unsigned char y = x << 4;

static_assert(0b1011'0110UI8 << 4 == 0b0110'0000UI8, "");
