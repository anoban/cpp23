// in C++, inheritance can be virtual

class shape {
    private:
        float area;

    public:
        unsigned count;
};

class square : public shape { // public inheritance
    private:
        float length;

    public:
        float get() const noexcept { return area; }
};

class circle : private shape { // public inheritance
    private:
        float length;

    public:
        float get() const noexcept { return area; }
};

int wmain() {
    //
    constexpr auto sq = square {};

    return 0;
}
