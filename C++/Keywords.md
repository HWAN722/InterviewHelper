# Keywords
## alignas
`alignas` is used to specify the alignment of a type or variable. Alignment refers to the way data is stored in memory. Usually, in order to improve the access speed, some data types need to be aligned on specific byte boundaries.
```cpp
#include <iostream>
#include <cstddef> // for std::max_align_t

struct alignas(16) MyStruct {
    double x;
    double y;
};

int main() {
    MyStruct s;
    std::cout << "Alignment of MyStruct: " << alignof(MyStruct) << std::endl;  //alignof returns the alignment requirement of a type.
    std::cout << "Size of MyStruct: " << sizeof(MyStruct) << std::endl;
    std::cout << "Address of s: " << &s << std::endl;
    return 0;
}
```
In this case, 'MyStruct' is specified as 16 byte alignment, which means that the starting address of each instance of 'MyStruct' will be a multiple of 16.

## auto
`auto` allows the compiler to infer the type of a variable according to its initialization expression, thus simplifying the code and improving readability. Auto can be used in scenarios such as variable declaration and function return type.
```cpp
auto x = 42;          // x is int
auto y = 3.14;        // y is double
auto z = "Hello";     // z is const char*
auto lambda = [](auto a, auto b) { //lambda expression
    return a + b;
};

int a = 10;
int& ref = a;
auto b = ref;  // b is int, not int&，because auto duplicate the value
```


## concept
`concept` is used to specify the constraints of template parameters. `concept` provide a clearer and more powerful way to define the requirements of templates, thus improving the readability and maintainability of the code.
```cpp
// define a concept，template T must support add
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

// use concept to constrain
template<Addable T>
T add(T a, T b) {
    return a + b;
}

int main() {
    std::cout << add(1, 2) << std::endl; // compile successfuly
    //std::cout << add("Hello", "World") << std::endl; // compile error，std::string is not Addable
    return 0;
}
```


## explicit
`explicit` in C++ is used to prevent unintended implicit type conversions when constructors or conversion operators are invoked.
```cpp
class Box {
public:
    // Constructor marked explicit
    explicit Box(int size) : size_(size) {}
    void display() {
        std::cout << "Box size: " << size_ << std::endl;
    }
private:
    int size_;
};

int main() {
    // Box b = 10;  // Error: implicit conversion is not allowed
    Box b(10);      // Explicit conversion
    b.display();    // Outputs: Box size: 10
    return 0;
}
```


## inline
The `inline` keyword in C++ is used as a request to the compiler to replace a function call with the actual code of the function at compile time. This can potentially improve performance by avoiding the overhead of a function call, but it comes with certain trade-offs.
```cpp
// Inline function definition
inline int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(3, 5);  // Function call
    std::cout << "Result: " << result << std::endl;
    return 0;
}
```


### Example: When Inlining May Be Ignored
1. Large Function:
```cpp
inline void largeFunction() {
    for (int i = 0; i < 100000; ++i) {
        std::cout << i << std::endl;
    }
}
```
The compiler may decide not to inline this function due to its size.
2. Recursive Function:

```cpp
inline int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);  // Recursive call
}
```
Recursive functions are generally not inlined because they involve multiple calls to themselves.


## typedef
`typedef` keyword in C++ allows you to define a new name (alias) for an existing type. This can make code more readable, improve portability, and simplify complex type declarations.
```cpp
typedef unsigned int UInt;

int main() {
    UInt a = 10;  // Equivalent to: unsigned int a = 10;
    std::cout << "Value of a: " << a << std::endl;
    return 0;
}
```


## virtual
`virtual` keyword in C++ is used to define polymorphic behavior in object-oriented programming. It allows derived classes to override a function in the base class and ensures that the overridden version of the function is called dynamically at runtime, even when accessed through a base class pointer or reference.
```cpp
class Base {
public:
    virtual void display() {  // Virtual function
        cout << "Base class display" << endl;
    }
};

class Derived : public Base {
public:
    void display() override {  // Override function
        cout << "Derived class display" << endl;
    }
};

int main() {
    Base* b;                 // Base class pointer
    Derived d;               // Derived class object
    b = &d;                  // Base pointer pointing to Derived object
    b->display();            // Calls Derived's display() at runtime
    return 0;
}
```
