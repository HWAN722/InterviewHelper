# Concepts

## lvalue & rvalue

### Lvalue
An lvalue (locator value) refers to an object that occupies some identifiable location in memory (i.e., it has an address). Lvalues are typically variables or objects that can appear on the left-hand side of an assignment.

**Characteristics of Lvalues:**

1. They have a persistent state in memory.
2. They can be assigned a new value.

**Examples of Lvalues:**
```C++
int x = 10;  // 'x' is an lvalue
x = 20;      // 'x' can be assigned a new value

int* ptr = &x;  // 'x' has an address, so you can take its address

std::string str = "Hello";
str[0] = 'h';   // 'str[0]' is an lvalue, referring to the first character of the string
```
### Rvalue
An rvalue (read value) is a temporary value that does not have a persistent memory location. Rvalues are typically literals or temporary objects that appear on the right-hand side of an assignment.

**Characteristics of Rvalues:**
1. They do not have a persistent state in memory.
2. They cannot be assigned a new value directly.

**Examples of Rvalues:**
```C++

int y = 5;      // '5' is an rvalue
int z = x + y;  // 'x + y' is an rvalue, the result of the addition

int a = 10;
int b = a + 1;  // 'a + 1' is an rvalue

std::string greeting = "Hello" + std::string(" World");  // The result of the string concatenation is an rvalue
```

**Overloaded Assignment Operator**
`t{} = b` must be an object that can be assigned to `a`
```cpp
struct t{
    int value;
    t() : value(0) {} // Initializer list
};

int main() {
    t a;
    t b;
    a=b={};
    //a={}=b; //error: initializer list cannot be used on the left hand side of operator '='
    a=t{}=b;  // complie success  
    return 0;
}
```

**Summary**
Lvalues: Have a memory address, can be assigned to, and are typically variables or objects.
Rvalues: Do not have a persistent memory address, are temporary, and are typically literals or temporary results of expressions.


## Override & Overload

### Override
Override refers to redefining a base class's virtual function in a derived class. This allows the derived class to provide a specific implementation for a function that is already defined in its base class. The override keyword, introduced in C++11, is used to indicate that a function is intended to override a base class function.

**Example of Override:**

```C++

#include <iostream>

class Base {
public:
    virtual void show() const {
        std::cout << "Base class show function" << std::endl;
    }
};

class Derived : public Base {
public:
    void show() const override {  // Overrides Base::show
        std::cout << "Derived class show function" << std::endl;
    }
};

int main() {
    Base* b = new Derived();
    b->show();  // Calls Derived::show due to polymorphism
    delete b;
    return 0;
}
```
**Without Virtual keywords**
```cpp
#include <iostream>

class Base {
public:
    void show() const {  // No 'virtual' keyword
        std::cout << "Base class show function" << std::endl;
    }
};

class Derived : public Base {
public:
    void show() const override {  // This does not override Base::show
        std::cout << "Derived class show function" << std::endl;
    }
};

int main() {
    Base* b = new Derived();
    b->show();  // Calls Base::show, not Derived::show
    delete b;
    return 0;
}
```

### Overload
Overload refers to defining multiple functions with the same name but different parameter lists within the same scope. Function overloading allows you to use the same function name for different operations, provided they have different signatures (i.e., different types or numbers of parameters).

**Example of Overload:**

```C++

#include <iostream>

class Printer {
public:
    void print(int i) {
        std::cout << "Printing int: " << i << std::endl;
    }

    void print(double d) {
        std::cout << "Printing double: " << d << std::endl;
    }

    void print(const std::string& s) {
        std::cout << "Printing string: " << s << std::endl;
    }
};

int main() {
    Printer p;
    p.print(5);           // Calls print(int)
    p.print(3.14);        // Calls print(double)
    p.print("Hello");     // Calls print(const std::string&)
    return 0;
}
```

### Key Differences
**Override:**
1. Involves inheritance and virtual functions.
2. Used to provide a new implementation for a base class's virtual function in a derived class.
3. Requires the override keyword for clarity and error checking (optional but recommended).

**Overload:**
1. Involves functions with the same name but different parameter lists.
2. Can occur within the same class or scope.
3. Does not involve inheritance or virtual functions.