'use strict';

function foo() {
    var x = 1;
    x = x + 1;
}

// x = x + 2; // ReferenceError: x is not defined
// ReferenceError! 无法在函数体外引用变量x


// 如果两个不同的函数各自申明了同一个变量，那么该变量只在各自的函数体内起作用。
// 换句话说，不同函数内部的同名变量互相独立，互不影响：
'use strict';

function foo() {
    var x = 1;
    x = x + 1;
    return x
}

function bar() {
    var x = 'A';
    x = x + 'B';
    return x
}

console.log(foo());
console.log(bar());


// 由于JavaScript的函数可以嵌套，此时，内部函数可以访问外部函数定义的变量，反过来则不行：
function foo() {
    var x = 1;

    function bar() {
        // 内部函数可以访问外部函数定义的变量
        var y = x + 1; // bar可以访问foo的变量x!
    }

    // 外部函数不能访问内部函数定义的变量
    // var z = y + 1; // ReferenceError! foo不可以访问bar的变量y!
}


// 如果内部函数和外部函数的变量名重名怎么办？来测试一下：
function foo() {
    var x = 1;

    function bar() {
        var x = 'A';
        console.log('x in bar() = ' + x); // 'A'
    }

    console.log('x in foo() = ' + x); // 1
    bar();
}

foo();

//这说明JavaScript的函数在查找变量时从自身函数定义开始，从“内”向“外”查找。
// 如果内部函数定义了与外部函数重名的变量，则内部函数的变量将“屏蔽”外部函数的变量。


// 变量提升
// JavaScript的函数定义有个特点，它会先扫描整个函数体的语句，把所有申明的变量“提升”到函数顶部：


function foo() {
    var x = 'Hello, ' + y;   // Hello, undefined

    console.log(x);
    var y = 'Bob';
}

foo();

// 虽然是strict模式，但语句var x = 'Hello, ' + y;并不报错，原因是变量y在稍后申明了。
// 但是console.log显示Hello, undefined，说明变量y的值为undefined。
// 这正是因为JavaScript引擎自动提升了变量y的声明，但不会提升变量y的赋值。

// 全局作用域

// 不在任何函数内定义的变量就具有全局作用域。
// 实际上，JavaScript默认有一个全局对象window，全局作用域的变量实际上被绑定到window的一个属性：
var course = 'Learn JavaScript';
// 注意：这个必须要在浏览器里面执行才能正常执行
alert(course); // 'Learn JavaScript'
alert(window.course); // 'Learn JavaScript'

// 你可能猜到了，由于函数定义有两种方式，以变量方式var foo = function () {}定义的函数实际上也是一个全局变量，
// 因此，顶层函数的定义也被视为一个全局变量，并绑定到window对象：
function windowFoo() {
    alert('windowFoo');
}

windowFoo(); // 直接调用foo()
window.windowFoo(); // 通过window.foo()调用


// 进一步大胆地猜测，我们每次直接调用的alert()函数其实也是window的一个变量：


// 名字空间
// 全局变量会绑定到window上，不同的JavaScript文件如果使用了相同的全局变量，
// 或者定义了相同名字的顶层函数，都会造成命名冲突，并且很难被发现。

// 减少冲突的一个方法是把自己的所有变量和函数全部绑定到一个全局变量中。例如：
// 唯一的全局变量MYAPP:
var MYAPP = {};

// 其他变量:
MYAPP.name = 'myapp';
MYAPP.version = 1.0;

// 其他函数:
MYAPP.foo = function () {
    return 'foo';
};

// 把自己的代码全部放入唯一的名字空间MYAPP中，会大大减少全局变量冲突的可能。
// 许多著名的JavaScript库都是这么干的：jQuery，YUI，underscore等等。
