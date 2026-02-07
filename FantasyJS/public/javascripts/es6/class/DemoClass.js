// JavaScript 语言中，生成实例对象的传统方法是通过构造函数。下面是一个例子。

/**
 * 定义一个Function
 * @param x
 * @param y
 * @constructor
 */
function Point(x, y) {
    this.x = x;
    this.y = y;
}

/**
 * TODO Point.prototype是个什么鬼？？
 * @returns {string}
 */
Point.prototype.toString = function () {
    return '(' + this.x + ', ' + this.y + ')';
};

const p = new Point(1, 2);

console.log(p);

// 上面这种写法跟传统的面向对象语言（比如 C++ 和 Java）差异很大，很容易让新学习这门语言的程序员感到困惑。

// ES6 提供了更接近传统语言的写法，引入了 Class（类）这个概念，作为对象的模板。通过class关键字，可以定义类。


// 基本上，ES6 的class可以看作只是一个语法糖，它的绝大部分功能，ES5 都可以做到，
// 新的class写法只是让对象原型的写法更加清晰、更像面向对象编程的语法而已。
// 上面的代码用 ES6 的class改写，就是下面这样。

class PointClass {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    toString() {
        return '(' + this.x + ', ' + this.y + ')';
    }
}

// 上面代码定义了一个“类”，可以看到里面有一个constructor方法，这就是构造方法，
// 而this关键字则代表实例对象。
// 也就是说，ES5 的构造函数Point，对应 ES6 的Point类的构造方法。


// Point类除了构造方法，还定义了一个toString方法。注意，定义“类”的方法的时候，
// 前面不需要加上function这个关键字，直接把函数定义放进去了就可以了。
// 另外，方法之间不需要逗号分隔，加了会报错。


//ES6 的类，完全可以看作构造函数的另一种写法。
class PointPrototype {
    // ...
}

console.log(typeof PointPrototype); // "function"
console.log(PointPrototype); // true
console.log(PointPrototype === PointPrototype.prototype.constructor); // true


// 类的实例
// 生成类的实例的写法，与 ES5 完全一样，也是使用new命令。
// 前面说过，如果忘记加上new，像函数那样调用Class，将会报错。

class PointDemo {
    constructor() {
        // ...
    }

    toString() {
        // ...
    }

    toValue() {
        // ...
    }
}

// 报错
//TypeError: Class constructor PointDemo cannot be invoked without 'new'
// var point = PointDemo(2, 3);

// 正确
var point = new PointDemo(2, 3);

// 构造函数的prototype属性，在 ES6 的“类”上面继续存在。事实上，类的所有方法都定义在类的prototype属性上面。

// 所以上面的代码等同于
PointDemo.prototype = {
    constructor() {
    },
    toString() {
    },
    toValue() {
    },
};


// 在类的实例上面调用方法，其实就是调用原型上的方法。
console.log("=====在类的实例上面调用方法，其实就是调用原型上的方法=======");

class B {
}

let b = new B();

console.log(b.constructor === B.prototype.constructor); // true
// 上面代码中，b是B类的实例，它的constructor方法就是B类原型的constructor方法。


console.log("=====Object.assign方法可以很方便地一次向类添加多个方法=======");
// 由于类的方法都定义在prototype对象上面，
// 所以类的新方法可以添加在prototype对象上面。Object.assign方法可以很方便地一次向类添加多个方法。
class PointPrototypeAssign {
    constructor() {
        // ...
    }
}

Object.assign(PointPrototypeAssign.prototype, {
    toString() {
    },
    toValue() {
    }
});


// prototype对象的constructor属性，直接指向“类”的本身，这与 ES5 的行为是一致的。
PointPrototypeAssign.prototype.constructor === PointPrototypeAssign // true


// 另外，类的内部所有定义的方法，都是不可枚举的（non-enumerable）。
