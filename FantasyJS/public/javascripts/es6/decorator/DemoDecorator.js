// Decorator 提案经过了大幅修改，目前还没有定案，不知道语法会不会再变。
// 下面的内容完全依据以前的提案，已经有点过时了。等待定案以后，需要完全重写。

// 装饰器（Decorator）是一种与类（class）相关的语法，用来注释或修改类和类方法。
// 许多面向对象的语言都有这项功能，目前有一个提案将其引入了 ECMAScript。

// 装饰器是一种函数，写成@ + 函数名。它可以放在类和类方法的定义前面。

// 文章参考：https://fed.taobao.org/blog/taofed/do71ct/es7-decorator/?spm=taofed.homepage.header.7.7eab5ac8fYHude
// 文章参考：https://imweb.io/topic/5b1403bbd4c96b9b1b4c4e9e


console.log("=============装饰器（Decorator）基础用法==============");


console.log("=============装饰器（Decorator）类的装饰==============");

/**
 * 当我们的装饰器方法用来装饰类的时候，并且需要传入参数。
 * 可以在装饰器外面再封装一层函数。
 * @param isTestable
 * @returns {function(...[*]=)}
 */
function testable(isTestable) {
    return function (target) {
        target.isTestable = isTestable;
    }
}

@testable(true)
class MyTestableClass {

}

@testable(false)
class MyTestableClassFalse {
}

// 上面代码中，装饰器testable可以接受参数，这就等于可以修改装饰器的行为。

console.log(MyTestableClass.isTestable);// true
console.log(MyTestableClassFalse.isTestable);// true
// 注意，装饰器对类的行为的改变，是代码编译时发生的，而不是在运行时。
// 这意味着，装饰器能在编译阶段运行代码。也就是说，装饰器本质就是编译时执行的函数。


// 前面的例子是为类添加一个静态属性，如果想添加实例属性，可以通过目标类的prototype对象操作。
console.log("=============装饰器（Decorator）基础用法:给类增加实例属性==============");

function testPrototype(target) {
    // 给target的原型链上增加一个isTestable属性
    target.prototype.isTestable = true;
}

@testPrototype
class MyTestProtoTypeClass {
}

let myTestProtoTypeClass = new MyTestProtoTypeClass();
console.log(myTestProtoTypeClass.isTestable); // true

// 上面代码中，装饰器函数testable是在目标类的prototype对象上添加属性，因此就可以在实例上调用。


// 下面是另外一个例子。
console.log("=============装饰器（Decorator）基础用法: 把一个对象的方法传给另一个类==============");

function MultiProperties(...list) {
    return function (target) {
        Object.assign(target.prototype, ...list)
    }
}

// 首先我们实例化一个对象，里面有一个方法foo
const Foo = {
    foo() {
        console.log('foo')
    }
};

@MultiProperties(Foo)
class MyMultiPropertiesClass {

}

let myMultiPropertiesClass = new MyMultiPropertiesClass();
myMultiPropertiesClass.foo();// 'foo'

// 实际开发中，React 与 Redux 库结合使用时，常常需要写成下面这样。

// class MyReactComponent extends React.Component {}
//
// export default connect(mapStateToProps, mapDispatchToProps)(MyReactComponent);

// 有了装饰器，就可以改写上面的代码。
// @connect(mapStateToProps, mapDispatchToProps)
// export default class MyReactComponent extends React.Component {}


console.log("=============装饰器（Decorator）方法的装饰==============");

// 装饰器不仅可以装饰类，还可以装饰类的属性。

class Person {
    @readonly
    name() {
        return `${this.first} ${this.last}`
    }

    @nonEnumerable
    get kidCount() {
        return this.children.length;
    }

    @log
    getCount(a, b) {
        return a + b;
    }
}

// 上面代码中，装饰器readonly用来装饰“类”的name方法。
// 装饰器函数readonly一共可以接受三个参数。
/**
 *
 * @param target 装饰器第一个参数是类的原型对象 也就是Person.prototype
 * 装饰器的本意是要“装饰”类的实例，但是这个时候实例还没生成，所以只能去装饰原型
 * （这不同于类的装饰，那种情况时target参数指的是类本身）；
 * @param name  第二个参数是所要装饰的属性名
 * @param descriptor 第三个参数是该属性的描述对象。
 * @returns {*}
 */
function readonly(target, name, descriptor) {
    // descriptor对象原来的值如下
    // {
    //   value: specifiedFunction,
    //   enumerable: false,
    //   configurable: true,
    //   writable: true
    // };
    descriptor.writable = false;
    return descriptor;
}

/**
 * 下面是另一个例子，修改属性描述对象的enumerable属性，使得该属性不可遍历。
 * @param target
 * @param name
 * @param descriptor
 * @returns {*}
 */
function nonEnumerable(target, name, descriptor) {
    descriptor.enumerable = false;
    return descriptor;
}

// readonly(Person.prototype, 'name', descriptor);
// 类似于
// Object.defineProperty(Person.prototype, 'name', descriptor);

function log(target, name, descriptor) {
    let oldValue = descriptor.value;

    descriptor.value = function () {
        console.log(`Calling ${name} with`, arguments);
        return oldValue.apply(this, arguments);
    };

    return descriptor;
}

const person = new Person();
person.getCount(1, 2);


// 装饰器不能用于函数？

// 装饰器只能用于类和类的方法，不能用于函数，因为存在函数提升。
let counter = 0;

const addDecorator = function () {
    counter++;
};

@addDecorator
function fooFunction() {
}





