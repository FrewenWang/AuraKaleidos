console.log('===============Reflect==============')
// 老写法
console.log('assign' in Object); // true
// 新写法
console.log(Reflect.has(Object, 'assign'));


console.log('===============Reflect==============')

let proxyObj = {
    name: 'Hello'
}
let loggedObj = new Proxy(proxyObj, {
    get(target, name) {
        console.log('get', target, name);
        return Reflect.get(target, name);
    },
    deleteProperty(target, name) {
        console.log('delete' + name);
        return Reflect.deleteProperty(target, name);
    },
    has(target, name) {
        console.log('has' + name);
        return Reflect.has(target, name);
    }
});

console.log(loggedObj.name);
console.log(loggedObj.hasOwnProperty('name'));


// 有了Reflect对象以后，很多操作会更易读。

console.log('===============Reflect  Apply==============')
// 老写法
Function.prototype.apply.call(Math.floor, undefined, [1.75]) // 1
// 新写法
Reflect.apply(Math.floor, undefined, [1.75]) // 1


// 静态方法

// Reflect对象一共有 13 个静态方法。

// Reflect.apply(target, thisArg, args)
// Reflect.construct(target, args)
// Reflect.get(target, name, receiver)
// Reflect.set(target, name, value, receiver)
// Reflect.defineProperty(target, name, desc)
// Reflect.deleteProperty(target, name)
// Reflect.has(target, name)
// Reflect.ownKeys(target)
// Reflect.isExtensible(target)
// Reflect.preventExtensions(target)
// Reflect.getOwnPropertyDescriptor(target, name)
// Reflect.getPrototypeOf(target)
// Reflect.setPrototypeOf(target, prototype)

// 上面这些方法的作用，大部分与Object对象的同名方法的作用都是相同的，而且它与Proxy对象的方法是一一对应的。下面是对它们的解释。


// Reflect.get(target, name, receiver)

// Reflect.get方法查找并返回target对象的name属性，如果没有该属性，则返回undefined。

console.log('===============Reflect  Reflect.get(target, name, receiver)==============')
let myObject = {
    foo: 1,
    bar: 2,
    get baz() {
        return this.foo + this.bar;
    },
};
console.log(Reflect.get(myObject, 'foo')) // 1
console.log(Reflect.get(myObject, 'bar')) // 2
console.log(Reflect.get(myObject, 'baz')) // 3


// 如果name属性部署了读取函数（getter），则读取函数的this绑定receiver。

let myReceiverObject = {
    foo: 4,
    bar: 4,
};

/// 答案是8的原因：如果name属性部署了读取函数（getter），则读取函数的this绑定receiver。
/// 所有这里面的this。绑定的是myReceiverObject对象
console.log(Reflect.get(myObject, 'baz', myReceiverObject)) // 8


console.log('===============Reflect  Reflect.set(target, name, value, receiver)==============')

console.log(Reflect.get(myObject, 'foo')) // 1
console.log(myObject.foo) // 1
console.log(Reflect.set(myObject, 'foo', 123))
console.log(Reflect.get(myObject, 'foo')) // 123
console.log(myObject.foo) // 123

