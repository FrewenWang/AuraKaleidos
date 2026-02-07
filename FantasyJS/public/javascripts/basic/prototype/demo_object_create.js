var arr = [1, 2, 3];

console.log(arr)


console.log("===========================Constuctor===============================");

function Student(name) {
    this.name = name;
    this.hello = function () {
        console.log('Hello, ' + this.name + '!');
    }
}

// 这确实是一个普通函数，但是在JavaScript中，可以用关键字new来调用这个函数，并返回一个对象：

var xiaoming = new Student('小明');
console.log(xiaoming.name); // '小明'
xiaoming.hello(); // Hello, 小明!

// 注意，如果不写new，这就是一个普通函数，它返回undefined。但是，如果写了new，它就变成了一个构造函数，
// 它绑定的this指向新创建的对象，并默认返回this，也就是说，不需要在最后写return this;

// 新创建的xiaoming的原型链是：
// xiaoming ----> Student.prototype ----> Object.prototype ----> null

// 也就是说，xiaoming的原型指向函数Student的原型。如果你又创建了xiaohong、xiaojun，那么这些对象的原型与xiaoming是一样的：

console.log(xiaoming.constructor === Student.prototype.constructor); // true
console.log(Student.prototype.constructor === Student); // true
console.log(Object.getPrototypeOf(xiaoming) === Student.prototype); // true
console.log(xiaoming instanceof Student); // true
