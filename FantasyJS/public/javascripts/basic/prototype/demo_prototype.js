// 我们定义一个对象Student
var Student = {
    name: 'Robot',
    height: 1.2,
    run: function () {
        console.log(this.name + ' is running...');
    }
};

var xiaoming = {
    name: '小明'
};
// 把xiaoming的原型指向了对象Student，看上去xiaoming仿佛是从Student继承下来的：
xiaoming.__proto__ = Student;

console.log(xiaoming.name); // '小明'
xiaoming.run(); // 小明 is running...


console.log("===========================createStudent===============================");

// 请注意，上述代码仅用于演示目的。在编写JavaScript代码时，
// 不要直接用obj.__proto__去改变一个对象的原型，并且，低版本的IE也无法使用__proto__。
// Object.create()方法可以传入一个原型对象，并创建一个基于该原型的新对象，但是新对象什么属性都没有，
// 因此，我们可以编写一个函数来创建xiaoming：
function createStudent(name) {
    // 基于Student原型创建一个新对象:
    var s = Object.create(Student);
    // 初始化新对象:
    s.name = name;
    return s;
}
// 这是创建原型继承的一种方法，JavaScript还有其他方法来创建对象
var xiaoming = createStudent('小明');
xiaoming.run(); // 小明 is running...
console.log(xiaoming.__proto__ === Student);// true
