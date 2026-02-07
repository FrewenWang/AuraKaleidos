// 定义Student的构造函数
function Student(props) {
    this.name = props.name || 'Unnamed';
}

// 创建Student原型链上的方法
Student.prototype.hello = function () {
    alert('Hello, ' + this.name + '!');
}


// PrimaryStudent构造函数:
function PrimaryStudent(props) {
    Student.call(this, props);
    this.grade = props.grade || 1;
}

// 空函数F:
function F() {
}

// 把F的原型指向Student.prototype:
F.prototype = Student.prototype;

// 把PrimaryStudent的原型指向一个新的F对象，F对象的原型正好指向Student.prototype:
PrimaryStudent.prototype = new F();

// 把PrimaryStudent原型的构造函数修复为PrimaryStudent:
PrimaryStudent.prototype.constructor = PrimaryStudent;

// 继续在PrimaryStudent原型（就是new F()对象）上定义方法：
PrimaryStudent.prototype.getGrade = function () {
    return this.grade;
};

// 创建xiaoming:
var xiaoming = new PrimaryStudent({
    name: '小明',
    grade: 2
});
console.log(xiaoming.name); // '小明'
console.log(xiaoming.grade); // 2

// 验证原型:
console.log(xiaoming.__proto__ === PrimaryStudent.prototype); // true
console.log(xiaoming.__proto__.__proto__ === Student.prototype); // true

// 验证继承关系:
console.log(xiaoming instanceof PrimaryStudent); // true
console.log(xiaoming instanceof Student); // true

console.log(PrimaryStudent)
console.log(Student)
