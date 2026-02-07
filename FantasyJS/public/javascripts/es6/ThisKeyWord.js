// node环境和浏览器的区别：

// 一、全局环境下this的指向
// 在node中this指向global而在浏览器中this指向window，这就是为什么underscore中一上来就定义了一 root；


//情况一：纯粹的函数调用
var globalVariable = 1;
var variable = 10000;

console.log('globalVariable', this.globalVariable);
console.log('variable', this.variable);

function testCommonFunction() {
    console.log('testCommonFunction', this);
    console.log('testCommonFunction', this.globalVariable);
}

testCommonFunction();  // 1


// 情况二：作为对象方法的调用

// 函数还可以作为某个对象的方法调用，这时this就指这个上级对象。

function testObjectMethod() {
    console.log('testObjectMethod', this);
    console.log('testObjectMethod', this.objVariable);
    console.log('testObjectMethod', this.variable);   // 1234567 因为this指的是这个上级对象
}

// 声明的object对象
var obj = {};
obj.objVariable = 1;
obj.variable = 1234567;
obj.method = testObjectMethod;

obj.method();


// 情况三 作为构造函数调用
// 所谓构造函数，就是通过这个函数，可以生成一个新对象。这时，this就指这个新对象。
function testMethodConstructor() {
    this.variable = "testMethodConstructor variable";
}

var methodConstructor = new testMethodConstructor();
console.log('testMethodConstructor', methodConstructor.variable); // "testMethodConstructor variable"

console.log('testMethodConstructor', variable); // "testMethodConstructor variable"


// 情况四 apply 调用
// apply()是函数的一个方法，作用是改变函数的调用对象。
// 它的第一个参数就表示改变后的调用这个函数的对象。因此，这时this指的就是这第一个参数。
function testMethodApply() {
    console.log(this.variable);
}

var objMethodApply = {};
objMethodApply.variable = "objMethodApply.variable";
objMethodApply.m = testMethodApply;
objMethodApply.m.apply();
