function init() {
    var name = "Mozilla"; // name 是一个被 init 创建的局部变量
    function displayName() { // displayName() 是内部函数，一个闭包
        // alert(name); // 使用了父函数中声明的变量
        console.log("name:" + name);
    }
    displayName();
}

init();

// init() 创建了一个局部变量 name 和一个名为 displayName() 的函数。
// displayName() 是定义在 init() 里的内部函数，并且仅在 init() 函数体内可用。

// 请注意，displayName() 没有自己的局部变量。
// 然而，因为它可以访问到外部函数的变量，
// 所以 displayName() 可以使用父函数 init() 中声明的变量 name 。

// 使用这个 JSFiddle 链接运行该代码后发现， displayName() 函数内的 alert() 语句成功显示出了变量 name 的值（该变量在其父函数中声明）。
// 这个词法作用域的例子描述了分析器如何在函数嵌套的情况下解析变量名。
// 词法（lexical）一词指的是，词法作用域根据源代码中声明变量的位置来确定该变量在何处可用。
// 嵌套函数可访问声明于它们外部作用域的变量。


function makeFunc() {
    var name = "makeFunc";
    function displayName() {
        console.log("name:" + name);
    }
    // 这个函数的最后一行返回了displayName。
    // 返回的其实我们理解他是一个function的执行体，也就是称作闭包
    return displayName;
}

var myFunc = makeFunc();
myFunc();


function makeAdder(x) {
    return function(y) {
        return x + y;
    };
}

var add5 = makeAdder(5);
var add10 = makeAdder(10);

console.log(add5(2));  // 7
console.log(add10(2)); // 12


var Counter = (function() {
    var privateCounter = 0;
    function changeBy(val) {
        privateCounter += val;
    }
    return {
        increment: function() {
            changeBy(1);
        },
        decrement: function() {
            changeBy(-1);
        },
        value: function() {
            return privateCounter;
        }
    }
})();

console.log(Counter.value()); /* logs 0 */
Counter.increment();
Counter.increment();
console.log(Counter.value()); /* logs 2 */
Counter.decrement();
console.log(Counter.value()); /* logs 1 */
