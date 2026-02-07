// ES6 允许使用“箭头”（=>）定义函数。

const f = v => v;

// 等同于
const functionArrow = function (v) {
    return v;
};

// 如果箭头函数不需要参数或需要多个参数，就使用一个圆括号代表参数部分。
let arrowFunctionNoParam = () => 5;

// 等同于
let arrowFunctionNoParamReplace = function () {
    return 5
};

let arrowFunctionHasParams = (num1, num2) => num1 + num2;
// 等同于
arrowFunctionHasParamsReplace = function (num1, num2) {
    return num1 + num2;
};


// 如果箭头函数的代码块部分多于一条语句，就要使用大括号将它们括起来，并且使用return语句返回。
arrowFunctionHasParams = (num1, num2) => {
    console.log('arrowFunctionHasParams');
    return num1 + num2;
};


// 由于大括号被解释为代码块，所以如果箭头函数直接返回一个对象，必须在对象外面加上括号，否则会报错。

// 报语法错误:SyntaxError: Unexpected token :
// let arrowFunctionReturnObject = id => { id: id, name: "Temp" };
// let arrowFunctionReturnObject = id => { id: id, name: "Temp" };

// 不报错
let arrowFunctionReturnObject = id => ({id: id, name: "Temp"});


// 下面是一种特殊情况，虽然可以运行，但会得到错误的结果。
let arrowFunctionReturnObjectSpecial = () => {
    a: 1
};

console.log(arrowFunctionReturnObjectSpecial());  // undefined 并不会得到正确的对象结果


// 如果箭头函数只有一行语句，且不需要返回值，可以采用下面的写法，就不用写大括号了。

let arrowFunctionHasHasOneCaseNoReturn = () => void doesNotReturn();  // 这个方法其实没有任何意义


// 箭头函数可以与变量解构结合使用。
person = {
    first: "frewen",
    last: "wong"
};
const arrowFunctionDeconstruction = ({first, last}) => first + ' ' + last;

// 等同于
function arrowFunctionDeconstructionReplace(person) {
    return person.first + ' ' + person.last;
}

console.log('arrowFunctionDeconstruction', arrowFunctionDeconstruction(person));  // undefined 并不会得到正确的对象结果




console.log('==========箭头函数的this的用法===================');
// 在探讨箭头函数对于this的优化之前，我们先得明白this究竟是什么，以及它是如何使用的。
// this是使用call方法调用函数时传递的第一个参数，它可以在函数调用时修改，在函数没有调用的时候，this的值是无法确定。

// 如果没有使用过call方法来调用函数的话，上面的对于this的定义可能不太明白。那么我们需要先理解函数调用的两种方法。

/**
 *
 * @param name
 */
function testCommonFunction(name) {
    console.log(name);
    console.log(this);
}

// 我们看一下testCommonFunction函数输出吧：
// Object [global] {
//     global: [Circular],
//         clearInterval: [Function: clearInterval],
//     clearTimeout: [Function: clearTimeout],
//     setInterval: [Function: setInterval],
//     setTimeout: [Function: setTimeout] { [Symbol(util.promisify.custom)]: [Function] },
//     queueMicrotask: [Function: queueMicrotask],
//     clearImmediate: [Function: clearImmediate],
//     setImmediate: [Function: setImmediate] {
//         [Symbol(util.promisify.custom)]: [Function]
//     },
//     arrowFunctionHasParamsReplace: [Function: arrowFunctionHasParamsReplace],
//     person: { first: 'frewen', last: 'wong' }
// }
// 我们可以看到这里的this是一个global对象。里面包含全局的一些的变量和方法定义
testCommonFunction('testCommonFunction');  //调用函数


function testCommonFunctionWithCall(name) {
    console.log(name);
    console.log(this);
}

testCommonFunctionWithCall.call(undefined, 'testCommonFunctionWithCall');




// 使用注意点:
// 箭头函数有几个使用注意点。
//
// （1）箭头函数体内的this对象，就是定义时所在的对象，而不是使用时所在的对象。
//
// （2）不可以当作构造函数，也就是说，不可以使用new命令，否则会抛出一个错误。
//
// （3）不可以使用arguments对象，该对象在函数体内不存在。如果要用，可以用 rest 参数代替。
//
// （4）不可以使用yield命令，因此箭头函数不能用作 Generator 函数。
//
// 上面四点中，第一点尤其值得注意。this对象的指向是可变的，但是在箭头函数中，它是固定的。


// 关于箭头函数方法的this应用，我们来看下面的例子：
function arrowFunctionWithThisReference() {
    // 这个setTimeout里面传入的是一个箭头函数，
    // 这个箭头函数的定义生效是在foo函数生成时，而它的真正执行要等到 100 毫秒后。

    // 如果是普通函数，执行时this应该指向全局对象window，这时应该输出21。

    // 但是，箭头函数导致this总是指向函数定义生效时所在的对象（本例是{id: 42}），所以输出的是42。
    setTimeout(() => {
        console.log('id:', this.id);
    }, 100);
}

var id = 21;

// 关于普通对象的.call的方法，我们其他地方研究
arrowFunctionWithThisReference.call({id: 42});


// 箭头函数可以让setTimeout里面的this，绑定定义时所在的作用域，而不是指向运行时所在的作用域。下面是另一个例子。

// 比如这个时候，我们定义一个Timer
// Timer函数内部设置了两个定时器，分别使用了箭头函数和普通函数。

// 前者的this绑定定义时所在的作用域（即Timer函数），后者的this指向运行时所在的作用域（即全局对象）
function Timer() {
    this.s1 = 0;
    this.s2 = 0;

    console.log('Timer: ', "begin");
    // 箭头函数.前者的this绑定定义时所在的作用域（即Timer函数）
    // 所以箭头函数定义时候所在的作用域是Timer函数
    setInterval(() => {
        console.log('setInterval1: ', "begin", this.s1);
        this.s1++;
    }, 1000);

    // 后者的this指向运行时所在的作用域（即全局对象）然而全局是找不到这个s2.所以他是undefined
    // 所以
    // 普通函数
    setInterval(function () {
        console.log('setInterval2: ', "begin", this.s2);
        this.s2++;
    }, 1000);
}

var timer = new Timer();
// 进行方法调用
console.log('setTimeout: ', "begin");
setTimeout(() => console.log('s1: ', timer.s1), 3100);
setTimeout(() => console.log('s2: ', timer.s2), 3100);


// 箭头函数可以让this指向固定化，这种特性很有利于封装回调函数。下面是一个例子，DOM 事件的回调函数封装在一个对象里面。
var handler = {
    id: '123456',

    init: function () {
        document.addEventListener('click',
            event => this.doSomething(event.type), false);
    },

    doSomething: function (type) {
        console.log('Handling ' + type + ' for ' + this.id);
    }
};


//在没有使用ES6之后，我们总是会写很多黑Hack的谢谢。比如this = this这种  例如：箭头函数转成 ES5 的代码如下。
// ES6
function arrowFunctionES6() {
    setTimeout(() => {
        console.log('arrowFunctionES6', 'id:', this.id);
    }, 100);
}

// ES5
function arrowFunctionReplaceES5() {
    var _this = this;

    setTimeout(function () {
        console.log('arrowFunctionReplaceES5', 'id:', _this.id);
    }, 100);
}

// arrowFunctionES6 id: undefined
// arrowFunctionReplaceES5 id: undefined
// 所以这样无论怎么样，都是指向当前定义对象的id.确实也就是undefined
arrowFunctionES6();
arrowFunctionReplaceES5();


function foo() {
    return () => {
        return () => {
            return () => {
                console.log('id:', this.id);
            };
        };
    };
}


// 不适用场合
// 上面讲了箭头函数的使用：
// 由于箭头函数使得this从“动态”变成“静态”，下面两个场合不应该使用箭头函数。
// 第一个场合是定义对象的方法，且该方法内部包括this。

const arrowFunctionObjectMethod = {
    lives: 9,
    // 上面代码中，cat.jumps()方法是一个箭头函数，这是错误的。
    jumps: () => {
        this.lives--;
    }
    //调用cat.jumps()时，如果是普通函数，该方法内部的this指向cat；
    // 如果写成上面那样的箭头函数，使得this指向全局对象，因此不会得到预期结果。
    // 这是因为对象不构成单独的作用域，导致jumps箭头函数定义时的作用域就是全局作用域。
};


// 第二个场合是需要动态this的时候，也不应使用箭头函数。

/**
var button = document.getElementById('press');
button.addEventListener('click', () => {
    this.classList.toggle('on');
});
 */

//上面代码运行时，点击按钮会报错，因为button的监听函数是一个箭头函数，导致里面的this就是全局对象。
// 如果改成普通函数，this就会动态指向被点击的按钮对象。

// 另外，如果函数体很复杂，有许多行，或者函数内部有大量的读写操作，不单纯是为了计算值，
// 这时也不应该使用箭头函数，而是要使用普通函数，这样可以提高代码可读性。
