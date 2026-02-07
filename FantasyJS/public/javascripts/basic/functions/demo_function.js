var abs = function (x) {
    if (x >= 0) {
        return x;
    } else {
        return -x;
    }
};

console.log(abs(10)); // 返回10
console.log(abs(-9));// 返回9


// 由于JavaScript允许传入任意个参数而不影响调用，
// 因此传入的参数比定义的参数多也没有问题，虽然函数内部并不需要这些参数：

console.log(abs(10, 'blablabla')); // 返回10
console.log(abs(-9, 'haha', 'hehe', null));// 返回9


//传入的参数比定义的少也没有问题：
console.log(abs()); // 返回NaN  这个为什么会返回NaN呢？


// 要避免收到undefined，可以对参数进行检查：
// TODO 话说这个和上面那个如果定义同样的名字的时候，为什么调用的是上面的
function absType(x) {
    if (typeof x !== 'number') {
        throw 'Not a number';
    }
    if (x >= 0) {
        return x;
    } else {
        return -x;
    }
}

// console.log(absType()); //  throw 'Not a number';


//arguments

// JavaScript还有一个免费赠送的关键字arguments，它只在函数内部起作用，
// 并且永远指向当前函数的调用者传入的所有参数。arguments类似Array但它不是一个Array：

// 利用arguments，你可以获得调用者传入的所有参数。也就是说，即使函数不定义任何参数，还是可以拿到参数的值：
function foo(x) {
    console.log('x = ' + x); // 10
    for (var i = 0; i < arguments.length; i++) {
        console.log('arg ' + i + ' = ' + arguments[i]); // 10, 20, 30
    }
}

foo(10, 20, 30);

function fooWithNoParam() {
    if (arguments.length === 0) {
        return 0;
    }
    var x = arguments[0];
    return x >= 0 ? x : -x;
}

console.log(fooWithNoParam()); // 0
console.log(fooWithNoParam(10)); // 10
console.log(fooWithNoParam(-9)); // 9

// 实际上arguments最常用于判断传入参数的个数。你可能会看到这样的写法：

// foo(a[, b], c)
// 接收2~3个参数，b是可选参数，如果只传2个参数，b默认为null：
function fooWithParaCount(a, b, c) {
    if (arguments.length === 2) {
        // 实际拿到的参数是a和b，c为undefined
        c = b; // 把b赋给c
        b = null; // b变为默认值
    }
    // ...
}

// 要把中间的参数b变为“可选”参数，就只能通过arguments判断，然后重新调整参数并赋值。


// rest参数
// 由于JavaScript函数允许接收任意个参数，于是我们就不得不用arguments来获取所有参数：
function fooRest(a, b) {
    var i, rest = [];
    if (arguments.length > 2) {
        for (i = 2; i < arguments.length; i++) {
            rest.push(arguments[i]);
        }
    }
    console.log('a = ' + a);
    console.log('b = ' + b);
    console.log(rest);
}


// 小心你的return语句

//前面我们讲到了JavaScript引擎有一个在行末自动添加分号的机制，这可能让你栽到return语句的一个大坑：

function foo() {
    return { name: 'foo' };
}

console.log(foo()); // { name: 'foo' }

// 如果把return语句拆成两行：
function foo() {
    return
    { name: 'foo' };
}

console.log(foo()); // undefined
