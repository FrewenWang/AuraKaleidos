// Promise.prototype.catch方法是.then(null, rejection)或.then(undefined, rejection)的别名，用于指定发生错误时的回调函数。

getJSON('/posts.json').then(function (posts) {
    // ...
}).catch(function (error) {
    // 处理 getJSON 和 前一个回调函数运行时发生的错误
    console.log('发生错误！', error);
});

// 上面代码中，getJSON方法返回一个 Promise 对象，
// 如果该对象状态变为resolved，则会调用then方法指定的回调函数；如果异步操作抛出错误，
// 状态就会变为rejected，就会调用catch方法指定的回调函数，处理这个错误。
// 另外，then方法指定的回调函数，如果运行中抛出错误，也会被catch方法捕获。

p.then((val) => console.log('fulfilled:', val))
    .catch((err) => console.log('rejected', err));

// 等同于
p.then((val) => console.log('fulfilled:', val))
    .then(null, (err) => console.log("rejected:", err));

// 下面是一个例子。

const promise = new Promise(function (resolve, reject) {
    throw new Error('test');
});
promise.catch(function (error) {
    console.log(error);
});
// Error: test

// 上面代码中，promise抛出一个错误，就被catch方法指定的回调函数捕获。注意，上面的写法与下面两种写法是等价的。

// 写法一
const promise = new Promise(function (resolve, reject) {
    try {
        throw new Error('test');
    } catch (e) {
        reject(e);
    }
});
promise.catch(function (error) {
    console.log(error);
});

// 写法二
const promise = new Promise(function (resolve, reject) {
    reject(new Error('test'));
});
promise.catch(function (error) {
    console.log(error);
});


// 比较上面两种写法，可以发现reject方法的作用，等同于抛出错误。
//
// 如果 Promise 状态已经变成resolved，再抛出错误是无效的。

const promise = new Promise(function (resolve, reject) {
    resolve('ok');
    throw new Error('test');
});
promise
    .then(function (value) {
        console.log(value)
    })
    .catch(function (error) {
        console.log(error)
    });
// ok

// 上面代码中，Promise 在resolve语句后面，再抛出错误，不会被捕获，等于没有抛出。
// 因为 Promise 的状态一旦改变，就永久保持该状态，不会再变了。
//
// Promise 对象的错误具有“冒泡”性质，会一直向后传递，直到被捕获为止。
// 也就是说，错误总是会被下一个catch语句捕获。
getJSON('/post/1.json').then(function (post) {
    return getJSON(post.commentURL);
}).then(function (comments) {
    // some code
}).catch(function (error) {
    // 处理前面三个Promise产生的错误
});


