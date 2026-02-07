// Promise.prototype.finally()

// finally方法用于指定不管 Promise 对象最后状态如何，都会执行的操作。该方法是 ES2018 引入标准的。

let promise = new Promise(function (resolve, reject) {
    setTimeout((param1, param2) => {
        console.log(param1);
        console.log(param2);
        resolve(param1);
    }, 100, "hello", "world");
});
promise
    .then((param1, param2) => {
        console.log('result', param1, param2)
    })
    .catch(error => {
        console.log('error')
    })
    .finally(() => {
        console.log('finally')
    });

// 上面代码中，不管promise最后的状态，在执行完then或catch指定的回调函数以后，都会执行finally方法指定的回调函数。
