/**
 * Promise.race()方法同样是将多个 Promise 实例，包装成一个新的 Promise 实例。
 *
 */

// const racePromise = Promise.race([p1, p2, p3]);
// 上面代码中，只要p1、p2、p3之中有一个实例率先改变状态，p的状态就跟着改变。那个率先改变的 Promise 实例的返回值，就传递给p的回调函数。


// Promise.race()方法的参数与Promise.all()方法一样;
//
// 如果不是 Promise 实例，就会先调用下面讲到的Promise.resolve()方法，将参数转为 Promise 实例，再进一步处理。


// 下面是一个例子，如果指定时间内没有获得结果，就将 Promise 的状态变为reject，否则变为resolve。
const racePromise = Promise.race([
    new Promise(function (resolve, reject) {
        setTimeout(() => resolve("请求结果返回"), 6000)
    }),
    new Promise(function (resolve, reject) {
        setTimeout(() => reject(new Error('request timeout')), 5000)
    })
]);

racePromise.then(console.log)
    .catch(console.error);


// 下面是一个例子，如果指定时间内没有获得结果，就将 Promise 的状态变为reject，否则变为resolve。
const racePromise2 = Promise.race([
    new Promise(function (resolve, reject) {
        setTimeout(() => resolve("请求结果返回"), 6000)
    }),
    new Promise(function (resolve, reject) {
        setTimeout(() => reject(new Error('request timeout')), 5000)
    })
]);

racePromise.then(console.log)
    .catch(console.error);

