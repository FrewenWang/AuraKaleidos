/**
 * Promise.all()方法用于将多个 Promise 实例，包装成一个新的 Promise 实例。
 */


// const p = Promise.all([p1, p2, p3]);

// 上面代码中，Promise.all()方法接受一个数组作为参数，
// p1、p2、p3都是 Promise 实例，
// 如果不是，就会先调用下面讲到的Promise.resolve方法，将参数转为 Promise 实例，再进一步处理。


// 另外，Promise.all()方法的参数可以不是数组，但必须具有 Iterator 接口，且返回的每个成员都是 Promise 实例。
// p的状态由p1、p2、p3决定，分成两种情况。

// （1）只有p1、p2、p3的状态都变成fulfilled，p的状态才会变成fulfilled，此时p1、p2、p3的返回值组成一个数组，传递给p的回调函数。

// （2）只要p1、p2、p3之中有一个被rejected，p的状态就变成rejected，此时第一个被reject的实例的返回值，会传递给p的回调函数。

const TAG = "Promise.all";
// 下面是一个具体的例子。

// 生成一个Promise对象的数组
const promises = [2, 3, 5, 7, 11, 13].map(function (id) {
    return getJSON('/post/' + id + ".json");
});

Promise.all(promises).then(function (result) {
    console.log(TAG, "result = " + result)
}).catch(function (reason) {
    console.log(TAG, "reason = " + reason)
});

// 上面代码中，promises是包含 6 个 Promise 实例的数组，只有这 6 个实例的状态都变成fulfilled，
// 或者其中有一个变为rejected，才会调用Promise.all方法后面的回调函数。

// 下面是另一个例子。
const databasePromise = connectDatabase();

const booksPromise = databasePromise
    .then(findAllBooks);

const userPromise = databasePromise
    .then(getCurrentUser);

Promise.all([
    booksPromise,
    userPromise
])
    .then(([books, user]) => pickTopRecommendations(books, user));

