// 文章参考：http://objcer.com/2017/10/12/async-await-with-forEach/


// 我们有多种方法来遍历 JavaScript 的数组或者对象，而它们之间的区别非常让人疑惑。Airbnb 编码风格禁止使用 for/in 与 for/of，你知道为什么吗？


// 这篇文章将详细介绍以下 4 种循环语法的区别：

// for (let i = 0; i < arr.length; ++i)
// arr.forEach((v, i) => { /* ... */ })
// for (let i in arr)
// for (const v of arr)

// 使用for和for/in，我们可以访问数组的下标，而不是实际的数组元素值：
let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 14, 15, 16, 16]

console.log("==============for======begin=========")
for (let i = 0; i < arr.length; ++i) {
    console.log(arr[i]);
}
console.log("==============for======end=========")

console.log("==============for===int===begin=========")
for (let i in arr) {
    console.log(arr[i]);
}
console.log("==============for===int===end=========")


// 使用for/of，则可以直接访问数组的元素值：
console.log("==============for===of===begin=========")
for (const v of arr) {
    console.log(v);
}
console.log("==============for===of===end=========")


console.log("==============forEach=====begin=========")
arr.forEach((v, i) => {

});
console.log("==============forEach=====end=========")


var getNumbers = () => {
    return Promise.resolve([1, 2, 3])
}

var multi = num => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (num) {
                resolve(num * num)
            } else {
                reject(new Error('num not specified'))
            }
        }, 1000)
    })
}


async function test () {
    var nums = await getNumbers()
    nums.forEach(async x => {
        var res = await multi(x)
        console.log(res)
    })
}
test();
