let s = Symbol();


console.log(typeof s);


// Symbol函数可以接受一个字符串作为参数，表示对 Symbol 实例的描述，
// 主要是为了在控制台显示，或者转为字符串时，比较容易区分。
let s1 = Symbol('foo');
let s2 = Symbol('bar');

console.log(s1); // Symbol(foo)
console.log(s2); // Symbol(bar)

console.log(s1.toString()); // "Symbol(foo)"
console.log(s2.toString()); // "Symbol(bar)"


//如果 Symbol 的参数是一个对象，就会调用该对象的toString方法，将其转为字符串，然后才生成一个Symbol值。
// 我们定义一个对象叫做obj
const obj = {
    toString() {
        return '对象toString';
    }
};
// 然后将这个对象作为参数传给Symbol
const sym = Symbol(obj);
console.log(sym);// Symbol(对象toString)


// 注意，Symbol函数的参数只是表示对当前 Symbol 值的描述，因此相同参数的Symbol函数的返回值是不相等的。

// 没有参数的情况
let s1noParam = Symbol();
let s2noParam = Symbol();

console.log(s1noParam === s2noParam); // false

// 有参数的情况
let s1hasParam = Symbol('foo');
let s2hasParam = Symbol('foo');

console.log(s1hasParam === s2hasParam); // false
// 上面代码中，s1和s2都是Symbol函数的返回值，而且参数相同，但是它们是不相等的。


