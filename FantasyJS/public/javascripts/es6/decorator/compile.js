/**
 * 方法一：require hook
 * 调用这个方法的node compile.js
 */
require('babel-register')({
    plugins: ['transform-decorators-legacy']
});
// require("./MultiProperties");
require("./DemoDecorator");
