/**
 * require
 require函数用于在当前模块中加载和使用别的模块，传入一个模块名，返回一个模块导出对象。
 模块名可使用相对路径（以./开头），或者是绝对路径（以/或C:之类的盘符开头）。
 另外，模块名中的.js扩展名可以省略。以下是一个例子。
 */

var foo1 = require('./foo');
var foo2 = require('./foo.js');
var foo3 = require('/home/user/foo');
var foo4 = require('/home/user/foo.js');