var express = require('express');
var path = require('path');
var favicon = require('serve-favicon');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');

//引入当前目录下的路由目下中的index
var index = require('./routes/index');
//引入当前目录下的路由目下中的Users
var users = require('./routes/admin');
//引入当前目录下的路由目下中的测试的路由分发
var test = require('./routes/admin');

//我们还可以在这里自定义设置监听的端口
var port = process.env.PORT || 5000;

var app = express();

// view engine setup
//设置视图的根目录，就是根目录下的"views"目录.使用path设置加入当前目录的views
app.set('views', path.join(__dirname, 'views/pages'));
//设置模板引擎为Jade
app.set('view engine', 'jade');


app.listen(port);

console.log("Server running at port:" + port);

// uncomment after placing your favicon in /public
//app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
app.use(logger('dev'));
app.use(bodyParser.json());

app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
//设置bodypath ，可以将表单中的数据格式化
app.use(bodyParser.urlencoded({extended: false}));
//不能再使用下面的方法若干再使用还是回报错
//body-parser deprecated bodyparser
//app.use(bodyParser.urlencoded());
//将bower_components的路径加入进去
app.use(express.static(path.join(__dirname, 'bower_components')));

//todo 设置测试的路由重定向
//设置路由重定向
app.use('/test', test);

app.use('/', index);
//设置路由重定向到users目录
app.use('/admin', users);


// catch 404 and forward to error handler
app.use(function (req, res, next) {
    var err = new Error('Not Found');
    err.status = 404;
    next(err);
});

// error handler
app.use(function (err, req, res, next) {
    // set locals, only providing error in development
    res.locals.message = err.message;
    res.locals.error = req.app.get('env') === 'development' ? err : {};

    // render the error page
    res.status(err.status || 500);
    res.render('error');
});

module.exports = app;
