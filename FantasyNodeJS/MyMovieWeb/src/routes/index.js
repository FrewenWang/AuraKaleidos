var express = require('express');
var router = express.Router();

/* GET home page. */
/*捕获到之前的路由设置
 * 然后对路由请求进行页面映射*/
router.get('/', function (req, res, next) {
    res.render('index', {title: '视频小站'});
});


/* GET home page. */
/*捕获到之前的路由设置
 * 然后对路由请求进行页面映射*/
router.get('/movieDetail', function (req, res, next) {
    res.render('movie_detail', {title: ' 视频详情页'});
});
//导出这个变量
module.exports = router;
