var express = require('express');
var router = express.Router();

/* GET users listing. */
router.get('/', function (req, res, next) {
    res.render('admin', {title: ' 后台管理页'});
});

/* GET users listing. */
router.get('/movie_list', function (req, res, next) {
    res.render('movie_list', {title: '后台视频列表'});
});


/* GET users listing. */
router.get('/add_movie', function (req, res, next) {
    res.render('movie_add', {title: '后台添加视频'});
});

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

module.exports = router;
