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

module.exports = router;
