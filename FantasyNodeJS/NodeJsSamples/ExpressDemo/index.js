const express = require('express');
var app = express();
var path = require('path');

app.get('/', function (req, res) {
    res.send('Hello World!')
})
app.post('/', function (req, res) {
    res.send('Got a POST request')
})

app.put('/user', function (req, res) {
    res.send('Got a PUT request at /user')
})
app.delete('/user', function (req, res) {
    res.send('Got a DELETE request at /user')
})

//引入服务器的静态文件
// app.use(express.static('public'))
//给public文件夹设置一个二级地址
// app.use('/image', express.static('public'));

app.use('/static', express.static(path.join(__dirname, 'public')));

app.listen(3000, function () {
    console.log('Example app listening on port 3000!')
})