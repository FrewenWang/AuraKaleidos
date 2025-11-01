/**
 * Created by Frewen on 2016/11/28.
 */

//使用严格模式
// 'use strict';
const request = require('request');
//导入文件依赖库
const fs = require('fs');
const Promise = require('bluebird');
const cheerio = require('cheerio');
try {
    // fs.statSync(path) 初始化文件夹状态
    // 同步 stat(). 返回 fs.Stats 的实例。
    fs.statSync('./download');
} catch (e) {
    // fs.mkdirSync(path[, mode])
    // 同步 mkdir().创建下载目录
    fs.mkdirSync('./download');
}

/**
 * ECMAScript 6中多了两个定义变量的关键词，
 * 一个是let，另一个是const，后者顾名思义就是常量定义，
 * 前者的作用域范围是块级的。
 */
let bar;

visitSite('http://www.mm131.com/').then(html => analyzeMain(html))

/**
 *请求爬取的网页地址
 * @param url
 */
function visitSite(url) {
    //数据返回解析操作
    return new Promise((resolve, reject) => {
        request(url, function (error, response, body) {
            if (error) {
                return reject(error);
            } else if (response.statusCode != 200) {
                return reject('request fail ' + response.statusCode);
            }
            console.log("response.body==" + response.body);
            resolve(response.body);
        });
    });
}

/**
 *进行请求下来的HTML进行数据分析
 * @param html
 */
function analyzeMain(html) {
    console.log("analyzeMain===html==="+html)
    return new Promise((resolve, reject) => {
        //将html文件传入cheerio进行解析
        let $ = cheerio.load(html);
        let arr = [];
        $('li.left-list_li > a').each((i, e) => {
            arr.push($(e).attr('href'));
        });
        $('.hot.public-box ').find('a').each((i, e) => {
            arr.push($(e).attr('href'));
        });
        $('.channel.public-box').find('a').each((i, e) => {
            arr.push($(e).attr('href'));
        })
        resolve(arr);
    });

}

