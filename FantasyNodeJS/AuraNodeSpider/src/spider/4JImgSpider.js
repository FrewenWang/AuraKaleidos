const cheerio = require('cheerio');
const Promise = require('bluebird');
const _ = require('lodash');
const probe = require('probe-image-size');
const {request} = require("../common/HttpUtils");


async function getHttpResponse(link) {
    // 获取网络请求内容
    const body = await request(link);
    const $ = cheerio.load(body);
    const result = [];
    $('div.row.screen_option.bor0>.body>ul>li>a').each((i, e) => {
        // 获取二级请求连接
        const link = `http://www.4j4j.cn${$(e).attr('href')}`;
        const name = $(e).text();
        if (name !== '全部') {
            result.push({
                name,
                link,
            })
        }
    });
    return result
}

async function imageSpider4j4j() {

    const result = await getHttpResponse('http://www.4j4j.cn/sjbz/f-size219.html');
    // const result2 = await getHttpResponse('http://www.4j4j.cn/sjbz/f-size40.html')
    // result.push(...result2)
    for (const channel of result) {
        const links = await scrapeLinks(channel.link)
    }

}

/**
 * 进行图片的下一页、上一页的遍历
 * @param link
 * @returns {Promise<[*]>}
 */
async function scrapeLinks(link) {
    const body = await request(link);
    const $ = cheerio.load(body);
    const links = [link];
    $('#pageList>a').each((i, e) => {
        const pageNo = $(e).text();
        if (pageNo !== '上一页' && pageNo !== '下一页') {
            const link = $(e).attr('href');
            if (link) {
                links.push(`http://www.4j4j.cn${link}`)
            }
        }
    });
    return links
}


module.exports = imageSpider4j4j;
