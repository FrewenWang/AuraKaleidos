import moment from "moment";
import request from 'requestretry';

const logger = require("./Logger");

/**
 * 网络请求模板方法
 * @param url 请求URL
 * @param config
 * @returns {Promise<string>}
 */
exports.request = function (url, config = {}) {
    // 这个方法是返回一个Promise对象
    return new Promise((resolve, reject) => {
        const start = moment();
        // 执行网络请求
        request(
            // 这是网络请求的参数
            Object.assign(
                {
                    url: url,
                    jar: true,
                    gzip: true,
                    followAllRedirects: true,
                    maxRedirects: 5,
                    timeout: 30000,
                    maxAttempts: 5,
                    retryDelay: 5000,
                    retryStrategy: request.RetryStrategies.HTTPOrNetworkError,
                },
                config,
            ),
            (error, response) => {
                const level = error ? 'error' : logger.getLevelFromHttpStatus(response.statusCode);
                const cost = moment.duration(moment().diff(start)).as('seconds');
                logger.log(level, `HttpRequest: ${url} ${response ? response.statusCode : ''} ${cost}s ${error ? error : ''}`);

                // 网络请求的结果返回处理
                if (error) {
                    return reject(error)
                } else if (response.statusCode >= 400) {
                    return reject(new Error(`HttpRequest ${url} failed ${response.statusCode} (${response.attempts}) ${JSON.stringify(response.body)}`))
                }

                try {
                    resolve(JSON.parse(response.body));
                } catch (error) {
                    resolve(response.body)
                }
            }
        );
    });
};
