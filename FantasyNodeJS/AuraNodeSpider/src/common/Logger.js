const winston = require('winston');
const {createLogger, transports, format} = winston;
const DailyRotateFile = require('winston-daily-rotate-file');
const moment = require('moment')
const fs = require('fs');
const path = require('path');
const LOG_DIR = path.resolve(__dirname, '../logs');

// 创建目录
if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR)
}

// 格式化日志打印
const formatter = format.printf(info => {
    if (info instanceof Error) {
        info.message = info.stack || info.message
    }
    let level = info.level.toUpperCase();
    const log = `${moment().format('YYYY-MM-DD HH:mm:ss.SSS')} ${level} ${
        info.message
    }`;
    return log
});


const logger = createLogger({
    levels: winston.config.syslog.levels,
    format: formatter,
    transports: [
        new transports.File({
            filename: `${LOG_DIR}/error.log`,
            level: 'error',
        }),
        new DailyRotateFile({
            level: 'info',
            filename: `${LOG_DIR}/%DATE%.log`,
            datePattern: 'YYYY-MM-DD',
            zippedArchive: false,
            maxSize: '100m',
            options: {
                flags: 'a',
            },
            maxRetries: 3,
        }),
    ],
    exitOnError: false,
});


if (process.env.NODE_ENV !== 'production') {
    logger.add(new transports.Console({level: 'debug'}))
}

module.exports = logger;

module.exports.getLevelFromHttpStatus = function (status) {
    let level = 'info';
    if (status >= 100) level = 'info';
    if (status >= 400) level = 'warn';
    if (status >= 500) level = 'error';
    return level
};
