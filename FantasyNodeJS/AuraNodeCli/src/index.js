#!/usr/bin/env node

const fs = require('fs');
const execCmd = require('./tools/execCmd');

// http://nodejs.cn/api/process.html
// https://javascript.ruanyifeng.com/nodejs/process.html
// err <Error> 未捕获的异常。
// origin <string> 表明异常是来自未处理的拒绝还是来自同步的错误。
// 可以是 'uncaughtException' 或 'unhandledRejection'。
process.on('uncaughtException', (err, origin) => {
    if (err) {
        console.log("AuraNodeCli", err)
    }
    //使用标准错误流：process.stderr
    fs.writeSync(
        process.stderr.fd,
        `捕获的异常: ${err}\n` +
        `异常的来源: ${origin}`
    );
    process.exit(1);
});

/// 执行输入的Aura-Node-Command-Line-Interface的参数
execCmd(process.argv);
