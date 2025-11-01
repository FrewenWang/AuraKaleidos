/**
 * 定义一个CMD_LIST数组，存储可能出现的命令
 * @type {string[]}
 */
const CMD_LIST = [
    '-v',
    '-V',
    '--version',
    '-h',
    '-H',
    '--help',
];

/**
 * 进行相关上报的工作
 * @param args
 */
function reportCli(args) {
    console.log("reportCli called");
}

function reportTime(cmdName) {
    if (CMD_LIST.includes(cmdName)) {
        console.log("reportTime called with = " + cmdName);
    }
}

// 进行reportCli方法的导出
exports.reportCli = reportCli;
exports.reportTime = reportTime;
