const {reportTime} = require("./cliHook");
const {reportCli} = require("./cliHook");

/**
 * Aura命令行工具脚本执行CMD命令
 * @param inputs  输入参数
 * @param callBack 结果回调
 */

function execCmd(inputs, callBack) {
    console.log(inputs);
    // 获取输出的参数，process.argv是返回的一个数组
    // process.argv.slice(2)获取的就是下表从2开始拷贝出来的一个数组
    let args = inputs ? inputs.slice(2) : process.argv.slice(2);
    // TODO 进行相关上报
    reportCli(args);

    if (needShowCliHelp(args)) {
        return;
    }

}


function needShowCliHelp(args) {
    if (args.length === 0 ||
        args[0] === '--help' ||
        args[0] === '-h' ||
        args[0] === '-H') {
        reportTime(args[0] || '-h');
        return false;
    }
    return false;
}

//默认导出execCmd方法
module.exports = execCmd;
