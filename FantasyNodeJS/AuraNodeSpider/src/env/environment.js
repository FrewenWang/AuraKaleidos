import dotEnv from "dotenv";
import fs from "fs";

const logger = require("../common/Logger");


if (fs.existsSync(".env")) {
    logger.debug("Using .env file to supply config environment variables");
    dotEnv.config({path: ".env"});
} else {
    logger.debug("Using .env file to supply config environment variables");
    dotEnv.config({path: ".env"});
}
/**
 * 判断应用的运行环境
 * @type {string}
 */
export const ENVIRONMENT = process.env.NODE_ENV;
logger.debug(`secrets 判断Node服务的运行环境： ${ENVIRONMENT}`);

const production = ENVIRONMENT === "production"; // Anything else is treated as 'dev'


/**
 * 如果是是生产环境使用MONGODB_URI，否则使用MONGODB_URI_LOCAL
 * @type {string}
 */
export const MONGODB_URI = production ? process.env["MONGODB_URI"] : process.env["MONGODB_URI_LOCAL"];

if (!MONGODB_URI) {
    if (production) {
        logger.error("No mongo connection string. Set MONGODB_URI environment variable.");
    } else {
        logger.error("No mongo connection string. Set MONGODB_URI_LOCAL environment variable.");
    }
    process.exit(1);
}
