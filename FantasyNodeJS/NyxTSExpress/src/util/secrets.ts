import logger from "./logger";
import dotenv from "dotenv";
import fs from "fs";

if (fs.existsSync(".env")) {
  logger.debug("Using .env file to supply config environment variables");
  dotenv.config({ path: ".env" });
} else {
  logger.debug("No .env file found; using process environment variables");
}
// 判断Node的环境
export const ENVIRONMENT = process.env.NODE_ENV;
// 判断是否是生产环境
const prod = ENVIRONMENT === "production"; // Anything else is treated as 'dev'

export const SESSION_SECRET = process.env["SESSION_SECRET"];
export const MONGODB_URI = prod
  ? process.env["MONGODB_URI"]
  : process.env["MONGODB_URI_LOCAL"];
export const APP_BASE_URL = (process.env["APP_BASE_URL"] || `http://localhost:${process.env.PORT || 3000}`)
  .replace(/\/$/, "");

if (!SESSION_SECRET) {
  logger.error("No client secret. Set SESSION_SECRET environment variable.");
  process.exit(1);
}

if (!MONGODB_URI) {
  if (prod) {
    logger.error(
      "No mongo connection string. Set MONGODB_URI environment variable."
    );
  } else {
    logger.error(
      "No mongo connection string. Set MONGODB_URI_LOCAL environment variable."
    );
  }
  process.exit(1);
}
