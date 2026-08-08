const dotEnv = require('dotenv');


dotEnv.config();

const TAG = "AuraNodeSpider";

async function main() {
    console.debug(TAG, "Spider Start");
}

if (require.main === module) {
    main().catch(error => {
        console.error(TAG, error);
        process.exitCode = 1;
    });
}

module.exports = {main};
