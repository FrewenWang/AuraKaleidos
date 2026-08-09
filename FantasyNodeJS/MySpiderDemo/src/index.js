const cheerio = require('cheerio');

async function visitSite(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`request failed: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

function analyzeMain(html) {
  const $ = cheerio.load(html);
  const links = [];
  $('li.left-list_li > a, .hot.public-box a, .channel.public-box a').each((_index, element) => {
    const href = $(element).attr('href');
    if (href) links.push(href);
  });
  return links;
}

async function main() {
  const html = await visitSite('http://www.mm131.com/');
  console.log(`Found ${analyzeMain(html).length} links`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

module.exports = {analyzeMain, main, visitSite};
