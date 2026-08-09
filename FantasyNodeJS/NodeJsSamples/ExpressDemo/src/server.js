const app = require('./app');

const port = Number.parseInt(process.env.PORT || '3000', 10);
const server = app.listen(port, () => {
  console.log(`Express demo listening on http://localhost:${port}`);
});

module.exports = server;
