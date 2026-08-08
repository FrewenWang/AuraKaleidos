const app = require('./app');

const port = Number(process.env.PORT || 5000);
const server = app.listen(port, () => {
  console.log(`Server running at port: ${port}`);
});

module.exports = server;
