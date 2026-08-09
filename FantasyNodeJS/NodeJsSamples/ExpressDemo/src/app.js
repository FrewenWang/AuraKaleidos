const express = require('express');
const path = require('path');

const app = express();

const handlers = {
  getRoot: (_request, response) => response.send('Hello World!'),
  postRoot: (_request, response) => response.send('Got a POST request'),
  putUser: (_request, response) => response.send('Got a PUT request at /user'),
  deleteUser: (_request, response) => response.send('Got a DELETE request at /user'),
};

app.get('/', handlers.getRoot);
app.post('/', handlers.postRoot);
app.put('/user', handlers.putUser);
app.delete('/user', handlers.deleteUser);
app.use('/static', express.static(path.join(__dirname, '..', 'public')));

module.exports = app;
module.exports.handlers = handlers;
