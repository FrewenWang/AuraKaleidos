from flask import Flask

"""
https://www.runoob.com/flask/flask-tutorial.html
Flask 是一个用 Python 编写的轻量级 Web 应用框架。
Flask 基于 WSGI（Web Server Gateway Interface）和 Jinja2 模板引擎，旨在帮助开发者快速、简便地创建 Web 应用。
Flask 被称为"微框架"，因为它使用简单的核心，用扩展增加其他功能。
"""
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run()
