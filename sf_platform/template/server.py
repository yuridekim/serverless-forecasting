from flask import Flask


app = Flask(__name__)


with open("entry.py", "r") as src_file:
    src = src_file.read()


@app.route('/')
def run():
    lambda_globals = {
        "__name__": "__serverless__"
    }

    exec(src, lambda_globals)

    if "__serverless_ret__" not in lambda_globals:
        raise Exception("Function completed execution without setting __serverless_ret__")

    return lambda_globals["__serverless_ret__"]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
