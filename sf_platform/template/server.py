from flask import Flask, request


app = Flask(__name__)
HTTP_METHODS = ['GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'CONNECT', 'OPTIONS', 'TRACE', 'PATCH']


with open("entry.py", "r") as src_file:
    src = src_file.read()


@app.route('/', methods=HTTP_METHODS)
def run():
    lambda_globals = {
        "__name__": "__serverless__",
        "__serverless_method__": request.method,
        "__serverless_query_params__": request.query_string,
        "__serverless_headers__": request.headers,
        "__serverless_body__": request.get_data()
    }

    exec(src, lambda_globals)

    if "__serverless_ret__" not in lambda_globals:
        raise Exception("Function completed execution without setting __serverless_ret__")

    return lambda_globals["__serverless_ret__"]


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
