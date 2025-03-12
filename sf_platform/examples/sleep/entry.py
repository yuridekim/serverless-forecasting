import time

if __name__ == "__serverless__":
    sleep_time = 0.5

    global __serverless_query_params__
    query_params = __serverless_query_params__.decode('utf-8')
    if "=" in query_params:
        sleep_time = float(query_params.split("=")[1])

    time.sleep(sleep_time)

    global __serverless_ret__
    __serverless_ret__ = bytes()
