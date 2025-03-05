import cowsay


if __name__ == "__serverless__":
    global __serverless_ret__
    __serverless_ret__ = cowsay.get_output_string('cow', 'I love serverless computing!')
