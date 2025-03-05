import asyncio
import concurrent.futures
import docker
import os
import random
import requests
import shutil
import tempfile
import threading
import time
from distutils.dir_util import copy_tree
from typing import Any, Optional


# Credit for ThreadSafeDict: https://stackoverflow.com/a/29532297
class ThreadSafeDict(dict) :
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()


def retry_request(method: str, url: str,
                  headers: Optional[dict[str, str]] = None, data: Optional[dict[str, Any] | str] = None,
                  timeout=3, retry_interval=0.1) -> Optional[requests.Response]:
    """
    Retry a HTTP request until it succeeds or reaches a timeout.

    Args:
        method (str): HTTP method (GET, POST, etc.)
        url (str): Full URL to send the request to
        headers (dict, optional): Headers to send with the request
        data (dict/str, optional): Data to send with the request
        timeout (float, optional): Total timeout in seconds. Defaults to 3.
        retry_interval (float, optional): Time between retries. Defaults to 0.1 seconds.

    Returns:
        requests.Response or None: Response if successful, None if timed out
    """
    start_time = time.time()
    attempts = 0
    max_attempts = int(timeout / retry_interval)

    while time.time() - start_time < timeout:
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data
            )

            # Consider the request successful if it doesn't raise an exception
            return response

        except requests.RequestException:
            # If request fails, wait and continue
            attempts += 1

            # Break if max attempts reached
            if attempts >= max_attempts:
                break

            time.sleep(retry_interval)

    # Return None if all attempts fail
    return None


class ServerlessPlatform:
    _docker_client_lock = threading.Lock  # TODO: do we need this? (up to you)
    _docker_client: docker.DockerClient
    _threadpool: concurrent.futures.ThreadPoolExecutor  # thread pool to run functions simultaneously
    _template_path: str  # Path to Docker template (usually in a directory called template)
    _functions: ThreadSafeDict[str, str]  # Mapping from function name to image name
    _num_instances: ThreadSafeDict[str, int]  # Mapping from function name to number of warmed or running containers
    _available_instances: ThreadSafeDict[str, list[str]]  # Mapping from function name to available container names
    _min_instances: ThreadSafeDict[str, int]  # Set through set_concurrent_warm_instances

    def __init__(self, template_path: str = "template/"):
        self._docker_client_lock = threading.Lock()
        self._docker_client = docker.from_env()
        self._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=256)
        self._template_path = template_path
        self._functions = ThreadSafeDict()  # TODO persistence of image names?
        # TODO: per-function granularity on lock? (up to you)
        self._num_instances = ThreadSafeDict()
        self._available_instances = ThreadSafeDict()
        self._min_instances = ThreadSafeDict()

    def __del__(self):
        with self._docker_client_lock:
            for image_name in self._functions.values():
                self._docker_client.images.remove(image=image_name)
        # TODO delete containers

    async def register_function(self, function_name: str, python_file: str, requirements_file: str) -> None:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._threadpool,
            self._register_function,
            function_name, python_file, requirements_file
        )

        await future

    def _register_function(self, function_name: str, python_file: str, requirements_file: str) -> None:
        function_name = function_name.strip()
        if len(function_name.split()) != 1 or function_name in self._functions:
            raise Exception("Invalid function name")

        with tempfile.TemporaryDirectory() as docker_dir:
            copy_tree(self._template_path, docker_dir)
            shutil.copy(python_file, os.path.join(docker_dir, "entry.py"))
            shutil.copy(requirements_file, os.path.join(docker_dir, "requirements.txt"))

            image_name = f"serverless/{function_name}"
            with self._docker_client_lock:
                image, logs = self._docker_client.images.build(
                    path=docker_dir,
                    tag=image_name
                )

            with self._functions as functions:
                functions[function_name] = image_name

            with self._available_instances as available_instances:
                available_instances[function_name] = list()

            with self._num_instances as num_instances:
                num_instances[function_name] = 0

            with self._min_instances as min_instances:
                self._min_instances[function_name] = 0

    async def run_function(self, function_name: str,
                           method: str = "GET", headers: Optional[dict[str, str]] = None, data: str = "", query_params: Optional[str] = None
                           ) -> bytes:
        """
        If there is a warm instance (container) available, directly curl

        If not, create a new container, then curl

        curl -X {method} -H {header} -d '{data}' http://localhost:<port>/?{query_params}
        """
        
        # TODO: add tracing (function entry time right here)
        
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._threadpool,
            self._run_function,
            function_name, method, headers, data, query_params
        )

        result = await future
        
        # TODO: add tracing (function exit time right here)
        
        return result

    def _run_function(self, function_name: str,
                      method: str, headers: Optional[dict[str, str]], data: str, query_params: Optional[str]) -> bytes:
        function_name = function_name.strip()
        if function_name not in self._functions:
            raise Exception("Function not found")

        # TODO: check if there is a warm instance available (using _available_instances)
        # (for now, i assume it doesn't exist and create a new one)

        host_port = random.randint(49152, 65535)
        container_name = f"serverless_{function_name}_{host_port}"

        with self._docker_client_lock:
            container = self._docker_client.containers.run(
                image=self._functions[function_name],
                detach=True,
                name=container_name,
                ports={'80/tcp': host_port},  # random host port (None) is not working :(, manually choose a random port
                auto_remove=True,  # when Flask is stopped, remove the container (we never restart stopped containers)
            )
        # host_port = container.attrs['HostConfig']['PortBindings']['80/tcp'][0]['HostPort']

        full_url = f"http://localhost:{host_port}/"
        if query_params is not None:
            full_url += f"?{query_params}"

        response = retry_request(
            method,
            full_url,
            headers,
            data
        )

        # TODO: do not always delete the container
        with self._docker_client_lock:
            container.stop(timeout=3)

        if not response:
            raise Exception("Attempts to request data failed")
        return response.content

    def set_concurrent_warm_instances(self, function_name: str, num_concurrent: int) -> None:
        """
        Sets the minimum number of warm instances for a function at any given time
        """
        function_name = function_name.strip()
        if function_name not in self._functions:
            raise Exception("Function not found")

        with self._min_instances as min_instances:
            min_instances[function_name] = num_concurrent
        # TODO: implement the function (check _num_instances, create new instances as necessary)

    async def _prune(self):
        # TODO: implement this function
        # loop through the functions, check if _num_isntances exceeds _min_instances, if so, prune some available instances
        pass
