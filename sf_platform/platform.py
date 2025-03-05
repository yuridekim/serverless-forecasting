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
    _image_names: ThreadSafeDict[str, str]  # Mapping from function name to image name
    _default_warm_periods: ThreadSafeDict[str, int]  # Default warming period (s) for a non-permanently-warmed container
    _instance_expirations: ThreadSafeDict[str, dict[str, int]]  # Mapping from function to the expiration timestamps
    _available_instances: ThreadSafeDict[str, list[str]]  # Mapping from function name to available container names

    def __init__(self, template_path: str = "sf_platform/template/"):
        self._docker_client_lock = threading.Lock()
        self._docker_client = docker.from_env()
        self._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=256)
        self._template_path = template_path
        self._image_names = ThreadSafeDict()  # TODO persistence of image names?
        self._default_warm_periods = ThreadSafeDict()
        # TODO: per-function granularity on lock? (up to you)
        self._instance_expirations = ThreadSafeDict()
        self._available_instances = ThreadSafeDict()

        asyncio.create_task(self._prune())  # begin the pruning loop

    def __del__(self):
        # TODO delete all containers (expired or not) in _instance_expirations

        with self._docker_client_lock:
            with self._image_names as image_names:
                for image_name in image_names.values():
                    self._docker_client.images.remove(image=image_name)

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
        if len(function_name.split()) != 1 or function_name in self._image_names:
            raise Exception("Invalid function name")

        with tempfile.TemporaryDirectory() as docker_dir:
            copy_tree(self._template_path, docker_dir)
            shutil.copy(python_file, os.path.join(docker_dir, "entry.py"))
            shutil.copy(requirements_file, os.path.join(docker_dir, "requirements.txt"))

            image_name = f"serverless/{function_name}"
            with self._docker_client_lock:
                image, logs = self._docker_client.images.build(
                    path=docker_dir,
                    tag=image_name,
                    rm=True  # remove intermediate containers made during the creation of the image
                )

            with self._image_names as image_names:
                image_names[function_name] = image_name

            with self._default_warm_periods as default_warm_periods:
                default_warm_periods[function_name] = 600  # 10 minutes

            with self._available_instances as available_instances:
                available_instances[function_name] = list()

            with self._instance_expirations as instance_expirations:
                instance_expirations[function_name] = dict()

    async def run_function(self, function_name: str,
                           method: str = "GET", headers: Optional[dict[str, str]] = None, data: str = "", query_params: Optional[str] = None
                           ) -> bytes:
        """
        If there is a warm instance (container) available, directly curl

        If not, create a new container, then send a request
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
        if function_name not in self._image_names:
            raise Exception("Function not found")

        host_port, container_name = self._get_container(function_name)

        full_url = f"http://localhost:{host_port}/"
        if query_params is not None:
            full_url += f"?{query_params}"

        response = retry_request(
            method,
            full_url,
            headers,
            data
        )

        self._return_container(function_name, container_name)

        if not response:
            raise Exception("Attempts to request data failed")
        return response.content

    def _get_container(self, function_name: str) -> tuple[int, str]:
        """
        Get a container used to run the function. If one is not immediately available, create a new one.
        """
        available_instance = None
        with self._available_instances as available_instances:
            instances = available_instances[function_name]
            if len(instances) > 0:
                available_instance = instances.pop()

        if available_instance is not None:
            host_port = int(available_instance.split("_")[2])  # third part of the name is the port number
            return host_port, available_instance

        # a new container made through get_container is not a permanently-warmed instance, use default expiration
        with self._default_warm_periods as default_warm_periods:
            warm_period = default_warm_periods[function_name]
        expiration = int(time.time()) + warm_period if warm_period > 0 else -1

        return self._create_new_container(function_name, expiration)

    def _return_container(self, function_name: str, container_name: str):
        """
        Tell the datastructure the container is available to be used again
        """
        # Renew the timeout
        with self._default_warm_periods as default_warm_periods:
            warm_period = default_warm_periods[function_name]

        with self._instance_expirations as instance_expirations:
            expiration = instance_expirations[function_name][container_name]
            if expiration > 0:
                instance_expirations[function_name][container_name] = (int(time.time()) + warm_period
                                                                       if warm_period > 0 else -1)

        with self._available_instances as available_instances:
            available_instances[function_name].append(container_name)

    def _create_new_container(self, function_name: str, expiration: int) -> tuple[int, str]:
        """
        Create a new docker container, return its host port and container name
        """
        # get a container name with a unique port
        with self._instance_expirations as instance_expirations:
            while True:
                host_port = random.randint(49152, 65535)
                container_name = f"serverless_{function_name}_{host_port}"
                if container_name not in instance_expirations[function_name]:
                    break

        with self._docker_client_lock:
            container = self._docker_client.containers.run(
                image=self._image_names[function_name],
                detach=True,
                name=container_name,
                ports={'80/tcp': host_port},
                auto_remove=True,  # when Flask is stopped, remove the container (we never restart stopped containers)
            )

        with self._instance_expirations as instance_expirations:
            instance_expirations[function_name][container_name] = expiration

        return host_port, container_name

    async def set_permanently_warm_instances(self, function_name: str, num_concurrent: int) -> None:
        """
        Sets the minimum number of warm instances for a function at any given time
        """
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._threadpool,
            self._set_permanently_warm_instances,
            function_name, num_concurrent
        )

        await future

    def _set_permanently_warm_instances(self, function_name: str, num_concurrent: int) -> None:
        function_name = function_name.strip()
        if function_name not in self._image_names:
            raise Exception("Function not found")

        # TODO: implement the function
        # check _instance_expirations for the number of instances that have expiration -1 (permanently warmed instance)
        # if there are not enough: create new instances (with expiration -1) as necessary
        # if there are too many: set an expiration of int(time.time()) for some of the instances

    async def _prune(self):
        await asyncio.sleep(5)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._threadpool,
            self.__prune
        )

        await future

        asyncio.create_task(self._prune())

    def __prune(self) -> None:
        # TODO: implement this function
        # loop through the functions, and inner loop through the instances (you can use _instance_expirations for this)
        # if an instance's timestamp is not -1 (warm forever) and also in the past:
        #   call stop() on the container (which will automatically delete it)
        #   have a guard to NOT stop any containers that are currently unavailable
        pass
