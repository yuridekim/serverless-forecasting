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
    _docker_client_lock = threading.Lock
    _docker_client: docker.DockerClient
    _threadpool: concurrent.futures.ThreadPoolExecutor  # thread pool to run functions simultaneously
    _template_path: str  # Path to Docker template (usually in a directory called template)
    _image_names: ThreadSafeDict[str, str]  # Mapping from function name to image name
    _default_warm_periods: ThreadSafeDict[str, int]  # Default warming period (s) for a non-permanently-warmed container
    _instance_expirations: ThreadSafeDict[str, dict[str, int]]  # Mapping from function to the expiration timestamps
    _available_instances: ThreadSafeDict[str, list[str]]  # Mapping from function name to available container names
    _function_times: ThreadSafeDict[str, dict[str, float]] # dictionary for execution times

    def __init__(self, template_path: str = "sf_platform/template/"):
        self._docker_client_lock = threading.Lock()
        self._docker_client = docker.from_env()
        self._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=256)
        self._template_path = template_path
        self._image_names = ThreadSafeDict()  # TODO persistence of image names?
        self._default_warm_periods = ThreadSafeDict()
        self._instance_expirations = ThreadSafeDict()
        self._available_instances = ThreadSafeDict()
        self._function_times = ThreadSafeDict()

        asyncio.create_task(self._prune())  # begin the pruning loop

    def __del__(self):
        # Delete all containers (expired or not) in _instance_expirations
        all_containers = []
        with self._instance_expirations as instance_expirations:
            for _, container_dict in instance_expirations.items():
                all_containers += list(container_dict.keys())

        all_images = []
        with self._image_names as image_names:
            all_images += list(image_names.values())

        with self._docker_client_lock:
            for container_name in all_containers:
                try:
                    container = self._docker_client.containers.get(container_name)
                    container.stop(timeout=3)
                except docker.errors.NotFound:
                    # Container might already be removed
                    pass

            for image_name in all_images:
                try:
                    self._docker_client.images.remove(image=image_name)
                except docker.errors.ImageNotFound:
                    pass

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
                _, _ = self._docker_client.images.build(
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
                
            with self._function_times as function_times:
                function_times[function_name] = {}

    async def run_function(self, function_name: str,
                           method: str = "GET", headers: Optional[dict[str, str]] = None, data: str = "", query_params: Optional[str] = None
                           ) -> bytes:
        """
        If there is a warm instance (container) available, directly curl

        If not, create a new container, then send a request
        """
        
        # add tracing for function entry time
        request_id = f"{function_name}_{time.time()}_{random.randint(0, 10000)}"
        with self._function_times as function_times:
            function_times[function_name][request_id] = time.time()
        
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._threadpool,
            self._run_function,
            function_name, method, headers, data, query_params)

        result = await future
        
        # add tracing for function exit time
        with self._function_times as function_times:
            exit_time = time.time()
            entry_time = function_times[function_name].pop(request_id, None)
            if entry_time:
                exec_time = exit_time - entry_time
                print(f"Function {function_name} execution time: {exec_time:.4f}s (request ID: {request_id})")
        
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
        expiration = int(time.time()) + warm_period if warm_period >= 0 else -1

        return self._create_new_container(function_name, expiration)

    def _return_container(self, function_name: str, container_name: str):
        """
        Tell the datastructure the container is available to be used again
        """
        # Renew the timeout
        with self._default_warm_periods as default_warm_periods:
            warm_period = default_warm_periods[function_name]

        should_remove_container = False
        with self._instance_expirations as instance_expirations:
            expiration = instance_expirations[function_name][container_name]
            if expiration > 0 and warm_period == 0:
                # Special handling for no-keep-warm, regular expirations are caught through _prune
                should_remove_container = True
                del instance_expirations[function_name][container_name]
            elif expiration > 0:
                instance_expirations[function_name][container_name] = (int(time.time()) + warm_period
                                                                       if warm_period >= 0 else -1)

        if should_remove_container:
            self._delete_container(container_name)
        else:
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

            instance_expirations[function_name][container_name] = expiration

        with self._docker_client_lock:
            _ = self._docker_client.containers.run(
                image=self._image_names[function_name],
                detach=True,
                name=container_name,
                ports={'80/tcp': host_port},
                auto_remove=True,  # when Flask is stopped, remove the container (we never restart stopped containers)
            )

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

        # Implementation for permanent warm instances
        warm_cnt = 0
        container_names = []
        
        # Count current permanently warm instances
        with self._instance_expirations as instance_expirations:
            function_instances = instance_expirations[function_name]
            for container_name, expiration in function_instances.items():
                if expiration == -1:
                    warm_cnt += 1
                    container_names.append(container_name)
        
        if warm_cnt < num_concurrent:
            instances_to_create = num_concurrent - warm_cnt
            for _ in range(instances_to_create):
                # make new permanently warm instances
                _, container_name = self._create_new_container(function_name, -1)
                with self._instance_expirations as instance_expirations:
                    instance_expirations[function_name][container_name] = -1

                with self._available_instances as available_instances:
                    available_instances[function_name].append(container_name)
        
        # in case of too many instances
        elif warm_cnt > num_concurrent:
            instances_to_expire = warm_cnt - num_concurrent
            current_time = int(time.time())  # Expire them now so they are deleted after finished running
            
            # Set expiration
            with self._instance_expirations as instance_expirations:
                function_instances = instance_expirations[function_name]
                expired_count = 0
                
                for container_name in container_names:
                    if expired_count >= instances_to_expire:
                        break

                    if container_name in function_instances:
                        function_instances[container_name] = current_time
                        expired_count += 1

    async def set_default_warm_period(self, function_name: str, warm_period: int) -> None:
        """
        Set default warm period for a function (seconds)
        """
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._threadpool,
            self._set_default_warm_period,
            function_name, warm_period
        )

        await future

    def _set_default_warm_period(self, function_name: str, warm_period: int) -> None:
        function_name = function_name.strip()
        if function_name not in self._image_names:
            raise Exception("Function not found")

        with self._default_warm_periods as default_warm_periods:
            prev_warm_period = default_warm_periods[function_name]
            default_warm_periods[function_name] = warm_period

        difference = warm_period - prev_warm_period
        if difference == 0:
            return

        with self._instance_expirations as instance_expirations:
            instance_expiration_dict = instance_expirations[function_name]
            for instance_name, expiration in instance_expiration_dict.items():
                if expiration > 0:
                    instance_expiration_dict[instance_name] += difference

    def _delete_container(self, container_name: str) -> None:
        with self._docker_client_lock:
            try:
                self._docker_client.containers.get(container_name).stop(timeout=3)
            except docker.errors.NotFound:
                pass

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
        current_time = int(time.time())
        containers_to_delete = []

        # Find expired containers
        with self._instance_expirations as instance_expirations:
            for function_name, container_dict in instance_expirations.items():
                for container_name, expiration in container_dict.items():
                    # check if expired
                    if 0 < expiration < current_time:
                        containers_to_delete.append((function_name, container_name))

        # Check if container is available, if so, make it unavailable, otherwise cancel the deletion
        containers_removed_from_available = []
        with self._available_instances as available_instances:
            for function_name, container_name in containers_to_delete:
                if container_name in available_instances[function_name]:
                    containers_removed_from_available.append((function_name, container_name))
                    available_instances[function_name].remove(container_name)
        containers_to_delete = containers_removed_from_available

        # remove expired containers from instance_expirations
        with self._instance_expirations as instance_expirations:
            for function_name, container_name in containers_to_delete:
                del instance_expirations[function_name][container_name]

        # delete the containers
        for _, container_name in containers_to_delete:
            self._delete_container(container_name)
