import docker
import os
import random
import shutil
import subprocess
import tempfile
import uuid
from distutils.dir_util import copy_tree
from typing import Optional


class ServerlessPlatform:
    _docker_client: docker.DockerClient
    _template_path: str  # Path to Docker template (usually in a directory called template)
    _functions: dict[str, str]  # Mapping from function name to image name

    def __init__(self, template_path: str = "template/"):
        self._docker_client = docker.from_env()
        self._template_path = template_path
        self._functions = dict()  # TODO persistence of image names?

    def __del__(self):
        for image_name in self._functions.values():
            self._docker_client.images.remove(image=image_name)

    def register_function(self, function_name: str, python_file: str, requirements_file: str) -> None:
        function_name = function_name.strip()
        if len(function_name.split()) != 1 or function_name in self._functions:
            raise Exception("Invalid function name")

        with tempfile.TemporaryDirectory() as docker_dir:
            copy_tree(self._template_path, docker_dir)
            shutil.copy(python_file, os.path.join(docker_dir, "entry.py"))
            shutil.copy(requirements_file, os.path.join(docker_dir, "requirements.txt"))

            image_name = f"serverless/{function_name}"
            image, logs = self._docker_client.images.build(
                path=docker_dir,
                tag=image_name
            )

            self._functions[function_name] = image_name

    def run_function(self, function_name: str,
                     method: str = "GET", header: str = "", data: str = "", query_params: Optional[str] = None
                     ) -> subprocess:
        """
        If there is a warm instance (container) available, directly curl

        If not, create a new container, then curl

        curl -X {method} -H {header} -d '{data}' http://localhost:<port>/?{query_params}
        """

        function_name = function_name.strip()
        if function_name not in self._functions:
            raise Exception("Function not found")

        # TODO: check if there is a warm instance available (using Redis)
        # (for now, i assume it doesn't exist and create a new one)

        container_name = f"serverless_{function_name}_{uuid.uuid4()}"
        host_port = random.randint(49152, 65535)
        container = self._docker_client.containers.run(
            image=self._functions[function_name],
            detach=True,
            name=container_name,
            ports={'80/tcp': host_port},  # random host port (None) is not working :(, manually choose a random port
        )

        # TODO: integrate with redis, remove the container if it's done running and we don't need it warmed

        # host_port = container.attrs['HostConfig']['PortBindings']['80/tcp'][0]['HostPort']

        full_params = ""
        if query_params is not None:
            full_params = f"?{query_params}"
        script = (
            # wait for container to come online, retry for up to 3 seconds
            f"while ! curl -s -X {method} -H '{header}' -d '{data}' http://localhost:{host_port}/{full_params} ; do\n"
            "((c++)) && ((c==30)) && break\n"
            "  sleep 0.1\n"
            "done\n"
            f""
        )

        return subprocess.Popen(["bash", "-c", script])
