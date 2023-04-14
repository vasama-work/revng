#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import os
from socket import AF_UNIX, SOCK_STREAM, getaddrinfo, socket
from typing import Mapping, Protocol

from psutil import Process

from revng.cli.commands_registry import Options
from revng.cli.support import popen


class DaemonHandler(Protocol):
    url: str

    async def wait_for_start(self):
        ...

    def terminate(self) -> int:
        ...


class ExternalDaemonHandler(DaemonHandler):
    def __init__(self, url):
        self.url = url

    async def wait_for_start(self):
        return

    def terminate(self):
        return 0


class InternalDaemonHandler(DaemonHandler):
    def __init__(self, url, options: Options, env: Mapping[str, str]):
        self.url = url
        self.process = popen(
            ["revng", "daemon", "--uvicorn-args=--log-level error", "-b", url], options, env
        )
        assert not isinstance(self.process, int)

    def check_socket_up(self) -> bool:
        if self.url.startswith("unix:"):
            family, type_ = (AF_UNIX, SOCK_STREAM)
            addr = self.url.removeprefix("unix:")
        else:
            host, port = self.url.rsplit(":", 1)
            family, type_, _, _, addr = getaddrinfo(host, int(port))  # type: ignore

        try:
            with socket(family, type_) as sock:
                sock.connect(addr)
        except OSError:
            return False

        return True

    async def wait_for_start(self):
        while True:
            if self.check_socket_up():
                return
            await asyncio.sleep(1.0)

    def terminate(self):
        clean_url = self.url.removeprefix("unix:")
        ps_process = Process(os.getpid())
        target_proc = None

        for proc in ps_process.children(recursive=True):
            cmdline = proc.cmdline()
            if cmdline[0].endswith("python3") and clean_url in cmdline:
                target_proc = proc
                break

        if target_proc is None:
            raise ValueError("Unable to find daemon process")

        target_proc.terminate()
        return self.process.wait()
