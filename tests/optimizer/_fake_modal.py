"""A fake Modal SDK and app module for offline execution tests.

The real provider package is never imported by these tests. This fake
mirrors the exact API surface the protocol pins (and that was inspected
against modal 1.5.5), so a drift in either the fake or the real SDK
surfaces as a failing capability probe rather than a silent divergence.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any


class FunctionStats:
    def __init__(self, *, backlog: int = 0, num_total_runners: int = 0) -> None:
        self.backlog = backlog
        self.num_total_runners = num_total_runners
        self.num_running_inputs = 0
        self.input_headroom = 0


class FunctionCall:
    def __init__(self, function: Function, arguments: tuple[Any, ...]) -> None:
        self.function = function
        self.arguments = arguments
        self.cancelled = False
        self.terminate_containers = False

    def cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled = True
        self.terminate_containers = terminate_containers
        self.function.app.log.append(("cancel", self.function.key))

    def get(self, timeout: float | None = None, *, index: int = 0) -> Any:
        del index
        self.function.app.timeouts.append((self.function.key, timeout))
        outcome = self.function.outcome(self.arguments)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Function:
    def __init__(self, key: str, app: App, raw: Any, kwargs: dict[str, Any]) -> None:
        self.key = key
        self.app = app
        self.raw = raw
        self.kwargs = kwargs
        self.stats = FunctionStats()

    def outcome(self, arguments: tuple[Any, ...]) -> Any:
        script = self.app.script.get(self.key)
        if callable(script):
            return script(*arguments)
        return script

    def spawn(self, *args: Any, **kwargs: Any) -> FunctionCall:
        del kwargs
        self.app.log.append(("spawn", self.key))
        self.app.calls.append((self.key, args))
        return FunctionCall(self, args)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        del kwargs
        return self.spawn(*args).get()

    def get_current_stats(self) -> FunctionStats:
        self.app.log.append(("stats", self.key))
        if self.app.stats_error is not None:
            raise self.app.stats_error
        return self.stats


class App:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.log: list[tuple[str, Any]] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.timeouts: list[tuple[str, float | None]] = []
        self.tags: dict[str, str] = {}
        self.script: dict[str, Any] = {}
        self.stats_error: BaseException | None = None
        self.registered_web_endpoints: list[str] = []
        self.function_kwargs: dict[str, dict[str, Any]] = {}
        self.entered = 0
        self.exited = 0
        self.enter_error: BaseException | None = None

    def function(
        self,
        _warn_parentheses_missing: Any = None,
        *,
        image: Any = None,
        schedule: Any = None,
        env: Any = None,
        secrets: Any = None,
        gpu: Any = None,
        serialized: Any = None,
        network_file_systems: Any = None,
        volumes: Any = None,
        cpu: Any = None,
        memory: Any = None,
        ephemeral_disk: Any = None,
        min_containers: Any = None,
        max_containers: Any = None,
        buffer_containers: Any = None,
        scaledown_window: Any = None,
        proxy: Any = None,
        retries: Any = None,
        timeout: Any = None,
        startup_timeout: Any = None,
        name: Any = None,
        is_generator: Any = None,
        cloud: Any = None,
        region: Any = None,
        routing_region: Any = None,
        nonpreemptible: Any = None,
        enable_memory_snapshot: Any = None,
        block_network: Any = None,
        restrict_modal_access: Any = None,
        single_use_containers: Any = None,
        i6pn: Any = None,
        include_source: Any = None,
        experimental_options: Any = None,
        max_inputs: Any = None,
    ) -> Any:
        captured = {
            "image": image,
            "secrets": secrets,
            "volumes": volumes,
            "cpu": cpu,
            "memory": memory,
            "timeout": timeout,
            "retries": retries,
            "min_containers": min_containers,
            "max_containers": max_containers,
            "buffer_containers": buffer_containers,
            "scaledown_window": scaledown_window,
            "max_inputs": max_inputs,
            "single_use_containers": single_use_containers,
            "block_network": block_network,
            "restrict_modal_access": restrict_modal_access,
        }
        if gpu is not None:
            captured["gpu"] = gpu

        def decorator(raw: Any) -> Function:
            key = getattr(raw, "__name__", "anonymous")
            self.function_kwargs[key] = captured
            return Function(key, self, raw, captured)

        return decorator

    @contextmanager
    def run(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        del kwargs
        self.entered += 1
        self.log.append(("app_run_enter", self.name))
        if self.enter_error is not None:
            raise self.enter_error
        try:
            yield self
        finally:
            self.exited += 1
            self.log.append(("app_run_exit", self.name))

    def set_tags(self, tags: Any, *, client: Any = None) -> None:
        del client
        self.tags = dict(tags)
        self.log.append(("set_tags", tuple(sorted(self.tags))))


class Image:
    def __init__(self, reference: str) -> None:
        self.reference = reference
        self.pip_packages: list[str] = []

    @classmethod
    def from_registry(cls, tag: str, *args: Any, **kwargs: Any) -> Image:
        del args, kwargs
        return cls(tag)

    @classmethod
    def debian_slim(cls, python_version: str | None = None, **kwargs: Any) -> Image:
        del kwargs
        return cls(f"debian_slim:{python_version}")

    def pip_install(self, *packages: str, **kwargs: Any) -> Image:
        del kwargs
        self.pip_packages.extend(packages)
        return self

    def env(self, values: Any) -> Image:
        del values
        return self


class VolumeManager:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.existing: list[str] = []
        self.delete_error: BaseException | None = None
        self.list_error: BaseException | None = None

    def delete(
        self,
        name: str,
        *,
        allow_missing: bool = False,
        environment_name: str | None = None,
        client: Any = None,
    ) -> None:
        del environment_name, client, allow_missing
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(name)
        self.existing = [item for item in self.existing if item != name]

    def list(self, **kwargs: Any) -> list[Any]:
        del kwargs
        if self.list_error is not None:
            raise self.list_error
        return [types.SimpleNamespace(name=name) for name in self.existing]


class Volume:
    objects = VolumeManager()

    def __init__(self, name: str, *, readonly: bool = False) -> None:
        self.name = name
        self.readonly = readonly
        self.commits = 0

    @classmethod
    def from_name(
        cls, name: str, *, create_if_missing: bool = False, **kwargs: Any
    ) -> Volume:
        del kwargs
        volume = cls(name)
        if create_if_missing and name not in cls.objects.existing:
            cls.objects.existing.append(name)
        return volume

    def with_mount_options(
        self, *, read_only: bool | None = None, sub_path: Any = None
    ) -> Volume:
        del sub_path
        return Volume(self.name, readonly=bool(read_only))

    def commit(self) -> None:
        self.commits += 1


class Secret:
    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Secret:
        raise AssertionError("this protocol must never create or read a Modal Secret")


def concurrent(
    _warn_parentheses_missing: Any = None,
    *,
    max_inputs: int | None = None,
    target_inputs: int | None = None,
) -> Any:
    def decorator(raw: Any) -> Any:
        raw.concurrency_max_inputs = max_inputs
        raw.concurrency_target_inputs = target_inputs
        return raw

    return decorator


@contextmanager
def enable_output():  # type: ignore[no-untyped-def]
    yield


class FunctionTimeoutError(RuntimeError):
    """Mirror of modal.exception.FunctionTimeoutError."""


class SandboxTerminatedError(RuntimeError):
    """Mirror of modal.exception.SandboxTerminatedError."""


def build_fake_modal(version: str = "1.5.5") -> types.ModuleType:
    """Return a module object that satisfies the pinned capability probe."""

    module = types.ModuleType("modal")
    module.__version__ = version  # type: ignore[attr-defined]
    module.App = App  # type: ignore[attr-defined]
    module.Image = Image  # type: ignore[attr-defined]
    module.Volume = Volume  # type: ignore[attr-defined]
    module.Function = Function  # type: ignore[attr-defined]
    module.FunctionCall = FunctionCall  # type: ignore[attr-defined]
    module.Secret = Secret  # type: ignore[attr-defined]
    module.concurrent = concurrent  # type: ignore[attr-defined]
    module.enable_output = enable_output  # type: ignore[attr-defined]
    Volume.objects = VolumeManager()
    return module
