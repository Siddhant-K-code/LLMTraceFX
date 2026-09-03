"""Fakes and fixtures shared by the deployment tests.

The point of the fake Modal module is to let the real entrypoint be
imported and inspected without the Modal SDK, without authentication and
without a network. What the entrypoint decides at import (how many GPUs,
what timeout, whether the staging function gets an accelerator at all) is
exactly what has to be asserted, and those decisions are only observable
by watching the decorators receive them.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

VALID_REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
VALID_DIGEST = "sha256:" + "a" * 64
PINNED_IMAGE = f"lmsysorg/sglang:v0.5.6@{VALID_DIGEST}"
TAG_ONLY_IMAGE = "lmsysorg/sglang:v0.5.6"


def valid_environ(**overrides: str) -> dict[str, str]:
    """A complete, approvable deployment environment.

    Sized so the worst case lands under the budget with room for the
    CPU, memory and storage terms as well as the accelerators.
    """
    environ = {
        "LLMTRACEFX_GLM_MAX_USD": "40.00",
        "LLMTRACEFX_GLM_GPU_TYPE": "H200",
        "LLMTRACEFX_GLM_GPU_COUNT": "4",
        "LLMTRACEFX_GLM_MAX_RUNTIME_SECONDS": "1800",
        "LLMTRACEFX_GLM_MAX_DEPLOYMENT_SECONDS": "3600",
        "LLMTRACEFX_GLM_USD_PER_GPU_HOUR": "1.00",
        "LLMTRACEFX_GLM_USD_PER_CPU_CORE_HOUR": "0.01",
        "LLMTRACEFX_GLM_USD_PER_GIB_MEMORY_HOUR": "0.005",
        "LLMTRACEFX_GLM_STORAGE_USD_PER_GIB_MONTH": "0.02",
        "LLMTRACEFX_GLM_STORAGE_RETENTION_DAYS": "1",
        "LLMTRACEFX_GLM_PRICE_EFFECTIVE_DATE": "2026-08-01",
        "LLMTRACEFX_GLM_PRICE_SOURCE": "https://modal.com/pricing",
        "LLMTRACEFX_GLM_MODEL_REVISION": VALID_REVISION,
        "LLMTRACEFX_GLM_IMAGE": PINNED_IMAGE,
        "LLMTRACEFX_GLM_FRAMEWORK_VERSION": "0.5.6",
        "LLMTRACEFX_GLM_CONTEXT_LENGTH": "131072",
        "LLMTRACEFX_GLM_STARTUP_TIMEOUT_SECONDS": "900",
        "LLMTRACEFX_GLM_AS_OF": "2026-08-30",
    }
    environ.update(overrides)
    return environ


class FakeImage:
    """Records the chained image builder calls without building anything."""

    def __init__(self, label: str, **kwargs: Any) -> None:
        self.label = label
        self.kwargs = kwargs
        self.pip_packages: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.local_dirs: list[tuple[str, str]] = []
        self.local_dir_options: list[dict[str, Any]] = []
        self.working_directory: str | None = None
        self.entrypoint_commands: list[str] | None = None

    def pip_install(self, *packages: str, **_: Any) -> FakeImage:
        self.pip_packages.extend(packages)
        return self

    def env(self, mapping: Mapping[str, str]) -> FakeImage:
        self.env_vars.update(mapping)
        return self

    def add_local_dir(
        self, local_path: str, *, remote_path: str, **kwargs: Any
    ) -> FakeImage:
        self.local_dirs.append((local_path, remote_path))
        self.local_dir_options.append(kwargs)
        return self

    def workdir(self, path: str) -> FakeImage:
        self.working_directory = path
        return self

    def entrypoint(self, commands: list[str]) -> FakeImage:
        self.entrypoint_commands = commands
        return self


class FakeVolume:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class FakeSecret:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.required_keys = tuple(kwargs.get("required_keys", ()))


class FakeRegistration:
    """One decorated function and every decorator argument it received."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.function_kwargs: dict[str, Any] = {}
        self.concurrent_kwargs: dict[str, Any] = {}
        self.asgi_app_kwargs: dict[str, Any] = {}
        self.web_server_args: tuple[Any, ...] = ()
        self.web_server_kwargs: dict[str, Any] = {}


class FakeApp:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.registrations: dict[str, FakeRegistration] = {}

    def _registration(self, func: Any) -> FakeRegistration:
        name = getattr(func, "_fake_name", None) or func.__name__
        return self.registrations.setdefault(name, FakeRegistration(name))

    def function(self, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            registration = self._registration(func)
            registration.function_kwargs = kwargs
            func._fake_name = registration.name
            return func

        return decorator

    def local_entrypoint(self, **_: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator


def build_fake_modal() -> types.ModuleType:
    """A stand-in ``modal`` module with just the surface the app uses."""
    module = types.ModuleType("modal")
    apps: list[FakeApp] = []
    images: list[FakeImage] = []
    volumes: list[FakeVolume] = []
    secrets: list[FakeSecret] = []

    def make_app(name: str, **kwargs: Any) -> FakeApp:
        app = FakeApp(name, **kwargs)
        apps.append(app)
        return app

    class Image:
        @staticmethod
        def debian_slim(**kwargs: Any) -> FakeImage:
            image = FakeImage("debian_slim", **kwargs)
            images.append(image)
            return image

        @staticmethod
        def from_registry(tag: str, **kwargs: Any) -> FakeImage:
            image = FakeImage("from_registry", tag=tag, **kwargs)
            images.append(image)
            return image

    class Volume:
        @staticmethod
        def from_name(name: str, **kwargs: Any) -> FakeVolume:
            volume = FakeVolume(name, **kwargs)
            volumes.append(volume)
            return volume

    class Secret:
        @staticmethod
        def from_name(name: str, **kwargs: Any) -> FakeSecret:
            secret = FakeSecret(name, **kwargs)
            secrets.append(secret)
            return secret

    def concurrent(**kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            name = getattr(func, "_fake_name", None) or func.__name__
            for app in apps:
                app.registrations.setdefault(name, FakeRegistration(name))
                app.registrations[name].concurrent_kwargs = kwargs
            func._fake_name = name
            return func

        return decorator

    def asgi_app(**kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            name = getattr(func, "_fake_name", None) or func.__name__
            for app in apps:
                app.registrations.setdefault(name, FakeRegistration(name))
                app.registrations[name].asgi_app_kwargs = kwargs
            func._fake_name = name
            return func

        return decorator

    def web_server(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            name = getattr(func, "_fake_name", None) or func.__name__
            for app in apps:
                app.registrations.setdefault(name, FakeRegistration(name))
                app.registrations[name].web_server_args = args
                app.registrations[name].web_server_kwargs = kwargs
            func._fake_name = name
            return func

        return decorator

    module.App = make_app  # type: ignore[attr-defined]
    module.Image = Image  # type: ignore[attr-defined]
    module.Volume = Volume  # type: ignore[attr-defined]
    module.Secret = Secret  # type: ignore[attr-defined]
    module.asgi_app = asgi_app  # type: ignore[attr-defined]
    module.concurrent = concurrent  # type: ignore[attr-defined]
    module.web_server = web_server  # type: ignore[attr-defined]
    module.__version__ = "1.0.5-fake"  # type: ignore[attr-defined]
    module._fake_apps = apps  # type: ignore[attr-defined]
    module._fake_images = images  # type: ignore[attr-defined]
    module._fake_volumes = volumes  # type: ignore[attr-defined]
    module._fake_secrets = secrets  # type: ignore[attr-defined]
    return module


APP_MODULE = "llmtracefx.deploy.modal_glm_app"


@contextmanager
def imported_app(
    environ: Mapping[str, str], *, app_module: str = APP_MODULE
) -> Iterator[tuple[Any, types.ModuleType]]:
    """Import the Modal entrypoint against a fake SDK and a given environment.

    Both the fake module and the imported entrypoint are removed
    afterwards so one test cannot observe another's registrations, and so
    a real ``modal`` install (or its absence) is irrelevant to the result.
    """
    import importlib
    import os

    fake = build_fake_modal()
    saved_modal = sys.modules.get("modal")
    saved_app = sys.modules.pop(app_module, None)
    saved_environ = dict(os.environ)
    sys.modules["modal"] = fake
    os.environ.clear()
    os.environ.update(environ)
    try:
        module = importlib.import_module(app_module)
        yield module, fake
    finally:
        sys.modules.pop(app_module, None)
        if saved_app is not None:
            sys.modules[app_module] = saved_app
        if saved_modal is not None:
            sys.modules["modal"] = saved_modal
        else:
            sys.modules.pop("modal", None)
        os.environ.clear()
        os.environ.update(saved_environ)
