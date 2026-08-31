"""Compatibility contract for the original Modal analyzer entrypoint."""

from __future__ import annotations

from _fakes import imported_app


def test_legacy_app_uses_modal_1_5_compatible_decorators() -> None:
    with imported_app({}, app_module="llmtracefx.modal_app") as (_, fake):
        app = fake._fake_apps[0]
        analyze = app.registrations["analyze_trace_modal"]
        web = app.registrations["web_app"]

        assert analyze.function_kwargs["gpu"] == "A10G"
        assert "keep_warm" not in web.function_kwargs
        assert web.function_kwargs["min_containers"] == 1
        assert web.asgi_app_kwargs == {}


def test_legacy_image_copies_the_local_package_into_the_image() -> None:
    with imported_app({}, app_module="llmtracefx.modal_app") as (_, fake):
        image = fake._fake_images[0]

        assert "plotly==7.0.0" in image.pip_packages[0]
        assert image.local_dirs == [("./llmtracefx", "/app/llmtracefx")]
        assert image.local_dir_options == [{"copy": True}]
        assert image.working_directory == "/app"
