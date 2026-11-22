"""Install hints for llmtracefx optional extras."""


def extra_install_hint(extra: str) -> str:
    """Return the commands that install one optional extra, uv first."""
    spec = f"'llmtracefx[{extra}]'"
    return (
        f"`uv tool install {spec}` (in a project: `uv add {spec}`; "
        f"from a source checkout: `uv sync --extra {extra}`; "
        f"with pip: `pip install {spec}`)"
    )
