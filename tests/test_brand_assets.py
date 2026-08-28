"""Tests for the brand assets and the shared brand constants.

The identity is committed to the repository rather than generated, so these
tests check the properties a broken regeneration would silently lose: that
every mark still parses, still carries an accessible name, and still holds
its declared colours, and that the constants the HTML renderers inline stay
free of anything that would reach the network.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llmtracefx.brand import (
    CHART_SEQUENCE,
    HEATMAP_SCALE,
    LOCKUP_SVG,
    MARK_SVG,
    PLOT_ANNOTATION,
    PLOT_LAYOUT,
    TOKENS_CSS,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "brand"

SVG_NS = "{http://www.w3.org/2000/svg}"

EXPECTED_SVGS = (
    "llmtracefx-lockup.svg",
    "llmtracefx-lockup-inverse.svg",
    "llmtracefx-lockup-mono.svg",
    "llmtracefx-mark.svg",
    "llmtracefx-mark-mono.svg",
    "llmtracefx-wordmark.svg",
    "llmtracefx-icon.svg",
)

# The palette these assets are drawn in. Changing one of these is a brand
# decision, so it should have to be made here as well as in DESIGN.md.
INK = "#16181A"
SIGNAL = "#C23D16"
BONE = "#F4F1EA"


@pytest.mark.parametrize("name", EXPECTED_SVGS)
def test_brand_svg_exists_and_is_well_formed(name):
    path = ASSETS / name
    assert path.is_file(), f"missing brand asset: {name}"

    root = ET.parse(path).getroot()

    assert root.tag == f"{SVG_NS}svg"
    assert root.get("viewBox")


@pytest.mark.parametrize("name", EXPECTED_SVGS)
def test_brand_svg_has_an_accessible_name(name):
    root = ET.parse(ASSETS / name).getroot()

    assert root.get("role") == "img"
    titles = root.findall(f"{SVG_NS}title")
    assert titles, f"{name} has no <title>"
    assert titles[0].text == "LLMTraceFX"
    assert root.get("aria-labelledby") == titles[0].get("id")


@pytest.mark.parametrize(
    "name",
    (
        "llmtracefx-lockup-mono.svg",
        "llmtracefx-mark-mono.svg",
        "llmtracefx-wordmark.svg",
    ),
)
def test_monochrome_marks_inherit_currentcolor(name):
    """One-colour reproduction has to be genuinely one colour: no hardcoded
    hex can survive in a mark that is meant to take on its surroundings."""
    source = (ASSETS / name).read_text(encoding="utf-8")

    assert "currentColor" in source
    assert not re.search(r'(?:fill|stroke)="#', source)


def test_primary_lockup_holds_the_palette():
    source = (ASSETS / "llmtracefx-lockup.svg").read_text(encoding="utf-8")

    assert SIGNAL in source
    assert INK in source


def test_inverse_lockup_draws_in_bone_and_keeps_the_signal_pad():
    source = (ASSETS / "llmtracefx-lockup-inverse.svg").read_text(encoding="utf-8")

    assert BONE in source
    assert SIGNAL in source
    assert INK not in source


def test_icon_drops_the_hairline_trace():
    """Below about 24px the secondary trace stops reading as evidence and
    starts reading as noise, so the favicon reduction carries one trace."""
    root = ET.parse(ASSETS / "llmtracefx-icon.svg").getroot()

    paths = root.iter(f"{SVG_NS}path")
    assert len(list(paths)) == 1
    assert len(list(root.iter(f"{SVG_NS}rect"))) == 1


def test_social_preview_is_exactly_the_size_github_asks_for():
    png = ASSETS / "social-preview.png"
    assert png.is_file()

    header = png.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", header[16:24])

    assert (width, height) == (1280, 640)


def test_social_preview_source_is_committed_and_self_contained():
    source = (ASSETS / "social-preview.html").read_text(encoding="utf-8")

    assert "<script" not in source
    assert "@import" not in source
    assert "https://" not in source
    assert "http://" not in source


# --- The constants the HTML renderers inline ------------------------------


@pytest.mark.parametrize("svg", (LOCKUP_SVG, MARK_SVG))
def test_inline_marks_reach_nothing_outside_the_document(svg):
    """These are inlined into offline reports, so an ``xmlns`` (which carries
    an http:// URL) or any external reference would break that guarantee."""
    assert "xmlns" not in svg
    assert "http://" not in svg
    assert "https://" not in svg
    assert "<script" not in svg


@pytest.mark.parametrize("svg", (LOCKUP_SVG, MARK_SVG))
def test_inline_marks_are_labelled_and_parse_as_markup(svg):
    assert 'role="img"' in svg
    assert 'aria-label="LLMTraceFX"' in svg
    ET.fromstring(svg)


def test_inline_pad_falls_back_to_the_literal_signal_colour():
    """The pad takes ``--signal`` from the host document, but has to still be
    orange in a context that never defines it."""
    assert "var(--signal, #c23d16)" in LOCKUP_SVG
    assert "var(--signal, #c23d16)" in MARK_SVG


@pytest.mark.parametrize(
    "token",
    (
        "--field",
        "--sheet",
        "--graticule",
        "--ink",
        "--muted",
        "--rule",
        "--rule-soft",
        "--signal",
        "--signal-tint",
        "--verify",
        "--breach",
        "--hold",
        "--sans",
        "--mono",
    ),
)
def test_token_block_defines_every_documented_token(token):
    assert f"{token}:" in TOKENS_CSS


def test_token_block_fetches_no_fonts():
    assert "@import" not in TOKENS_CSS
    assert "http" not in TOKENS_CSS


# --- Chart theming ---------------------------------------------------------


def _relative_luminance(hex_colour: str) -> float:
    channels = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("colour", CHART_SEQUENCE)
def test_every_series_colour_carries_a_readable_label(colour):
    """Bars are labelled on the inside, so a series colour that cannot hold
    white text is a series colour that loses its own label."""
    assert _contrast(colour, "#ffffff") >= 4.5
    assert _contrast(colour, "#fbfaf7") >= 4.5


def test_series_colours_are_distinct():
    assert len(set(CHART_SEQUENCE)) == len(CHART_SEQUENCE)


def test_plot_layout_puts_figures_on_the_sheet():
    assert PLOT_LAYOUT["paper_bgcolor"] == "#fbfaf7"
    assert PLOT_LAYOUT["plot_bgcolor"] == "#fbfaf7"
    assert PLOT_LAYOUT["colorway"] == list(CHART_SEQUENCE)


def test_plot_layout_reclaims_the_reserved_chart_title_band():
    """Charts are titled by the panel heading in the page, so the band the
    library reserves for a figure title is dead space above every chart."""
    margin = PLOT_LAYOUT["margin"]
    assert isinstance(margin, dict)
    assert margin["t"] <= 60
    # Base margins stay tight so a narrow column spends its width on plot
    # area. Plotly's autoexpand grows them again wherever tick labels or an
    # axis title genuinely need the room.
    assert margin["l"] <= 48
    assert margin["r"] <= 32
    assert margin["b"] <= 48


def test_plot_annotation_demotes_subplot_titles_below_panel_headings():
    """Subplot titles arrive as 16px paper annotations. Left alone they read
    as a second heading competing with the panel heading above the chart."""
    font = PLOT_ANNOTATION["font"]
    assert isinstance(font, dict)
    assert font["size"] <= 12
    assert font["color"] == "#5b6167"
    assert "monospace" in font["family"]


def test_heatmap_scale_spans_zero_to_one_in_order():
    stops = [stop for stop, _ in HEATMAP_SCALE]
    assert stops[0] == 0.0
    assert stops[-1] == 1.0
    assert stops == sorted(stops)
