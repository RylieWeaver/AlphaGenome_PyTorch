#!/usr/bin/env python3
"""Generate the model-architecture diagrams used by the documentation."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DOCS_DIR / "_static" / "model-architecture"
WIDTH = 1220


DEFS = """<defs>
    <linearGradient id="encoder-gradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#d6e7f1"/>
      <stop offset="1" stop-color="#8ebbd5"/>
    </linearGradient>
    <linearGradient id="decoder-gradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#8ebbd5"/>
      <stop offset="1" stop-color="#d6e7f1"/>
    </linearGradient>
    <filter id="card-shadow" x="-10%" y="-10%" width="120%" height="125%">
      <feDropShadow dx="4" dy="6" stdDeviation="0"
                    flood-color="#dfe6ee" flood-opacity="0.9"/>
    </filter>
    <marker id="arrow-flow" viewBox="0 0 10 10" refX="10" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" class="flow-marker"/>
    </marker>
    <style>
      text {
        fill: #000000;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
          BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .canvas { fill: #f8fafc; }
      .title { font-size: 42px; font-weight: 800; }
      .resolution { font-size: 32px; font-weight: 800; }
      .module-title { font-size: 32px; font-weight: 800; }
      .module-subtitle { font-size: 20px; font-weight: 700; }
      .block-resolution { font-size: 18px; font-weight: 700; }
      .compact-shape { font-size: 18px; }
      .arrow-label { font-size: 18px; font-weight: 800; }
      .input-title { font-size: 27px; font-weight: 800; }
      .input-resolution { font-size: 21px; font-weight: 700; }
      .side-note { font-size: 22px; font-weight: 700; }
      .section-label {
        font-size: 20px; font-weight: 800;
        letter-spacing: 1.4px;
      }
      .card-title { font-size: 23px; font-weight: 800; }
      .card-shape { font-size: 20px; font-weight: 700; }
      .small-note { font-size: 20px; font-weight: 700; }
      .head-title { font-size: 34px; }
      .diagram-title { font-size: 25px; font-weight: 800; }
      .diagram-shape { font-size: 21px; font-weight: 700; }
      .diagram-equation { font-size: 18px; font-weight: 700; }
      .diagram-note { font-size: 19px; font-weight: 700; }
      .encoder-module { fill: url(#encoder-gradient); stroke: #2f6f9e; stroke-width: 2; }
      .decoder-module { fill: url(#decoder-gradient); stroke: #2f6f9e; stroke-width: 2; }
      .flow-line { stroke: #000000; }
      .flow-marker { fill: #000000; }
      .transformer-card { fill: #95bdd7; stroke: #2f6f9e; stroke-width: 2.5; }
      .output-128 { fill: #95bdd7; stroke: #2f6f9e; stroke-width: 2; }
      .output-1 { fill: #c5deed; stroke: #568fb5; stroke-width: 2; }
      .pair-block { fill: #efb49f; stroke: #b94f30; stroke-width: 2.5; }
      .pair-output { fill: #efb49f; stroke: #b94f30; stroke-width: 2; }
      .output-heads { fill: #d9dee6; stroke: #596579; stroke-width: 2.5; }
      .embedding-group { fill: #eceff3; stroke: #7b8491; stroke-width: 1.5; }
      .diagram-group { fill: #eceff3; stroke: #7b8491; stroke-width: 1.5; }
      .runtime-card { fill: #c5deed; stroke: #568fb5; stroke-width: 2; }
      .parameter-card { fill: #ffffff; stroke: #596579; stroke-width: 2; }
      .selection-card { fill: #d9dee6; stroke: #596579; stroke-width: 2; }
      .operation-card { fill: #95bdd7; stroke: #2f6f9e; stroke-width: 2.5; }
      .organism-0 { fill: #b9d9ec; stroke: #2f6f9e; stroke-width: 2; }
      .organism-1 { fill: #f2c0aa; stroke: #b94f30; stroke-width: 2; }
      .organism-2 { fill: #c7dfbd; stroke: #557f48; stroke-width: 2; }
      .compact-card {
        fill: #ffffff; stroke: #cfd9e6; stroke-width: 1.5;
        filter: url(#card-shadow);
      }
      .bank-card { fill: #f8fafc; stroke: #98a2b3; stroke-width: 1.5; }
      .compact-main-title { font-size: 48px; font-weight: 800; }
      .compact-section-title {
        font-size: 24px; font-weight: 800; letter-spacing: 1.3px;
      }
      .compact-caption { font-size: 23px; font-weight: 700; }
      .compact-label { font-size: 28px; font-weight: 800; }
      .compact-legend { font-size: 24px; font-weight: 800; }
      .compact-small { font-size: 23px; font-weight: 700; }
      .neutral-card { fill: #eef1f5; stroke: #667085; stroke-width: 1.8; }
      .pair-cell { fill: #f8d8ca; stroke: #b94f30; stroke-width: 1.8; }
      .pair-cell-selected { fill: #dc7654; stroke: #8f351f; stroke-width: 2.5; }
      .selected-row { fill: #fff4ef; stroke: #b94f30; stroke-width: 2.5; }
      .row-lane { fill: #ffffff; stroke: #98a2b3; stroke-width: 1.6; }
      .row-note { font-size: 21px; font-weight: 700; fill: #000000; }
      .pair-flow { stroke: #000000; }
      .pair-marker { fill: #000000; }
      .architecture-diagram .title { font-size: 58px; }
      .architecture-diagram .module-title { font-size: 42px; }
      .architecture-diagram .module-subtitle { font-size: 27px; }
      .architecture-diagram .block-resolution { font-size: 25px; }
      .architecture-diagram .compact-shape { font-size: 23px; }
      .architecture-diagram .arrow-label { font-size: 23px; }
      .architecture-diagram .input-title { font-size: 37px; }
      .architecture-diagram .input-resolution { font-size: 29px; }
      .architecture-diagram .side-note { font-size: 29px; }
      .architecture-diagram .section-label { font-size: 34px; }
      .architecture-diagram .card-title { font-size: 32px; }
      .architecture-diagram .card-shape { font-size: 27px; }
      .architecture-diagram .head-title { font-size: 44px; }
      .shape-sub { font-size: 19px; baseline-shift: sub; }
    </style>
  </defs>"""


class Svg:
    """Small SVG builder for the architecture figure."""

    def __init__(
        self,
        *,
        height: int,
        title: str,
        description: str,
        width: int = WIDTH,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self.elements: list[str] = []

    @staticmethod
    def number(value: float) -> str:
        return f"{value:g}"

    def raw(self, value: str) -> None:
        self.elements.append(value)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        css_class: str,
        radius: float = 0,
    ) -> None:
        radius_attr = (
            f' rx="{self.number(radius)}" ry="{self.number(radius)}"'
            if radius
            else ""
        )
        self.raw(
            f'<rect x="{self.number(x)}" y="{self.number(y)}" '
            f'width="{self.number(width)}" height="{self.number(height)}"'
            f'{radius_attr} class="{css_class}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        css_class: str,
        anchor: str = "start",
    ) -> None:
        self.raw(
            f'<text x="{self.number(x)}" y="{self.number(y)}" '
            f'class="{css_class}" text-anchor="{anchor}" '
            f'dominant-baseline="middle">{escape(value)}</text>'
        )

    def text_with_subscripts(
        self,
        x: float,
        y: float,
        parts: list[tuple[str, bool]],
        *,
        css_class: str,
        anchor: str = "start",
        subscript_dx: float = 0,
    ) -> None:
        """Draw non-nested, single-level subscripts from ordinary text."""
        subscript_offset = (
            f' dx="{self.number(subscript_dx)}"' if subscript_dx else ""
        )
        content = "".join(
            f'<tspan class="shape-sub"{subscript_offset}>'
            f"{escape(value)}</tspan>"
            if is_subscript
            else escape(value)
            for value, is_subscript in parts
        )
        self.raw(
            f'<text x="{self.number(x)}" y="{self.number(y)}" '
            f'class="{css_class}" text-anchor="{anchor}" '
            f'dominant-baseline="middle">{content}</text>'
        )

    def path(
        self,
        d: str,
        *,
        css_class: str,
        width: float = 2.2,
        marker: str | None = None,
    ) -> None:
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        self.raw(
            f'<path d="{d}" fill="none" class="{css_class}" '
            f'stroke-width="{self.number(width)}" stroke-linecap="round" '
            f'stroke-linejoin="round"{marker_attr}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        css_class: str,
        width: float = 2.2,
        marker: str | None = None,
    ) -> None:
        self.path(
            f"M {self.number(x1)} {self.number(y1)} "
            f"L {self.number(x2)} {self.number(y2)}",
            css_class=css_class,
            width=width,
            marker=marker,
        )

    def polygon(self, points: str, *, css_class: str) -> None:
        self.raw(f'<polygon points="{points}" class="{css_class}"/>')

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        css_class: str,
    ) -> None:
        self.raw(
            f'<circle cx="{self.number(x)}" cy="{self.number(y)}" '
            f'r="{self.number(radius)}" class="{css_class}"/>'
        )

    def render(self) -> str:
        body = "\n  ".join(self.elements)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by docs/_scripts/generate_model_architecture_diagrams.py. -->
<svg xmlns="http://www.w3.org/2000/svg"
     width="{self.width}" height="{self.height}"
     viewBox="0 0 {self.width} {self.height}"
     role="img" aria-labelledby="svg-title svg-description">
  <title id="svg-title">{escape(self.title)}</title>
  <desc id="svg-description">{escape(self.description)}</desc>
  {DEFS}
  {body}
</svg>
"""


def architecture_svg() -> str:
    svg = Svg(
        width=1550,
        height=2125,
        title="AlphaGenome model architecture",
        description=(
            "A rounded convolutional encoder reduces one-base-pair DNA to "
            "128-base-pair tokens with 1,536 channels. A rounded, flat-color "
            "transformer exchanges information with 2048-base-pair Pair "
            "Blocks through horizontal arrows: the outgoing route mean-pools "
            "by sixteen, and the return route repeats each pair axis by "
            "sixteen. A rounded decoder with the opposite color gradient "
            "restores one-base-pair resolution. Seven curved, resolution-matched "
            "skip connections join encoder and decoder stages. Solid black "
            "arrows show every tensor-flow route, including the skips, and "
            "connect the decoder, transformer, and Pair Blocks to three "
            "vertically grouped output embeddings: 1-base-pair, 128-base-pair, "
            "and pair representations. A transformer branch is projected, "
            "repeated along the sequence axis, and merged with the decoder "
            "route before the 1-base-pair embedding. The embedding group then "
            "feeds the prediction heads."
        ),
    )
    svg.rect(0, 0, svg.width, svg.height, css_class="canvas")
    svg.raw('<g class="architecture-diagram">')
    svg.text(
        720,
        45,
        "AlphaGenome Model Architecture",
        css_class="title",
        anchor="middle",
    )

    # Rounded trapezoids keep the original vertical silhouette while avoiding
    # the hard polygon corners of the earlier draft.
    svg.text(
        720,
        105,
        "one-hot DNA Sequence",
        css_class="input-title",
        anchor="middle",
    )
    svg.text(
        720,
        145,
        "1-bp resolution",
        css_class="input-resolution",
        anchor="middle",
    )
    svg.raw('<g transform="translate(0 40)">')
    svg.raw(
        '<path d="M 470 190 Q 458 190 465 204 '
        'L 622 408 Q 630 420 644 420 H 796 '
        'Q 810 420 818 408 L 975 204 Q 982 190 970 190 Z" '
        'class="encoder-module"/>'
    )
    svg.text(720, 260, "Encoder", css_class="module-title", anchor="middle")
    svg.text(
        720,
        310,
        "7× conv + max pool",
        css_class="module-subtitle",
        anchor="middle",
    )
    svg.text(
        720,
        350,
        "1 bp  →  128 bp",
        css_class="module-subtitle",
        anchor="middle",
    )
    svg.text(
        720,
        385,
        "resolution",
        css_class="module-subtitle",
        anchor="middle",
    )

    # A single rounded, flat-color square represents the transformer tower.
    svg.rect(540, 490, 360, 330, css_class="transformer-card", radius=18)
    svg.line(720, 420, 720, 490, css_class="flow-line", width=3.2, marker="arrow-flow")
    svg.text(720, 600, "Transformer", css_class="module-title", anchor="middle")
    svg.text(720, 650, "Blocks", css_class="module-title", anchor="middle")
    svg.text(
        720,
        710,
        "128-bp resolution",
        css_class="block-resolution",
        anchor="middle",
    )

    svg.raw(
        '<path d="M 644 900 Q 630 900 622 912 '
        'L 465 1126 Q 458 1140 470 1140 H 970 '
        'Q 982 1140 975 1126 L 818 912 Q 810 900 796 900 Z" '
        'class="decoder-module"/>'
    )
    svg.text(720, 970, "Decoder", css_class="module-title", anchor="middle")
    svg.text(
        720,
        1020,
        "7× conv + upsample",
        css_class="module-subtitle",
        anchor="middle",
    )
    svg.text(
        720,
        1060,
        "128 bp  →  1 bp",
        css_class="module-subtitle",
        anchor="middle",
    )
    svg.text(
        720,
        1095,
        "resolution",
        css_class="module-subtitle",
        anchor="middle",
    )

    # Rounded orthogonal routes show every resolution-matched U-Net skip.
    for index in range(7):
        source_y = 215 + index * 27
        target_y = 1100 - index * 30
        # These equations are the left sloping edges of the rounded encoder
        # and decoder paths. Endpoints therefore meet the outlines exactly
        # instead of extending inside either filled block.
        source_x = 465 + (source_y - 204) * 157 / 204
        target_x = 622 - (target_y - 912) * 157 / 214
        route_x = 300 + index * 30
        svg.path(
            f"M {source_x} {source_y} H {route_x + 12} "
            f"Q {route_x} {source_y} {route_x} {source_y + 12} "
            f"V {target_y - 12} "
            f"Q {route_x} {target_y} {route_x + 12} {target_y} "
            f"H {target_x}",
            css_class="flow-line",
            width=3.2,
            marker="arrow-flow",
        )
    svg.text(175, 615, "7 skip", css_class="side-note", anchor="middle")
    svg.text(175, 655, "connections", css_class="side-note", anchor="middle")

    # Pair blocks use the same block grammar as the transformer. The two
    # horizontal arrows show sequence-to-pair updates and returned bias.
    svg.rect(1080, 500, 400, 310, css_class="pair-block", radius=18)
    svg.text(1280, 600, "Pair Blocks", css_class="module-title", anchor="middle")
    svg.text(
        1280,
        675,
        "(2048-bp × 2048-bp)",
        css_class="block-resolution",
        anchor="middle",
    )
    svg.text(
        1280,
        715,
        "resolution",
        css_class="block-resolution",
        anchor="middle",
    )
    svg.path(
        "M 900 560 H 1080",
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.text(990, 515, "mean pool", css_class="arrow-label", anchor="middle")
    svg.text(990, 545, "×16", css_class="arrow-label", anchor="middle")
    svg.path(
        "M 1080 750 H 900",
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.text(990, 685, "repeat", css_class="arrow-label", anchor="middle")
    svg.text(990, 715, "16 × 16", css_class="arrow-label", anchor="middle")

    # The transformer output splits between the decoder and the 128-bp output
    # path. Its projected, repeated branch later rejoins the decoder output
    # before the shared route enters the 1-bp output embedder.
    svg.line(720, 820, 720, 860, css_class="flow-line", width=3.2)
    svg.line(720, 860, 720, 900, css_class="flow-line", width=3.2, marker="arrow-flow")
    svg.path(
        "M 720 860 H 1120 Q 1140 860 1140 880 V 1220",
        css_class="flow-line",
        width=3.2,
    )

    # A lightly shaded grouping box keeps the three public embeddings together
    # without reading as another model module.
    svg.rect(320, 1260, 800, 545, css_class="embedding-group", radius=20)
    svg.text(350, 1300, "EMBEDDINGS", css_class="section-label")

    svg.rect(400, 1350, 640, 115, css_class="output-1", radius=16)
    svg.text(720, 1387, "1-bp embedding", css_class="card-title", anchor="middle")
    svg.text_with_subscripts(
        645,
        1430,
        [
            ("[B, S", False),
            ("1", True),
            (", C", False),
            ("1", True),
            ("]", False),
        ],
        css_class="card-shape",
    )

    svg.rect(400, 1495, 640, 115, css_class="output-128", radius=16)
    svg.text(720, 1532, "128-bp embedding", css_class="card-title", anchor="middle")
    svg.text_with_subscripts(
        610,
        1575,
        [
            ("[B, S", False),
            ("128", True),
            (", C", False),
            ("128", True),
            ("]", False),
        ],
        css_class="card-shape",
    )

    svg.rect(400, 1640, 640, 115, css_class="pair-output", radius=16)
    svg.text(
        720,
        1677,
        "(2048-bp × 2048-bp) embedding",
        css_class="card-title",
        anchor="middle",
    )
    svg.text_with_subscripts(
        580,
        1720,
        [
            ("[B, S", False),
            ("pair", True),
            (", S", False),
            ("pair", True),
            (", C", False),
            ("pair", True),
            ("]", False),
        ],
        css_class="card-shape",
        subscript_dx=-3,
    )

    # The repeated branch joins the two vertical routes horizontally. The
    # transformer lane continues independently to the 128-bp card.
    svg.line(
        720,
        1140,
        720,
        1220,
        css_class="flow-line",
        width=3.2,
    )
    svg.line(
        1140,
        1220,
        720,
        1220,
        css_class="flow-line",
        width=3.2,
    )
    svg.line(
        960,
        1220,
        900,
        1220,
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.line(
        720,
        1220,
        720,
        1350,
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.text(
        930,
        1185,
        "repeat ×128",
        css_class="arrow-label",
        anchor="middle",
    )
    svg.path(
        "M 1140 1220 V 1532.5 Q 1140 1552.5 1120 1552.5 H 1040",
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.path(
        "M 1280 810 V 1677.5 Q 1280 1697.5 1260 1697.5 H 1040",
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.raw("</g>")

    # The group-level arrow means that heads select the embeddings they need;
    # it does not imply that every head concatenates all three tensors.
    svg.rect(365, 1935, 710, 120, css_class="output-heads", radius=20)
    svg.path(
        "M 720 1845 V 1935",
        css_class="flow-line",
        width=3.2,
        marker="arrow-flow",
    )
    svg.text(
        720,
        1995,
        "Prediction Heads",
        css_class="module-title head-title",
        anchor="middle",
    )

    # Draw this connector after its target so the full shaft and arrowhead
    # remain visible against the encoder outline.
    svg.line(720, 170, 720, 230, css_class="flow-line", width=3.2, marker="arrow-flow")
    svg.raw("</g>")

    return svg.render()


def multi_organism_linear_svg() -> str:
    svg = Svg(
        width=1400,
        height=760,
        title="Multi-organism linear layer selection",
        description=(
            "An organism index of one selects the orange organism-one linear "
            "layer from the MultiOrganismLinear weight and bias tensors. A "
            "DNA-derived representation with C-star channels then flows "
            "through that selected layer to produce a prediction."
        ),
    )
    svg.rect(0, 0, svg.width, svg.height, css_class="canvas")
    svg.raw('<g transform="translate(0 20)">')

    def add_c_star_text(
        x: float,
        y: float,
        *,
        before: str,
        after: str,
        css_class: str,
        anchor: str = "middle",
    ) -> None:
        """Draw text containing ``C`` with a typographic subscript star."""
        svg.text(
            x,
            y,
            f"{before}C⁎{after}",
            css_class=css_class,
            anchor=anchor,
        )

    def add_slice(
        x: float,
        y: float,
        *,
        css_class: str,
        number: int,
        number_x_offset: float = 28,
        show_shapes: bool = False,
        width: float = 250,
        height: float = 150,
    ) -> None:
        svg.rect(x, y, width, height, css_class=css_class, radius=10)
        svg.text(
            x + number_x_offset,
            y + 40,
            str(number),
            css_class="compact-label",
            anchor="middle",
        )
        if not show_shapes:
            return

        content_x = x + width / 2
        svg.text(
            content_x,
            y + 76,
            "W: [O, C, T]",
            css_class="compact-small",
            anchor="middle",
        )
        svg.text(
            content_x,
            y + 117,
            "b: [O, T]",
            css_class="compact-small",
            anchor="middle",
        )

    def add_selected_layer(
        x: float,
        y: float,
        *,
        css_class: str,
    ) -> None:
        """Draw one selected organism-specific affine layer."""
        width, height = 270, 150
        svg.rect(x, y, width, height, css_class=css_class, radius=10)
        svg.text(
            x + width / 2,
            y + 35,
            "Linear Layer 1",
            css_class="compact-label",
            anchor="middle",
        )
        add_c_star_text(
            x + width / 2,
            y + 82,
            before="W: [",
            after=", T]",
            css_class="compact-small",
        )
        svg.text(
            x + width / 2,
            y + 119,
            "b: [T]",
            css_class="compact-small",
            anchor="middle",
        )

    svg.text(
        700,
        40,
        "Multi-Organism Linear",
        css_class="compact-main-title",
        anchor="middle",
    )
    svg.text(
        260,
        100,
        "LEGEND",
        css_class="compact-section-title",
        anchor="start",
    )
    legend = (
        ("Organism 0", "organism-0"),
        ("Organism 1", "organism-1"),
        ("Organism 2", "organism-2"),
    )
    for column, (label, css_class) in enumerate(legend):
        x = 410 + column * 270
        svg.rect(x, 79, 42, 42, css_class=css_class, radius=6)
        svg.text(x + 56, 100, label, css_class="compact-legend", anchor="start")

    svg.rect(35, 140, 1330, 560, css_class="compact-card", radius=18)

    svg.rect(475, 170, 450, 320, css_class="bank-card", radius=14)
    svg.text(
        700,
        213,
        "MultiOrganismLinear",
        css_class="compact-section-title",
        anchor="middle",
    )
    # The slices follow a consistent back-to-front diagonal order. Exposed
    # corners retain every layer number while organism two sits on top.
    add_slice(525, 255, css_class="organism-0", number=0)
    add_slice(
        575,
        280,
        css_class="organism-1",
        number=1,
    )
    add_slice(
        625,
        305,
        css_class="organism-2",
        number=2,
        show_shapes=True,
    )

    svg.rect(65, 290, 300, 90, css_class="neutral-card", radius=10)
    svg.text(
        165,
        317,
        "Organism",
        css_class="compact-label",
        anchor="middle",
    )
    svg.text(
        165,
        353,
        "Index",
        css_class="compact-label",
        anchor="middle",
    )
    svg.rect(290, 308, 56, 54, css_class="organism-1", radius=8)
    svg.text(318, 335, "1", css_class="compact-label", anchor="middle")
    svg.line(
        365,
        335,
        475,
        335,
        css_class="flow-line",
        width=2.8,
        marker="arrow-flow",
    )
    svg.text(420, 309, "selects", css_class="compact-small", anchor="middle")

    # The selected organism-one layer is inserted into the main data path.
    svg.line(700, 490, 700, 520, css_class="flow-line", width=2.8, marker="arrow-flow")

    svg.rect(65, 520, 320, 150, css_class="neutral-card", radius=12)
    svg.text(225, 570, "DNA Representation", css_class="input-title", anchor="middle")
    add_c_star_text(
        225,
        620,
        before="x  [..., ",
        after="]",
        css_class="compact-small",
    )

    add_selected_layer(
        565,
        520,
        css_class="organism-1",
    )

    svg.rect(1015, 520, 320, 150, css_class="neutral-card", radius=12)
    svg.text(1175, 570, "Prediction", css_class="compact-label", anchor="middle")
    svg.text(1175, 620, "y  [..., T]", css_class="compact-small", anchor="middle")

    svg.line(385, 595, 565, 595, css_class="flow-line", width=2.8, marker="arrow-flow")
    svg.line(835, 595, 1015, 595, css_class="flow-line", width=2.8, marker="arrow-flow")

    svg.raw("</g>")

    return svg.render()


def row_attention_svg() -> str:
    svg = Svg(
        width=1200,
        height=710,
        title="Row attention over pairwise representations",
        description=(
            "A four-by-four grid of pair vectors is separated into four rows. "
            "Arrows within the first row show one pair vector attending across "
            "that row. The same row attention repeats independently for each "
            "row."
        ),
    )
    svg.rect(0, 0, svg.width, svg.height, css_class="canvas")
    svg.raw(
        '<defs><marker id="arrow-pair" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" class="pair-marker"/></marker></defs>'
    )
    svg.text(
        600,
        48,
        "Row Attention",
        css_class="compact-main-title",
        anchor="middle",
    )
    svg.text(300, 98, "LEGEND", css_class="compact-section-title", anchor="end")
    svg.rect(330, 82, 32, 32, css_class="pair-cell", radius=5)
    svg.text(
        382,
        98,
        "Pairwise Representation (C_pair channels)",
        css_class="row-note",
    )

    # A single content card follows the simple visual grammar of the source:
    # one pair grid on the left and the same cells grouped by row on the right.
    svg.raw('<g transform="translate(0 7)">')
    svg.rect(28, 118, 1144, 545, css_class="compact-card", radius=18)
    svg.text(290, 174, "Pairwise Representation", css_class="compact-section-title", anchor="middle")
    svg.text(
        290,
        207,
        "(2048-bp × 2048-bp) Resolution",
        css_class="row-note",
        anchor="middle",
    )
    svg.text(870, 174, "Attention Within Each Row", css_class="compact-section-title", anchor="middle")

    grid_x, grid_y, cell, gap = 120, 238, 72, 16
    for row in range(4):
        for column in range(4):
            svg.rect(
                grid_x + column * (cell + gap),
                grid_y + row * (cell + gap),
                cell,
                cell,
                css_class="pair-cell",
                radius=8,
            )

    svg.text(290, 600, "[S_pair, S_pair, C_pair]", css_class="row-note", anchor="middle")
    svg.text(
        290,
        638,
        "Example: 8,192 bp → 4 × 4 pair grid",
        css_class="row-note",
        anchor="middle",
    )
    svg.line(500, 383, 570, 383, css_class="pair-flow", width=3, marker="arrow-pair")

    # On the right, outlines make row independence visible without axis names
    # or formulas. The first row illustrates attention from one vector to the
    # other vectors in that same row; the operation repeats for every vector.
    lane_x, lane_y = 625, 238
    lane_width, lane_height, lane_gap = 490, 72, 18
    for row in range(4):
        y = lane_y + row * (lane_height + lane_gap)
        svg.rect(
            lane_x,
            y,
            lane_width,
            lane_height,
            css_class="selected-row" if row == 0 else "row-lane",
            radius=10,
        )
        for column in range(4):
            svg.rect(
                lane_x + 18 + column * 118,
                y + 10,
                52,
                52,
                css_class="pair-cell-selected" if row == 0 and column == 3 else "pair-cell",
                radius=7,
            )

    source_x = lane_x + 18 + 3 * 118 + 26
    source_y = lane_y + 10
    for column, lift in zip(range(3), (70, 52, 34), strict=True):
        target_x = lane_x + 18 + column * 118 + 26
        svg.path(
            f"M {source_x} {source_y} Q {(source_x + target_x) / 2} "
            f"{lane_y - lift} {target_x} {lane_y + 10}",
            css_class="pair-flow",
            width=2.5,
            marker="arrow-pair",
        )

    svg.text(
        870,
        638,
        "Repeated independently for each row",
        css_class="row-note",
        anchor="middle",
    )

    svg.raw("</g>")

    return svg.render()


def generated_files() -> dict[Path, str]:
    return {
        OUTPUT_DIR / "architecture.svg": architecture_svg(),
        OUTPUT_DIR / "multi-organism-linear.svg": multi_organism_linear_svg(),
        OUTPUT_DIR / "row-attention.svg": row_attention_svg(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the checked-in SVG is stale.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = generated_files()

    if args.check:
        stale = [
            path
            for path, contents in outputs.items()
            if not path.exists() or path.read_text() != contents
        ]
        if stale:
            for path in stale:
                print(f"stale generated file: {path}")
            return 1
        for path in outputs:
            print(f"up to date: {path}")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, contents in outputs.items():
        path.write_text(contents)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
