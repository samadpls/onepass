"""Write assets/hero.svg, the animated README banner.

The text is drawn as outlines because GitHub shows README images through a proxy that loads no web
fonts. Fonts (from github.com/google/fonts) go in data/fonts: Bricolage.ttf and Geist.ttf.

    uv run --with uharfbuzz --with fonttools python assets/make_hero.py
"""

from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONTS = Path("data/fonts")
INK, MUTED, BLUE, YELLOW, YELLOW_TEXT, RULE = (
    "#0F1452",
    "#5B6190",
    "#1D2CF3",
    "#FFC61A",
    "#9A6000",
    "#B9BEE6",
)
W, H, LOOP = 1280, 400, 10
TILE = 184


def num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


class Font:
    def __init__(self, file, **axes):
        face = hb.Face(hb.Blob.from_file_path(FONTS / file))
        self.font, self.unit = hb.Font(face), 1 / face.upem
        self.font.set_variations(axes)

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {})
        return zip(buf.glyph_infos, buf.glyph_positions)

    def width(self, text, size, tracking=0):
        return sum(p.x_advance * size * self.unit + tracking for _, p in self._shape(text))

    def path(self, text, x, y, size, tracking=0):
        s = size * self.unit
        pen = SVGPathPen(None, ntos=num)
        for info, p in self._shape(text):
            self.font.draw_glyph_with_pen(
                info.codepoint, TransformPen(pen, (s, 0, 0, -s, x + p.x_offset * s, y - p.y_offset * s))
            )
            x += p.x_advance * s + tracking
        return pen.getCommands()


def fade(name, windows, base):
    """Opacity 1 inside each (start, end) percent window, with 1% fades."""
    stops = {0.0: 0.0, 100.0: 0.0}
    for a, b in windows:
        stops |= {max(a - 1, 0): 0.0, a: 1.0, b - 1: 1.0, b: 0.0}
    if windows[0][0] == 0:
        stops[0.0] = 1.0
    body = "".join(f"{num(t)}%{{opacity:{o:g}}}" for t, o in sorted(stops.items()))
    return f"@keyframes {name}{{{body}}}.{name}{{opacity:{base};animation:{name} {LOOP}s linear infinite}}"


def main():
    title = Font("Bricolage.ttf", wght=800, opsz=96)
    text = Font("Geist.ttf", wght=400)
    bold = Font("Geist.ttf", wght=600)

    x0 = 432
    parts = [
        f'<path fill="{INK}" d="{title.path("onepass", x0, 158, 112, -3)}"/>',
        f'<path fill="{MUTED}" d="{text.path("One pass. Every answer.", x0, 216, 30)}"/>',
    ]
    status = [
        ("read", MUTED, "Reading the text and the questions", [(0, 31)], 0),
        ("score", BLUE, "Scoring every option in one pass", [(32, 62)], 0),
        ("answer", YELLOW_TEXT, "A probability for every answer", [(63, 98)], 1),
    ]
    css = []
    for name, color, words, windows, base in status:
        parts.append(
            f'<g class="{name}"><circle cx="{x0 + 8}" cy="271" r="7" fill="{color}"/>'
            f'<path fill="{color}" d="{bold.path(words, x0 + 26, 281, 28)}"/></g>'
        )
        css.append(fade(name, windows, base))
    assert max(x0 + 26 + bold.width(words, 28) for _, _, words, _, _ in status) < W - 60

    links = ["github.com/samadpls", "samadpls.com", "Open source, Apache-2.0"]
    x = x0
    for k, link in enumerate(links):
        if k:
            parts.append(f'<circle cx="{num(x + 14)}" cy="327" r="2.5" fill="{RULE}"/>')
            x += 28
        parts.append(f'<path fill="{BLUE}" d="{bold.path(link, x, 334, 21)}"/>')
        x += bold.width(link, 21)
    assert x < W - 60 and x0 + title.width("onepass", 112, -3) < W - 60

    # The ring draws while reading, the arrow passes through it, then the arrowhead lands.
    css += [
        "@keyframes ring{0%,1%{stroke-dashoffset:943;opacity:1}27%{stroke-dashoffset:0}97%{opacity:1}100%{stroke-dashoffset:0;opacity:0}}"
        f".ring{{stroke-dasharray:943 943;animation:ring {LOOP}s ease-in-out infinite}}",
        "@keyframes shaft{0%,32%{stroke-dashoffset:332}40%,96%{stroke-dashoffset:0}100%{stroke-dashoffset:-332}}"
        f".shaft{{stroke-dasharray:330 400;animation:shaft {LOOP}s ease-in-out infinite}}",
        "@keyframes head{0%,63%{transform:scale(0)}67%{transform:scale(1.3)}70%,97%{transform:scale(1)}100%{transform:scale(0)}}"
        f".head{{transform-box:fill-box;transform-origin:center;animation:head {LOOP}s ease-out infinite}}",
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}",
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="onepass: one pass, every answer">
<title>onepass: one pass, every answer</title>
<defs><radialGradient id="glow" cx="235" cy="200" r="430" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#E1E5FD"/><stop offset="1" stop-color="#E1E5FD" stop-opacity="0"/></radialGradient></defs>
<style>{"".join(css)}</style>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="24" fill="#F7F8FE" stroke="#E3E6FA"/>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="24" fill="url(#glow)"/>
<g transform="translate({235 - TILE / 2:.2f} {200 - TILE / 2:.2f}) scale({TILE / 512:.4f})">
<rect width="512" height="512" rx="112" fill="{BLUE}"/>
<circle class="ring" cx="256" cy="256" r="150" fill="none" stroke="#FFFFFF" stroke-width="44" transform="rotate(-90 256 256)"/>
<path class="shaft" d="M64 256H392" stroke="{YELLOW}" stroke-width="44" stroke-linecap="round"/>
<path class="head" d="M356 196L436 256L356 316" fill="none" stroke="{YELLOW}" stroke-width="44" stroke-linecap="round" stroke-linejoin="round"/>
</g>
{"".join(parts)}
</svg>
"""
    out = Path("assets/hero.svg")
    out.write_text(svg)
    print(out, f"{len(svg) // 1024} KB")


if __name__ == "__main__":
    main()
