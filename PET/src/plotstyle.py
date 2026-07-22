"""
The one place every figure in the study gets its look from.

Before this module each figure carried its own fonts, palette, marker shapes and hand-placed
text, and they drifted: five marker shapes across four figures, three different phrasings of
the same axis, boxed annotations competing with the marks. The rules encoded here (see
VIZ_REDESIGN_TASK.md) are deliberately narrow:

  * COLOUR IS THE ONLY FAMILY ENCODING. One marker shape — a circle — everywhere. Facts
    that used to ride on marker shape (inverted attack, operating point) belong in a
    caption or a config label, because a reader should never have to learn a glyph legend.
  * NO SENTENCES ON THE GRAPH. A figure carries a short NOUN-PHRASE title, axis labels,
    and short point/series labels — nothing else. Takeaway sentences, subtitle sentences
    and bottom caption paragraphs live in Figure_Explainer.docx, which is where a reader
    who wants the argument is already looking. `style_figure` still accepts a subtitle and
    a caption because two report-tier figures need a single short clause (see
    CAPTION_CLAUSE_MAX), but a figure that passes prose is a bug, not a style choice.
  * REFERENCE LINES CARRY ONE-WORD END TAGS. A stripped figure has no subtitle left to
    explain what the rule at 0.5 is, so the line labels itself: "chance", "baseline". One
    lowercase word, at the line's end, on every figure that draws one.
  * LINES ONLY WHERE y IS A FUNCTION OF x. Nothing here enforces that — it is a property of
    what you pass to `ax.plot` — but it is the rule the sweep figures exist to satisfy, and
    the reason the tradeoff figures are scatter-only.
  * DIRECT-LABEL WHEN ENDS SEPARATE, LEGEND WHEN THEY CONVERGE. Direct labels beat a
    lookup table right up until two series end in the same place, at which point they
    collide and a small legend is the honest choice. See `small_legend`.

The palette is not eyeballed. The five family hues were checked with the dataviz palette
validator (lightness band, chroma floor, CVD separation, normal-vision floor, contrast vs
surface) and all pairs pass; the closest pair is DP-SGD green vs Output orange at ΔE 6.8
under protanopia, which sits in the band that is legal only with a secondary encoding — so
every figure that draws both ALSO direct-labels or facets them. Keep that property if you
retune these.
"""
import textwrap

import matplotlib as mpl

# Neutrals. INK is for anything a reader must actually read; MUTED is for the supporting
# text (subtitles, captions, tick labels) that should recede until looked at directly.
INK = "#1A1A1A"
MUTED = "#6B7280"
GRID = "#D8D8D6"
AXIS = "#9CA3AF"
SURFACE = "#FFFFFF"

# The undefended baseline is a REFERENCE, not one of the families, so it is deliberately
# achromatic — it never competes with a family hue for attention and it is exempt from the
# categorical palette checks (a grey fails the chroma floor by design).
BASELINE_COLOR = "#3F3F46"

DPSGD_METHOD = "DP-SGD (fixed-clip baseline)"

# The validated categorical palette. Hues are the study's originals, retuned for contrast
# against white; re-run scripts/validate_palette.js if you change any of them.
FAMILY_COLOR = {
    "Input":               "#3B75AF",
    "Output":              "#D2760C",
    "Output (last-layer)": "#9E4A2F",
    DPSGD_METHOD:          "#3B9C5A",
    "Personalized":        "#C7318F",
    "Baseline":            BASELINE_COLOR,
}

# Rule 2, as a constant: there is one marker shape in this study and this is it.
MARKER = "o"

# Text sizes. The title is a short noun phrase naming the figure, and it is the only large
# text on the canvas.
TITLE_SIZE = 13.5
SUBTITLE_SIZE = 9.5
CAPTION_SIZE = 7.5

# The one-word tags every reference line wears. Constants rather than string literals at
# each call site because the whole value of the rule is that all ~12 figures say the SAME
# word — which is exactly what drifted last time (three wordings of "chance" across four
# figures).
CHANCE_LABEL = "chance"
BASELINE_LABEL = "baseline"

# A caption may survive the prose strip only as a single short clause naming an EXCLUSION
# or a units caveat — "Personalized excluded — single-client simulation, no formal ε". Two
# report-tier figures need one. Anything longer is the paragraph that moved to the doc, so
# `style_figure` refuses it rather than quietly re-growing the caption block.
CAPTION_CLAUSE_MAX = 120

# Characters per inch of figure width, measured per size class rather than guessed: bold
# title text runs ~9 chars/inch at 13.5pt, regular subtitle ~13.5 at 9.5pt, caption ~16 at
# 7.5pt. Used to wrap text to the canvas so nothing overflows when the PDF copy is re-saved
# at report width (6.5in) with the same point sizes.
TITLE_CPI = 9.0
SUBTITLE_CPI = 13.5
CAPTION_CPI = 16.0

# Left margin for the title block, in figure fractions. Matches the default tight_layout
# gutter closely enough that the title reads as aligned with the y-axis label.
TEXT_X = 0.012


def family_color(method, default=MUTED):
    """Colour for a family, by its results-table `Method` name."""
    return FAMILY_COLOR.get(method, default)


def apply_style():
    """
    Install the study's rcParams. Idempotent; call once before creating any figure.

    Everything here is subtractive — dropping the top/right spines, dropping the x-grid,
    fading the remaining grid to a hint — because the marks are the content and the frame
    is not. The y-grid survives alone since every figure in the study is read by comparing
    heights or levels, never horizontal positions against a rule.
    """
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": None,      # captions are placed in figure coords; never crop them
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",     # y only — see docstring
        "axes.axisbelow": True,    # grid behind the marks, never through them
        "axes.labelcolor": INK,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.titlecolor": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.25,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK,
        "font.size": 10,
        "text.color": INK,
    })


def _wrap(text, width_in, cpi):
    """Collapse whitespace and wrap to the figure's actual width."""
    return textwrap.fill(" ".join(str(text).split()), width=max(28, int(width_in * cpi)))


def _block_height(fontsize, n_lines, height_in):
    """Height of an n-line text block, in figure fractions (1.45 = line spacing)."""
    return n_lines * fontsize * 1.45 / (height_in * 72.0)


# Hard ceilings on how much of the canvas the text blocks may claim, as figure fractions.
# When a block would exceed its ceiling the FONT shrinks until it fits, which is the one
# lever that costs no information.
#
# These must leave room for the axes AND their decorations — axis labels and tick labels sit
# inside the tight_layout rect, so text blocks claiming 60% of a short canvas do not merely
# shrink the plot, they leave tight_layout unable to place the y-label at all and it lands on
# the subtitle. Together these cap the text at half the canvas, which is what keeps the
# report copies (re-exported at 6.5in, where captions wrap to more lines) legible.
MAX_TOP_FRAC = 0.24
MAX_CAPTION_FRAC = 0.26
MIN_FONT_SIZE = 5.5


def _fit_block(text, base_size, base_cpi, width_in, height_in, max_frac):
    """
    Largest font size at or below `base_size` whose wrapped block fits inside `max_frac`.

    Returns (size, wrapped_text, n_lines). Shrinking the font makes the text narrower AND
    shorter at once — more characters per line, and each line less tall — so this converges
    in a few steps; MIN_FONT_SIZE stops it going past legibility, at which point the block
    is allowed to be slightly too tall rather than becoming unreadable.

    chars-per-inch is quoted at each class's design size, so it scales inversely with the
    font: half the point size fits twice the characters across the same inch.
    """
    size = base_size
    while True:
        wrapped = _wrap(text, width_in, base_cpi * base_size / size)
        n = wrapped.count("\n") + 1
        if _block_height(size, n, height_in) <= max_frac or size <= MIN_FONT_SIZE:
            return size, wrapped, n
        size -= 0.5


def style_figure(fig, title=None, subtitle=None, caption=None, top_gap=0.022):
    """
    Lay out a figure's title / subtitle / caption and reserve exactly the space they need.

    Every figure gets its headline the same way: a bold left-aligned NOUN PHRASE naming the
    figure. Left-aligned rather than centred because a title is read from its start, and a
    centred one moves its start every time the text changes length.

    `subtitle` and `caption` survive for the two report-tier figures that need one short
    clause (an exclusion note, a units caveat). They are NOT for prose — a caption longer
    than CAPTION_CLAUSE_MAX raises, because that is the paragraph that belongs in
    Figure_Explainer.docx and a silent re-grow is how the last strip got undone.

    IDEMPOTENT, and that is the whole point: `save_figure` re-saves each figure at report
    width (6.5in) with the same point sizes, so the same text needs more lines on a narrower
    canvas. Calling this again REWRAPS and repositions the existing artists — and recomputes
    the tight_layout rect from the new line counts — instead of stacking a second copy on
    top. Pass it straight to `save_figure(..., relayout=...)`.
    """
    for name, text in (("subtitle", subtitle), ("caption", caption)):
        if text and len(" ".join(str(text).split())) > CAPTION_CLAUSE_MAX:
            raise ValueError(
                f"{name} is {len(str(text))} chars; figures carry a short clause at most "
                f"({CAPTION_CLAUSE_MAX}). Move the prose to Figure_Explainer.docx.")

    width_in, height_in = fig.get_size_inches()
    store = fig.__dict__.setdefault("_plotstyle_text", {})

    def place(key, wrapped, size, y, **kw):
        art = store.get(key)
        if art is None:
            store[key] = fig.text(TEXT_X, y, wrapped, ha="left", va="bottom",
                                  fontsize=size, **kw)
        else:
            art.set_text(wrapped)
            art.set_position((TEXT_X, y))
            art.set_fontsize(size)

    # --- top block: title then subtitle, growing downward from the canvas top ---
    # Title and subtitle share one ceiling: on a short figure it is the pair that has to
    # give, and shrinking only one of them inverts their hierarchy.
    head_frac = MAX_TOP_FRAC / 2 if (title and subtitle) else MAX_TOP_FRAC
    cursor = 1.0 - 0.014
    if title:
        size, wrapped, n = _fit_block(title, TITLE_SIZE, TITLE_CPI,
                                      width_in, height_in, head_frac)
        cursor -= _block_height(size, n, height_in)
        place("title", wrapped, size, cursor, fontweight="bold", color=INK)
    if subtitle:
        size, wrapped, n = _fit_block(subtitle, SUBTITLE_SIZE, SUBTITLE_CPI,
                                      width_in, height_in, head_frac)
        cursor -= _block_height(size, n, height_in) + 0.004
        place("subtitle", wrapped, size, cursor, color=MUTED)
    top_rect = min(1.0, cursor - top_gap) if (title or subtitle) else 1.0

    # --- foot: the caption, with the axes pushed up by however many lines it wraps to ---
    bottom_rect = 0.0
    if caption:
        size, wrapped, n = _fit_block(caption, CAPTION_SIZE, CAPTION_CPI,
                                      width_in, height_in, MAX_CAPTION_FRAC)
        # The y here is the block's BASELINE, so the last line's descenders hang below it:
        # anchoring at ~0 clips the tails of every 'g', 'p' and 'y' in the final line.
        caption_y = 0.5 * CAPTION_SIZE / (height_in * 72.0)
        place("caption", wrapped, size, caption_y, color=MUTED, style="italic")
        bottom_rect = caption_y + _block_height(size, n, height_in) + 0.012

    fig.tight_layout(rect=(0, bottom_rect, 1, top_rect))
    return fig


def scale_text_sizes(texts, base_sizes, width_in, design_width_in):
    """
    Shrink a specific list of text artists in step with the canvas; returns the factor used.

    `base_sizes` are the sizes the figure was designed with, captured before any scaling so
    repeated calls (the export path relayouts twice) never compound. Never enlarges past the
    design size, however wide the canvas gets.
    """
    k = min(1.0, width_in / design_width_in)
    for text, size in zip(texts, base_sizes):
        text.set_fontsize(size * k)
    return k


def _managed(artist):
    """True if a caller already owns this artist's size (see `scale_text_sizes` users)."""
    return getattr(artist, "_plotstyle_managed", False)


def scale_axes_text(fig, design_width_in, skip=()):
    """
    Shrink EVERY piece of text inside the axes in step with the canvas.

    Titles and captions are wrapped to the figure width (`_fit_block`), but text inside the
    axes cannot wrap — it is pinned to a data position or to an axis. Point sizes do not
    shrink with the canvas, so the report copy at 6.5in hands every axis label, tick label,
    legend entry and point annotation ~1.6x its designed share of the figure. That is what
    made the two-panel figures collide on export: adjacent panels' x-labels ran into each
    other, and the y-label grew until tight_layout could not place it below the subtitle.

    Base sizes are recorded on each artist the first time it is seen, so this is idempotent
    across the resize/restore that `save_figure` performs. Tick labels are driven through
    tick_params rather than per-artist, because matplotlib regenerates tick Text objects on
    every draw and any attribute stored on them is lost.

    Artists in `skip` — or flagged by a caller that scales them itself, e.g. direct labels
    whose OFFSETS must scale too — are left alone.
    """
    k = min(1.0, fig.get_size_inches()[0] / design_width_in)
    skip = set(map(id, skip))

    for ax in fig.axes:
        base_ticks = getattr(ax, "_plotstyle_base_tick_size", None)
        if base_ticks is None:
            base_ticks = mpl.rcParams["xtick.labelsize"]
            ax._plotstyle_base_tick_size = base_ticks
        # which="both": on a log axis matplotlib labels MINOR ticks too (the "6x10^-2"
        # between decades), and those keep the rcParams size unless asked otherwise —
        # leaving one tick label conspicuously larger than every other on the figure.
        ax.tick_params(axis="both", which="both", labelsize=base_ticks * k)

        artists = [ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            artists += list(legend.get_texts())
        for art in artists:
            if art is None or id(art) in skip or _managed(art):
                continue
            base = getattr(art, "_plotstyle_base_size", None)
            if base is None:
                base = art.get_fontsize()
                art._plotstyle_base_size = base
            art.set_fontsize(base * k)
    return k


def small_legend(ax, handles=None, loc="best", **kw):
    """
    The fallback for when direct labels would collide — a quiet, frameless legend.

    The study prefers direct labels: a legend is a lookup table for something the end of a
    curve already says. But two figures have series that CONVERGE — utility_retention's
    curves meet at both the top-right and the bottom-right, and the ε chart's Output and
    Output-LL rows end within a decade of each other — so their end labels overprint no
    matter how they are nudged. There, a legend is the honest choice.

    Deliberately understated: no frame, muted text, tight padding. It is a key, not a
    panel, and it must not read as heavier than the curves it names.
    """
    legend = ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8.5,
                       labelspacing=0.4, handlelength=1.6, borderaxespad=0.4, **kw)
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


def chance_line(ax, x=0.5, label=CHANCE_LABEL, axis="x"):
    """
    The 0.5 rule on a leakage axis — "an attacker that learns nothing".

    Every leakage axis in the study needs it and every one of them used to draw it slightly
    differently (three colours, two alphas, two label wordings across four figures).

    The tag DEFAULTS on: with the subtitles stripped, an unlabelled rule at 0.5 is a line a
    reader cannot resolve without the doc. Pass `label=None` only where the same tag is
    already printed by a neighbouring panel (see fig_sweeps, which tags column 0 only).
    """
    draw = ax.axvline if axis == "x" else ax.axhline
    draw(x, color=AXIS, linewidth=1.0, alpha=0.9, zorder=1)
    if label:
        if axis == "x":
            ax.annotate(label, (x, 0.012), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="left", va="bottom")
        else:
            ax.annotate(label, (0.012, x), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="left", va="bottom")


def baseline_line(ax, y, label=BASELINE_LABEL, axis="y"):
    """
    Dashed undefended-baseline reference. Grey, thin, and always behind the marks.

    Tagged by default for the same reason `chance_line` is — see its docstring for when to
    pass `label=None`.
    """
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(y, color=BASELINE_COLOR, linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    if label:
        if axis == "y":
            ax.annotate(label, (0.012, y), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="left", va="bottom")
        else:
            ax.annotate(label, (y, 0.012), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="left", va="bottom")
