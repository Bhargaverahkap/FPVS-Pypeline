# Colour palette and ttk styling for the FPVS desktop app.
#
# The palette is taken from the project logo: red, black and a ramp of greys on
# white. Red is reserved for actions and warnings, never for large fills.
# Change a value here and the whole app follows.

import matplotlib

PALETTE = {
    "red":       "#E8232A",   # accent: run buttons, active menu, bad channels
    "red_dark":  "#8A1216",   # pressed / hover state of a red control
    "black":     "#111111",   # text, hairline borders
    "dark":      "#4D4D4D",   # panel headers, secondary text
    "mid":       "#9A9A9A",   # disabled text, grid lines
    "light":     "#D9D9D9",   # panel fill
    "pale":      "#F2F2F2",   # window background
    "white":     "#FFFFFF",   # plot canvas, entry fields
}

# Plotted traces keep matplotlib's own default colour cycle, so a channel looks
# the same here as it does in the notebooks. The palette above dresses the
# window chrome and the axes, not the data.

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")


def apply(root):
    """Apply the palette to ttk widgets and to matplotlib figures."""
    from tkinter import ttk

    style = ttk.Style(root)
    try:
        style.theme_use("clam")   # the only stock theme that honours colours everywhere
    except Exception:
        pass

    p = PALETTE
    root.configure(background=p["pale"])

    style.configure(".", background=p["pale"], foreground=p["black"], font=FONT)
    style.configure("TFrame", background=p["pale"])
    style.configure("Panel.TFrame", background=p["light"])
    style.configure("TLabel", background=p["pale"], foreground=p["black"])
    style.configure("Panel.TLabel", background=p["light"], foreground=p["black"])
    style.configure("Header.TLabel", background=p["pale"], foreground=p["dark"], font=FONT_BOLD)
    style.configure("Title.TLabel", background=p["pale"], foreground=p["black"], font=FONT_TITLE)
    style.configure("Hint.TLabel", background=p["pale"], foreground=p["dark"])

    style.configure("TButton", background=p["light"], foreground=p["black"],
                    bordercolor=p["black"], focuscolor=p["mid"], padding=(10, 5))
    style.map("TButton",
              background=[("active", p["mid"]), ("disabled", p["light"])],
              foreground=[("disabled", p["mid"])])

    style.configure("Accent.TButton", background=p["red"], foreground=p["white"],
                    bordercolor=p["black"], padding=(12, 6), font=FONT_BOLD)
    style.map("Accent.TButton",
              background=[("active", p["red_dark"]), ("disabled", p["mid"])],
              foreground=[("disabled", p["light"])])

    style.configure("TMenubutton", background=p["dark"], foreground=p["white"],
                    padding=(10, 6), font=FONT_BOLD, arrowcolor=p["white"])
    style.map("TMenubutton", background=[("active", p["red"])])

    style.configure("TEntry", fieldbackground=p["white"], foreground=p["black"],
                    bordercolor=p["mid"], insertcolor=p["black"])
    style.configure("TCombobox", fieldbackground=p["white"], background=p["light"])
    style.configure("TCheckbutton", background=p["pale"], foreground=p["black"])
    style.configure("TRadiobutton", background=p["pale"], foreground=p["black"])
    style.configure("TSeparator", background=p["black"])
    style.configure("Horizontal.TProgressbar", background=p["red"],
                    troughcolor=p["light"], bordercolor=p["mid"])

    matplotlib.rcParams.update({
        "figure.facecolor": p["white"],
        "axes.facecolor": p["white"],
        "axes.edgecolor": p["black"],
        "axes.labelcolor": p["black"],
        "axes.titlecolor": p["black"],
        "text.color": p["black"],
        "xtick.color": p["dark"],
        "ytick.color": p["dark"],
        "grid.color": p["mid"],
        "legend.facecolor": p["white"],
        "legend.edgecolor": p["mid"],
        "font.size": 9,
    })


def listbox_options():
    """Keyword arguments that put a plain tk.Listbox in the palette."""
    p = PALETTE
    return dict(
        background=p["white"],
        foreground=p["black"],
        selectbackground=p["red"],
        selectforeground=p["white"],
        highlightthickness=1,
        highlightbackground=p["mid"],
        relief="flat",
        font=FONT,
    )


def text_options():
    """Keyword arguments for the log pane."""
    p = PALETTE
    return dict(
        background=p["white"],
        foreground=p["black"],
        insertbackground=p["black"],
        highlightthickness=1,
        highlightbackground=p["mid"],
        relief="flat",
        font=("Consolas", 9),
    )
