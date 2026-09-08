# Shared Tkinter helpers for the desktop (Tkinter) build of the pipeline.
#
# The Jupyter build in the repository root draws its interactive figures with
# ipywidgets, which only render inside a notebook. This build keeps exactly the
# same function names and signatures but paints the same matplotlib figures into
# plain Tkinter windows, so the modules can be used from a normal Python script.

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def root():
    """Return a hidden, reusable Tk root so several windows can coexist."""
    instance = getattr(root, "_instance", None)
    if instance is None or not instance.winfo_exists():
        instance = tk.Tk()
        instance.withdraw()
        root._instance = instance
    return instance


class PlotWindow:
    """A window with a controls column on the left and a matplotlib figure on the right."""

    def __init__(self, title, figsize=(10, 5)):
        self.win = tk.Toplevel(root())
        self.win.title(title)

        self.controls = ttk.Frame(self.win, padding=8)
        self.controls.pack(side=tk.LEFT, fill=tk.Y)

        plot_frame = ttk.Frame(self.win)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=figsize)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

    def clear(self):
        """Wipe the figure and hand back a fresh axis."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        return self.ax

    def draw(self):
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def wait(self):
        """Block until the user closes the window, the way a notebook cell blocks."""
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
        root().wait_window(self.win)


def make_listbox(parent, label, options, height=12, multiple=True, default=()):
    """A labelled Listbox. Selection indices line up with the options list."""
    ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
    box = tk.Listbox(
        parent,
        selectmode=tk.EXTENDED if multiple else tk.BROWSE,
        height=height,
        exportselection=False,
    )
    for text in options:
        box.insert(tk.END, text)
    for idx in default:
        if idx < box.size():
            box.selection_set(idx)
    box.pack(anchor="w")
    return box


def selection(box):
    """Currently selected indices of a Listbox, as a plain list."""
    return list(box.curselection())
