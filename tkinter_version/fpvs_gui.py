# FPVS desktop application.
#
# Run it with:   python fpvs_gui.py
#
# The window has four parts:
#   * a top bar with Load data and the path of the file being worked on,
#   * four dropdown menus on the left, built entirely from gui_registry.py,
#   * the summary plot in the centre, always on screen, blank until data loads,
#   * a log pane at the bottom carrying everything the pipeline prints.
#
# Nothing in this file names a pipeline function. To move, add or remove a
# button, edit gui_registry.py - see GUI_README.md.

import ast
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import theme
import gui_registry as registry
from summary_view import SummaryView

import FPyVS_appylication as app
import cust_funcs as cf

MODULES = {"app": app, "cf": cf}

DATA_SUFFIXES = {".npy", ".pkl"}
BDF_SUFFIXES = {".bdf"}
MAT_SUFFIXES = {".mat", ".lw6"}


# ---------------------------------------------------------------------------
# turning typed text into arguments
# ---------------------------------------------------------------------------

def _number(text):
    """int when the text has no decimal point, float otherwise."""
    text = text.strip()
    value = float(text)
    return int(value) if value.is_integer() and "." not in text and "e" not in text.lower() else value


def parse_value(kind, text):
    """Convert one field. An empty field means "leave the default alone"."""
    text = (text or "").strip()
    if kind == "bool":
        return bool(text)
    if text == "":
        return None

    if kind == "int":
        return int(float(text))
    if kind == "float":
        return float(text)
    if kind in ("str", "path", "dirpath"):
        return text
    if kind == "numbers":
        return [_number(part) for part in text.split(",") if part.strip()]
    if kind == "ints":
        return [int(float(part)) for part in text.split(",") if part.strip()]
    if kind == "strs":
        return [part.strip() for part in text.split(",") if part.strip()]
    if kind == "json":
        return ast.literal_eval(text)
    raise ValueError(f"unknown parameter kind {kind!r}")


def resolve(dotted):
    """"app.bandpassfilter" -> the function object."""
    module_name, _, func_name = dotted.partition(".")
    try:
        module = MODULES[module_name]
    except KeyError:
        raise KeyError(f"{dotted!r}: module prefix must be one of {sorted(MODULES)}")
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise AttributeError(f"{dotted!r}: {module_name} has no function {func_name!r}")


# ---------------------------------------------------------------------------
# argument builders - one per Step.builder value
# ---------------------------------------------------------------------------

def _require_file(app_window):
    if app_window.filepath is None:
        raise RuntimeError("Load a data file first.")
    return app_window.filepath


def _require_data(app_window):
    if app_window.npy_data is None:
        raise RuntimeError("Load a data file first.")
    return app_window.npy_data, app_window.meta_data


def _activations(app_window, values):
    """Reduce the loaded data to one value per channel for the topomaps."""
    npy_data, _ = _require_data(app_window)
    sample = values.get("sample")
    data = np.asarray(npy_data)
    picked = data[int(sample)] if sample is not None else np.mean(data, axis=0)
    while np.ndim(picked) > 1:
        picked = np.mean(picked, axis=-1)
    return np.asarray(picked)


def build_file_step(app_window, step, values):
    return resolve(step.func), (_require_file(app_window), app_window.stepno), dict(values)


def build_file_only(app_window, step, values):
    return resolve(step.func), (_require_file(app_window),), dict(values)


def build_folder_event(app_window, step, values):
    folder = values.get("folderpath")
    label = values.get("event_label")
    if not folder or not label:
        raise RuntimeError("Both a folder and an event label are needed.")
    return resolve(step.func), (Path(folder), label), {}


def build_ica_fit(app_window, step, values):
    return resolve(step.func), (_require_file(app_window),), dict(values)


def build_ica_apply(app_window, step, values):
    state = app_window.ica_state
    if state is None:
        raise RuntimeError("Fit the ICA first, from the Preprocessing menu.")
    rmidx = values.get("rmidx") or []
    return resolve(step.func), (_require_file(app_window), state["ica"], state["raw"], rmidx), {}


def build_data_only(app_window, step, values):
    npy_data, meta_data = _require_data(app_window)
    return resolve(step.func), (npy_data, meta_data), {}


def build_ica_overlay(app_window, step, values):
    npy_data, meta_data = _require_data(app_window)
    state = app_window.ica_state
    if state is None:
        raise RuntimeError("Fit the ICA first, from the Preprocessing menu.")
    labels = np.array(meta_data["chanlocs"]["labels"], dtype=object)
    subjid = app_window.filepath.stem.split()[-1] if app_window.filepath else None
    return (resolve(step.func),
            (npy_data, state["ica_data"]),
            {"labels": labels, "chid": values.get("chid"), "subjid": subjid})


def build_spectrogram(app_window, step, values):
    npy_data, meta_data = _require_data(app_window)
    kwargs = {"fs": meta_data.get("fs", 256)}
    for name in ("titlestr", "freqlim", "binsize", "isoverlap"):
        kwargs[name] = values.get(name)
    return resolve(step.func), (npy_data, values.get("chid") or 0), kwargs


def build_mw_spectrogram(app_window, step, values):
    npy_data, meta_data = _require_data(app_window)
    kwargs = {"fs": meta_data.get("fs", 256)}
    for name in ("titlestr", "freqlim", "binsize", "isoverlap"):
        kwargs[name] = values.get(name)
    return (resolve(step.func),
            (npy_data, values.get("chid") or 0, values.get("epochid") or 0),
            kwargs)


def build_topomap(app_window, step, values):
    _, meta_data = _require_data(app_window)
    activations = _activations(app_window, values)
    title = values.get("title")
    func = resolve(step.func)
    # showme2DTopomap names its title argument differently from the 3D ones.
    keyword = "titlestr" if func.__name__ == "showme2DTopomap" else "title"
    kwargs = {keyword: title} if title else {}
    return func, (activations, meta_data), kwargs


BUILDERS = {
    "file_step": build_file_step,
    "file_only": build_file_only,
    "folder_event": build_folder_event,
    "ica_fit": build_ica_fit,
    "ica_apply": build_ica_apply,
    "data_only": build_data_only,
    "ica_overlay": build_ica_overlay,
    "spectrogram": build_spectrogram,
    "mw_spectrogram": build_mw_spectrogram,
    "topomap": build_topomap,
}


# ---------------------------------------------------------------------------
# the parameter screen
# ---------------------------------------------------------------------------

class ParamScreen(tk.Toplevel):
    """One screen per step: a field per parameter, then Run."""

    def __init__(self, parent, step, stepno, on_run):
        super().__init__(parent)
        self.title(step.label)
        self.configure(background=theme.PALETTE["pale"])
        self.transient(parent)
        self.step = step
        self.on_run = on_run
        self.vars = {}

        body = ttk.Frame(self, padding=14)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text=step.label, style="Title.TLabel").pack(anchor="w")
        if step.help:
            ttk.Label(body, text=step.help, style="Hint.TLabel", wraplength=430,
                      justify="left").pack(anchor="w", pady=(2, 10))
        ttk.Separator(body).pack(fill=tk.X, pady=(0, 10))

        if step.stepno:
            row = ttk.Frame(body)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text="Step number", width=22).pack(side=tk.LEFT)
            self.stepno_var = tk.StringVar(value=str(stepno))
            ttk.Entry(row, textvariable=self.stepno_var, width=8).pack(side=tk.LEFT)
            ttk.Label(body, text="Prefixed to the output file name.",
                      style="Hint.TLabel").pack(anchor="w", padx=(22, 0))
        else:
            self.stepno_var = None

        for param in step.params:
            self._add_field(body, param)

        if not step.params and not step.stepno:
            ttk.Label(body, text="This step takes no parameters.",
                      style="Hint.TLabel").pack(anchor="w", pady=6)

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Run", style="Accent.TButton",
                   command=self._run).pack(side=tk.RIGHT, padx=(0, 8))

        self.bind("<Return>", lambda _e: self._run())
        self.bind("<Escape>", lambda _e: self.destroy())

    def _add_field(self, body, param):
        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=param.label, width=22).pack(side=tk.LEFT)

        var = tk.StringVar(value=str(param.default))
        self.vars[param.name] = var

        if param.kind == "bool":
            var.set("1" if str(param.default).lower() in ("1", "true", "yes") else "")
            ttk.Checkbutton(row, variable=var, onvalue="1", offvalue="").pack(side=tk.LEFT)
        else:
            ttk.Entry(row, textvariable=var, width=34).pack(side=tk.LEFT, fill=tk.X, expand=True)
            if param.kind in ("path", "dirpath"):
                ttk.Button(row, text="Browse", width=8,
                           command=lambda v=var, k=param.kind: self._browse(v, k)).pack(
                    side=tk.LEFT, padx=(6, 0))

        if param.help:
            ttk.Label(body, text=param.help, style="Hint.TLabel", wraplength=430,
                      justify="left").pack(anchor="w", padx=(22, 0))

    def _browse(self, var, kind):
        chosen = filedialog.askdirectory() if kind == "dirpath" else filedialog.askopenfilename()
        if chosen:
            var.set(chosen)

    def _run(self):
        values = {}
        try:
            for param in self.step.params:
                raw = self.vars[param.name].get()
                if param.required and not raw.strip():
                    raise ValueError(f"{param.label} is required.")
                values[param.name] = parse_value(param.kind, raw)
            stepno = int(self.stepno_var.get()) if self.stepno_var is not None else None
        except Exception as exc:
            messagebox.showerror("Check the values", str(exc), parent=self)
            return

        # Drop anything left blank so the function keeps its own default.
        values = {k: v for k, v in values.items() if v is not None}
        self.destroy()
        self.on_run(self.step, values, stepno)


# ---------------------------------------------------------------------------
# stdout capture, so the pipeline's prints land in the log pane
# ---------------------------------------------------------------------------

class _LogStream:
    def __init__(self, sink, original):
        self.sink = sink
        self.original = original

    def write(self, text):
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        if text:
            self.sink(text)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# the main window
# ---------------------------------------------------------------------------

class FPVSApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("FPVS pipeline")
        self.geometry("1180x780")
        theme.apply(self)

        self.filepath = None
        self.npy_data = None
        self.meta_data = None
        self.ica_state = None
        self.stepno = 1
        self.busy = False

        self.results = queue.Queue()
        self.log_lines = queue.Queue()

        self._build_top_bar()
        self._build_body()
        self._build_status_bar()

        sys.stdout = _LogStream(self.log_lines.put, sys.stdout)
        self.after(100, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.log("Ready. Load a data file to begin.\n")

    # -- layout -------------------------------------------------------------

    def _build_top_bar(self):
        bar = ttk.Frame(self, padding=(12, 10))
        bar.pack(fill=tk.X)

        ttk.Label(bar, text="FPVS pipeline", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Button(bar, text="Load data", style="Accent.TButton",
                   command=self.load_data).pack(side=tk.LEFT, padx=(16, 8))
        ttk.Button(bar, text="Reload", command=self.reload_current).pack(side=tk.LEFT)

        self.path_var = tk.StringVar(value="no file loaded")
        ttk.Label(bar, textvariable=self.path_var, style="Hint.TLabel").pack(
            side=tk.LEFT, padx=(16, 0))

        ttk.Separator(self).pack(fill=tk.X)

    def _build_body(self):
        body = ttk.Frame(self, padding=(12, 10))
        body.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(body, style="Panel.TFrame", padding=10)
        side.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(side, text="Steps", style="Panel.TLabel").pack(anchor="w", pady=(0, 6))

        for menu_name in registry.MENUS:
            steps = registry.steps_for(menu_name)
            if not steps:
                continue
            button = ttk.Menubutton(side, text=menu_name, width=20, direction="right")
            dropdown = tk.Menu(button, tearoff=False,
                               background=theme.PALETTE["white"],
                               foreground=theme.PALETTE["black"],
                               activebackground=theme.PALETTE["red"],
                               activeforeground=theme.PALETTE["white"],
                               font=theme.FONT)
            for step in steps:
                dropdown.add_command(label=step.label,
                                     command=lambda s=step: self.open_step(s))
            button["menu"] = dropdown
            button.pack(anchor="w", pady=3, fill=tk.X)

        ttk.Separator(side).pack(fill=tk.X, pady=10)
        self.ica_var = tk.StringVar(value="ICA: not fitted")
        ttk.Label(side, textvariable=self.ica_var, style="Panel.TLabel").pack(anchor="w")

        centre = ttk.Frame(body)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.summary = SummaryView(centre)
        self.summary.pack(fill=tk.BOTH, expand=True)

        ttk.Label(centre, text="Log", style="Header.TLabel").pack(anchor="w", pady=(8, 2))
        self.log_widget = tk.Text(centre, height=8, wrap="word", **theme.text_options())
        self.log_widget.pack(fill=tk.X)
        self.log_widget.configure(state=tk.DISABLED)

    def _build_status_bar(self):
        ttk.Separator(self).pack(fill=tk.X)
        bar = ttk.Frame(self, padding=(12, 6))
        bar.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bar, textvariable=self.status_var, style="Hint.TLabel").pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT)

    # -- logging ------------------------------------------------------------

    def log(self, text):
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    # -- loading ------------------------------------------------------------

    def load_data(self):
        chosen = filedialog.askopenfilename(
            title="Load data",
            filetypes=[("All supported", "*.npy *.pkl *.bdf *.mat *.lw6"),
                       ("Processed data", "*.npy *.pkl"),
                       ("BioSemi", "*.bdf"),
                       ("Letswave", "*.mat *.lw6"),
                       ("All files", "*.*")])
        if not chosen:
            return

        path = Path(chosen)
        suffix = path.suffix.lower()
        if suffix in BDF_SUFFIXES:
            self.filepath = path
            self._start("Importing .bdf", app.extractdatafrombdf, (path,), {}, on_thread=True)
        elif suffix in MAT_SUFFIXES:
            self.filepath = path
            self._start("Converting .mat", app.convertMATtoPY, (path,), {}, on_thread=True)
        elif suffix in DATA_SUFFIXES:
            self.filepath = path
            self.reload_current()
        else:
            messagebox.showerror("Unsupported file", f"Cannot load {path.name}.", parent=self)

    def reload_current(self):
        """Re-read the current file from disk and repaint the centre panel."""
        if self.filepath is None:
            self.summary.clear()
            return
        try:
            npy_data, meta_data, _, _ = app.loadalldata(self.filepath)
        except Exception as exc:
            self.npy_data = self.meta_data = None
            self.summary.clear()
            self.log(f"Could not read {self.filepath}: {exc}\n")
            return

        self.npy_data = npy_data
        self.meta_data = meta_data
        self.path_var.set(str(self.filepath))
        self.summary.set_data(npy_data, meta_data)
        self.log(f"Loaded {self.filepath}  shape {np.asarray(npy_data).shape}\n")

    # -- running a step ------------------------------------------------------

    def open_step(self, step):
        if self.busy:
            messagebox.showinfo("Busy", "Wait for the running step to finish.", parent=self)
            return
        ParamScreen(self, step, self.stepno, self.run_step)

    def run_step(self, step, values, stepno):
        if stepno is not None:
            self.stepno = stepno
        try:
            builder = BUILDERS[step.builder]
        except KeyError:
            messagebox.showerror("Registry error",
                                 f"{step.key}: unknown builder {step.builder!r}", parent=self)
            return

        try:
            func, args, kwargs = builder(self, step, values)
        except Exception as exc:
            messagebox.showerror(step.label, str(exc), parent=self)
            return

        self._start(step.label, func, args, kwargs, on_thread=not step.ui, step=step)

    def _start(self, title, func, args, kwargs, on_thread, step=None):
        self.busy = True
        self.status_var.set(f"Running: {title}")
        self.progress.start(12)
        self.log(f"\n--- {title} ---\n")

        if on_thread:
            worker = threading.Thread(
                target=self._work, args=(func, args, kwargs, step), daemon=True)
            worker.start()
        else:
            # Anything that opens its own windows has to stay on the Tk thread.
            self.update_idletasks()
            try:
                result = func(*args, **kwargs)
            except Exception:
                self.results.put(("error", traceback.format_exc(), step))
            else:
                self.results.put(("ok", result, step))

    def _work(self, func, args, kwargs, step):
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.results.put(("error", traceback.format_exc(), step))
        else:
            self.results.put(("ok", result, step))

    # -- results -------------------------------------------------------------

    def _drain(self):
        while True:
            try:
                self.log(self.log_lines.get_nowait())
            except queue.Empty:
                break

        while True:
            try:
                status, payload, step = self.results.get_nowait()
            except queue.Empty:
                break
            self._finish(status, payload, step)

        self.after(100, self._drain)

    def _finish(self, status, payload, step):
        self.busy = False
        self.progress.stop()

        if status == "error":
            self.status_var.set("Failed")
            self.log(payload + "\n")
            messagebox.showerror("Step failed", payload.strip().splitlines()[-1], parent=self)
            return

        self.status_var.set("Idle")

        if step is not None and step.builder == "ica_fit":
            self._store_ica(payload)
            return

        new_path = self._path_from(payload)
        if new_path is not None:
            self.filepath = new_path
            if step is not None and step.stepno:
                self.stepno += 1
            self.reload_current()
        elif step is not None and step.builder in ("file_only", "folder_event"):
            self.reload_current()

    def _store_ica(self, payload):
        """Keep the ICA result in memory and show the component overlay."""
        try:
            ica, raw, ica_data, eog_indices = payload
        except Exception:
            self.log("ICA returned an unexpected result, nothing stored.\n")
            return

        self.ica_state = {"ica": ica, "raw": raw, "ica_data": ica_data}
        self.ica_var.set(f"ICA: {ica_data.shape[0]} components")
        self.log(f"Suspected EOG components: {eog_indices}\n")

        if self.npy_data is None:
            return
        labels = np.array(self.meta_data["chanlocs"]["labels"], dtype=object)
        subjid = self.filepath.stem.split()[-1] if self.filepath else None
        try:
            cf.showmeICAoverlayedondata(self.npy_data, ica_data, labels=labels, subjid=subjid)
        except Exception:
            self.log(traceback.format_exc() + "\n")

    @staticmethod
    def _path_from(result):
        """Pull the output file path out of whatever a step returned."""
        if isinstance(result, Path):
            return result
        if isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, Path):
                    return item
        return None

    # -- shutdown ------------------------------------------------------------

    def _close(self):
        if isinstance(sys.stdout, _LogStream):
            sys.stdout = sys.stdout.original
        self.destroy()


def main():
    FPVSApp().mainloop()


if __name__ == "__main__":
    main()
