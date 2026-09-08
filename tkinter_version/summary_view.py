# The summary viewer as an embeddable widget.
#
# The app keeps one of these in the centre of the main window at all times. It
# draws the same time/frequency plot as FPyVS_appylication.showmesummaryplot,
# but into a frame you give it rather than a window of its own, so it can sit
# permanently on the main screen. With no data loaded it shows a blank canvas.

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import theme


class SummaryView(ttk.Frame):
    """Channel picker, domain switch and a matplotlib canvas."""

    def __init__(self, parent, use_stem=False, epoch_offset=0):
        super().__init__(parent)
        self.use_stem = use_stem
        self.epoch_offset = epoch_offset

        self.npy_data = None
        self.meta_data = None
        self.labels = None
        self.fs = None

        controls = ttk.Frame(self, style="Panel.TFrame", padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        self.controls = controls

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(7, 4.5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.domain = tk.StringVar(value="Time")
        self.freq_min = tk.StringVar(value="0")
        self.freq_max = tk.StringVar(value="20")

        ttk.Label(controls, text="Domain", style="Panel.TLabel").pack(anchor="w")
        for option in ("Time", "Frequency"):
            ttk.Radiobutton(controls, text=option, value=option, variable=self.domain,
                            command=self._toggle_freq).pack(anchor="w")

        self.freq_box = ttk.Frame(controls, style="Panel.TFrame")
        ttk.Label(self.freq_box, text="Freq range (Hz)", style="Panel.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(self.freq_box, textvariable=self.freq_min, width=6).grid(row=1, column=0)
        ttk.Entry(self.freq_box, textvariable=self.freq_max, width=6).grid(row=1, column=1)
        for var in (self.freq_min, self.freq_max):
            var.trace_add("write", lambda *_: self.refresh())

        self.epoch_label = ttk.Label(controls, text="Epoch", style="Panel.TLabel")
        self.epoch_box = tk.Listbox(controls, selectmode=tk.BROWSE, height=6,
                                    exportselection=False, **theme.listbox_options())
        self.epoch_box.bind("<<ListboxSelect>>", lambda _e: self.refresh())

        self.channel_label = ttk.Label(controls, text="Channels", style="Panel.TLabel")
        self.channel_label.pack(anchor="w", pady=(8, 0))
        self.channel_box = tk.Listbox(controls, selectmode=tk.EXTENDED, height=14,
                                      exportselection=False, **theme.listbox_options())
        self.channel_box.pack(anchor="w")
        self.channel_box.bind("<<ListboxSelect>>", lambda _e: self.refresh())

        self.clear()

    # -- data ---------------------------------------------------------------

    def clear(self):
        """Forget the data and show an empty canvas."""
        self.npy_data = None
        self.meta_data = None
        self.channel_box.delete(0, tk.END)
        self.epoch_box.delete(0, tk.END)
        self._toggle_epochs()
        self._toggle_freq()
        self._blank()

    def set_data(self, npy_data, meta_data):
        """Show a new dataset, keeping the current domain choice."""
        if npy_data is None or meta_data is None:
            self.clear()
            return

        self.npy_data = np.asarray(npy_data)
        self.meta_data = meta_data
        self.fs = meta_data.get("fs", 256)
        labels = np.squeeze(meta_data["chanlocs"]["labels"])
        self.labels = np.atleast_1d(labels).astype(str)

        self.channel_box.delete(0, tk.END)
        for idx in range(self.npy_data.shape[1]):
            name = self.labels[idx] if idx < len(self.labels) else f"ch {idx}"
            self.channel_box.insert(tk.END, name)
        for idx in range(min(4, self.channel_box.size())):
            self.channel_box.selection_set(idx)

        self.epoch_box.delete(0, tk.END)
        if self.npy_data.ndim >= 3:
            for idx in range(self.npy_data.shape[2]):
                self.epoch_box.insert(tk.END, f"Ep: {idx + self.epoch_offset}")
            self.epoch_box.selection_set(0)

        self._toggle_epochs()
        self._toggle_freq()
        self.refresh()

    # -- drawing ------------------------------------------------------------

    def _blank(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def _toggle_freq(self):
        # The range boxes belong directly under the domain switch, so they are
        # packed ahead of the channel list rather than at the end of the column.
        if self.domain.get() == "Frequency" and self.npy_data is not None:
            self.freq_box.pack(anchor="w", pady=(4, 0), before=self.channel_label)
        else:
            self.freq_box.pack_forget()

    def _toggle_epochs(self):
        # A 2D dataset has no epochs, so the list is hidden rather than empty.
        if self.npy_data is not None and self.npy_data.ndim >= 3:
            self.epoch_label.pack(anchor="w", pady=(8, 0), before=self.channel_label)
            self.epoch_box.pack(anchor="w", before=self.channel_label)
        else:
            self.epoch_label.pack_forget()
            self.epoch_box.pack_forget()

    def _float(self, var, fallback):
        try:
            return float(var.get())
        except (TypeError, ValueError):
            return fallback

    def refresh(self, *_):
        """Redraw from the current widget state."""
        self._toggle_freq()
        if self.npy_data is None:
            self._blank()
            return

        selected = list(self.channel_box.curselection())
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.ax = ax

        if not selected:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Select channels from the list", ha="center", va="center",
                    color=theme.PALETTE["dark"])
            self.canvas.draw_idle()
            return

        data = self.npy_data
        n_timepoints = data.shape[0]
        fs = self.fs
        is3d = data.ndim >= 3
        epoch_sel = list(self.epoch_box.curselection())
        epoch_idx = epoch_sel[0] if epoch_sel else 0
        domain = self.domain.get()

        for s_idx in selected:
            series = data[:, s_idx, epoch_idx] if is3d else data[:, s_idx]
            series = np.squeeze(series)
            label = self.labels[s_idx] if s_idx < len(self.labels) else f"ch {s_idx}"

            if domain == "Time":
                ax.plot(np.arange(n_timepoints) / fs, series, label=label)
            else:
                fft_vals = np.fft.rfft(series)
                fft_freqs = np.fft.rfftfreq(n_timepoints, d=1 / fs)
                magnitude = (np.abs(fft_vals) / n_timepoints) * 2
                fmin = self._float(self.freq_min, 0.0)
                fmax = self._float(self.freq_max, fs / 2)
                keep = np.where((fft_freqs >= fmin) & (fft_freqs <= fmax))
                if self.use_stem:
                    ax.stem(fft_freqs[keep], magnitude[keep], label=label,
                            linefmt="-", markerfmt="o", basefmt=" ")
                else:
                    ax.plot(fft_freqs[keep], magnitude[keep], label=label)
                ax.set_xlim(fmin, fmax)

        if domain == "Time":
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(r"Amplitude ($\mu$V)")
        else:
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(r"Magnitude ($\mu$V)")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.2))
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        suffix = f" | Epoch: {epoch_idx + self.epoch_offset}" if is3d else ""
        ax.set_title(f"Domain: {domain}{suffix}")

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()
