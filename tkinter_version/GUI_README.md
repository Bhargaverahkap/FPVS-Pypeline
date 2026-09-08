# FPVS desktop app

```bash
cd tkinter_version
python fpvs_gui.py
```

Run it from inside this folder so `import cust_funcs` picks up the Tkinter copy
rather than the notebook one in the repository root.

## The window

```
┌──────────────────────────────────────────────────────────────────────┐
│ FPVS pipeline   [Load data] [Reload]   path of the current file      │
├───────────────┬──────────────────────────────────────────────────────┤
│ Preprocessing▸│                                                      │
│ Postprocess. ▸│              summary plot, always on screen          │
│ Merged events▸│              (blank until a file is loaded)          │
│ Display      ▸│                                                      │
│               ├──────────────────────────────────────────────────────┤
│ ICA: …        │  Log                                                 │
├───────────────┴──────────────────────────────────────────────────────┤
│ Idle                                                  [progress bar] │
└──────────────────────────────────────────────────────────────────────┘
```

**Load data** picks the loader from the file extension: `.bdf` imports through
`extractdatafrombdf`, `.mat`/`.lw6` through `convertMATtoPY`, `.npy`/`.pkl` load
straight in. Every step afterwards works on the file named in the top bar. When
a step writes a new file, that file becomes the current one and the centre panel
redraws from it.

**The centre panel** is the summary viewer: domain switch, frequency range,
epoch list and channel list. With no data it draws nothing at all - no message,
no axes.

**The four menus** come from `gui_registry.py`. Preprocessing is steps 1-8 of
the FPVS cheat sheet, Postprocessing is steps 9-14, Merged events holds the
`ME*` family, and Display holds the viewers, which never write a file.

**Threading.** Tkinter is single threaded, so a long function would freeze the
window. Steps that only compute run on a worker thread and report back through a
queue; steps that open windows of their own run on the main thread and are
marked `ui=True` in the registry. Only one step runs at a time.

**ICA** is two steps. *ICA - fit and inspect* fits the matrix on a worker
thread, then shows the component overlay and keeps `ica`, `raw` and `ica_data`
in memory - the sidebar shows how many components were found. *ICA - remove
components* takes the numbers you noted from the overlay and applies them. The
state lives in memory only; it is gone when the app closes.

## Changing which button calls which function

Everything is in `gui_registry.py`. The app itself never names a pipeline
function, so you never have to touch `fpvs_gui.py` to rewire a button.

### Move a button to a different menu

Change its `menu` field to one of the names in `MENUS`:

```python
Step("downsample", "Postprocessing", "6. Downsample", ...)
#                   ^^^^^^^^^^^^^^ was "Preprocessing"
```

### Point a button at a different function

Change `func`. The prefix says which module: `app.` is
`FPyVS_appylication`, `cf.` is `cust_funcs`.

```python
Step("view_summary", "Display", "Summary plot (lines)",
     "app.showmesummaryplot_alt",   # was app.showmesummaryplot
     "data_only", ui=True),
```

### Delete a button

Delete its `Step(...)` block. The function stays in the module, it just stops
having a menu entry. That is how `performICA` and `overlayICAondata` are hidden:
they are still importable, they simply are not registered.

### Add a button

Append a `Step(...)`. The five things you must decide:

| field | what it is |
|---|---|
| `key` | any unique string |
| `menu` | one of `MENUS` |
| `label` | the menu text |
| `func` | `"app.<name>"` or `"cf.<name>"` |
| `builder` | how the arguments get assembled, see the table below |

Then list the parameters you want the user to fill in.

```python
Step("my_step", "Preprocessing", "My new step",
     "app.mynewfunction", "file_step", stepno=True,
     params=[
         Param("threshold", "Threshold (µV)", "float", "75",
               "Anything above this is treated as an artefact.", required=True),
     ],
     help="One sentence describing what the step does."),
```

### Builders

The builder decides what the function is called with. Pick the one that matches
the function's signature:

| builder | call shape |
|---|---|
| `file_step` | `func(filepath, stepno, **params)` |
| `file_only` | `func(filepath, **params)` |
| `folder_event` | `func(folderpath, event_label)` |
| `ica_fit` | `func(filepath, **params)`, result kept in memory, overlay shown |
| `ica_apply` | `func(filepath, ica, raw, rmidx)` from the stored ICA |
| `data_only` | `func(npy_data, meta_data)` on the loaded data |
| `ica_overlay` | `func(npy_data, ica_data, labels, chid, subjid)` |
| `spectrogram` | `func(npy_data, chid, fs, titlestr, freqlim, binsize, isoverlap)` |
| `mw_spectrogram` | as above with an extra `epochid` |
| `topomap` | `func(activations, meta_data, title)`, activations reduced to one value per channel |

If none of them fits, add a function to `BUILDERS` in `fpvs_gui.py`. A builder
takes `(app_window, step, values)` and returns `(function, args, kwargs)`.

### Parameter kinds

| kind | typed as | becomes |
|---|---|---|
| `int` | `8` | `8` |
| `float` | `1.2` | `1.2` |
| `str` | `Fp1` | `"Fp1"` |
| `bool` | checkbox | `True` / `False` |
| `numbers` | `4, 0.1, 100` | `[4, 0.1, 100]` (whole numbers stay `int`) |
| `ints` | `1, 2, 5` | `[1, 2, 5]` |
| `strs` | `EXG1, EXG2` | `["EXG1", "EXG2"]` |
| `path` | picked with Browse | the path string |
| `dirpath` | picked with Browse | the folder string |
| `json` | `{"TOP": [210, 212]}` | the dict |

An empty field is dropped from the call, so the function keeps its own default.
Mark a parameter `required=True` when the function cannot run without it - the
screen then refuses to submit while the field is empty.

`stepno=True` adds the step-number field. The app pre-fills it with a counter
that goes up each time a numbered step succeeds, and the number becomes the
prefix of the output file name.

## Colours

`theme.py` holds the palette taken from the project logo and applies it to both
ttk widgets and matplotlib. Red is for actions and warnings only; plot traces
use matplotlib default colours. Change a value there and the whole app
follows.

## Tk

Tk is not installed by pip - it ships with Python. `setup_windows.bat` and
`setup_macos.sh` check for it before creating the environment: the Linux branch
installs it (`python3-tk` / `python3-tkinter`), the macOS branch installs
`python-tk` through Homebrew, and the Windows branch tells you to re-run the
Python installer and tick "tcl/tk and IDLE". Check it yourself with
`python -c "import tkinter"`.

## Known limits

- Plotly viewers (`showme3DTopomap`, `showme3DTopomapnewmesh`) open in the web
  browser. That is how plotly works; they are not embedded in the window.
- The ICA state is not written to disk.
- One step at a time. The menus stay live but a second step is refused while one
  is running.
