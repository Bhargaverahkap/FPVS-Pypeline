# Tkinter build

A second copy of the pipeline modules that runs outside Jupyter.

## Two builds, same functions

| | Jupyter build | Tkinter build |
|---|---|---|
| Location | repository root | `tkinter_version/` |
| Interactive viewers | `ipywidgets` + `IPython.display` | Tkinter windows with an embedded matplotlib canvas |
| Runs in | notebooks only | any Python script, notebooks too |

Every function keeps the same name, arguments and return values in both builds.
Only the four interactive viewers were rewritten:

- `FPyVS_appylication.showmesummaryplot`
- `FPyVS_appylication.showmesummaryplot_alt`
- `cust_funcs.showmeICAoverlayedondata`
- `cust_funcs.showmeSignalUIandInterp`

In this build those two summary viewers share one implementation,
`_summaryplot(npy_data, meta_data, use_stem, epoch_offset)`. `showmesummaryplot`
calls it with lines, `showmesummaryplot_alt` with stems and epochs numbered
from 1, matching the Jupyter build.

`_tk_helpers.py` holds the shared window machinery (`PlotWindow`,
`make_listbox`, `selection`) and is only used by this build.

## Blocking behaviour

Each viewer opens a window and blocks until it is closed, the way a notebook
cell blocks while its widget is on screen. Everything runs on the Tk main
thread, so a long computation started from a viewer will freeze the window
until it finishes.

## Running it

```bash
cd tkinter_version
python
>>> from pathlib import Path
>>> import FPyVS_appylication as FP
>>> npy_data, meta_data, filepath = FP.convertMATtoPY(Path("../your file"))
>>> FP.showmesummaryplot(npy_data, meta_data)
```

Run from inside this folder so `import cust_funcs as cf` picks up the Tkinter
copy rather than the notebook one in the repository root.

Data assets (`biosemi_locations_64_10-20_fixP9P10_add4.xyz`, `skull_1.obj`,
`head_openneck.obj`) are not duplicated. This build resolves them against the
repository root through `_ASSET_DIR`.

## Requirements

Same `requirements.txt` as the root build, plus Tk:

- Windows and macOS: Tk ships with the python.org installer, nothing to do.
- Linux: `sudo apt install python3-tk`.

Check with `python -c "import tkinter"`.

## Keeping the two builds in sync

The copies are independent files. A change to a non-viewer function in the root
build has to be repeated here (and the other way round). `diff` shows what
drifted:

```bash
diff ../FPyVS_appylication.py FPyVS_appylication.py
diff ../cust_funcs.py cust_funcs.py
```

Expected differences: the header comment, the `_ASSET_DIR` block, the removed
`ipywidgets` imports and the four rewritten viewers.
