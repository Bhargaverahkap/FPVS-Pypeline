# The button map of the FPVS desktop app.
#
# Every menu entry in the app comes from this file and nothing else. Add an
# entry here and a menu item appears; delete one and it disappears; change a
# `func` and that button calls a different function. The app never hard-codes a
# function name.
#
# See GUI_README.md for a walk-through of each field.

MENUS = ["Preprocessing", "Postprocessing", "Merged events", "Display"]


class Param:
    """One field on a step's parameter screen.

    name    keyword argument name passed to the function
    label   text shown next to the field
    kind    how the typed text is converted:
              "int"     -> int            "float"   -> float
              "str"     -> str            "bool"    -> checkbox, True/False
              "numbers" -> list of numbers, comma separated (ints stay ints)
              "ints"    -> list of ints, comma separated
              "strs"    -> list of strings, comma separated
              "path"    -> file path with a Browse button
              "dirpath" -> folder path with a Browse button
              "json"    -> any Python literal, e.g. a dict
    default  text pre-filled in the field ("" means "leave it to the function")
    help     one line shown under the field
    required True when the function cannot run without it, so the screen refuses
             to submit while the field is empty
    """

    def __init__(self, name, label, kind, default="", help="", required=False):
        self.name = name
        self.label = label
        self.kind = kind
        self.default = default
        self.help = help
        self.required = required


class Step:
    """One menu entry.

    key       unique id, used nowhere but in error messages
    menu      which dropdown it appears under (must be one of MENUS)
    label     the text of the menu item
    func      "app.<name>" for FPyVS_appylication, "cf.<name>" for cust_funcs
    builder   how the arguments are assembled, see fpvs_gui.BUILDERS:
                "file_step"      func(filepath, stepno, **params)
                "file_only"      func(filepath, **params)
                "folder_event"   func(folderpath, event_label)
                "ica_fit"        fits ICA on the current file, then shows the
                                 component overlay and keeps ica/raw in memory
                "ica_apply"      func(filepath, ica, raw, rmidx) using that state
                "data_only"      func(npy_data, meta_data)
                "ica_overlay"    func(npy_data, ica_data, labels, chid, subjid)
                "spectrogram"    func(npy_data, chid, fs, ...)
                "mw_spectrogram" func(npy_data, chid, epochid, fs, ...)
                "topomap"        func(activations, meta_data, title)
    params    list of Param, in the order they appear on the screen
    stepno    True if the function takes the pipeline step number
    ui        True if the function opens windows itself and so must run on the
              main thread; False lets it run on a worker thread
    help      sentence shown at the top of the parameter screen
    """

    def __init__(self, key, menu, label, func, builder, params=(), stepno=False,
                 ui=False, help=""):
        self.key = key
        self.menu = menu
        self.label = label
        self.func = func
        self.builder = builder
        self.params = list(params)
        self.stepno = stepno
        self.ui = ui
        self.help = help


# ---------------------------------------------------------------------------
# The steps. Order here is the order in the dropdown.
# ---------------------------------------------------------------------------

STEPS = [

    # --- Preprocessing: steps 1-8 of the FPVS cheat sheet --------------------
    Step("import_bdf", "Preprocessing", "Import from .bdf file",
         "app.extractdatafrombdf", "file_only",
         help="Reads a BioSemi .bdf recording and writes the .npy/.pkl pair."),

    Step("import_mat", "Preprocessing", "Import from .mat / .lw6",
         "app.convertMATtoPY", "file_only",
         help="Converts a Letswave .mat/.lw6 pair into .npy/.pkl."),

    Step("rename_channels", "Preprocessing", "1. Rename channels",
         "app.renamechannels", "file_step", stepno=True,
         params=[
             Param("targets", "Channels to rename", "strs", "EXG1, EXG2, EXG3, EXG4",
                   "Current labels, comma separated."),
             Param("replacements", "New names", "strs", "I1, I2, PO9, PO10",
                   "Same order as above. Remember the dead electrodes."),
         ],
         help="Edit electrode labels (cheat sheet step 1)."),

    Step("electrode_locations", "Preprocessing", "2. Electrode locations",
         "app.electrodelocationchange", "file_step", stepno=True,
         params=[
             Param("electrodelocationfilepath", "Coordinate file", "path", "",
                   "Leave empty to use biosemi_locations_64_10-20_fixP9P10_add4.xyz."),
         ],
         help="Write electrode coordinates into the metadata (step 2)."),

    Step("delete_channels", "Preprocessing", "3. Remove unused channels",
         "app.delete_channels", "file_step", stepno=True,
         params=[
             Param("deletechnames", "Channels to delete", "strs", "EXG5, EXG6, Status",
                   "Status and the unused externals."),
         ],
         help="Drop channels from the data and the metadata (step 3)."),

    Step("bandpass", "Preprocessing", "4. Butterworth bandpass",
         "app.bandpassfilter", "file_step", stepno=True,
         params=[
             Param("filterparameters", "Order, low cut, high cut", "numbers", "4, 0.1, 100",
                   "Three values: filter order, low cutoff (Hz), high cutoff (Hz)."),
         ],
         help="Bandpass the continuous data (step 4)."),

    Step("notch", "Preprocessing", "5. Notch filter",
         "app.notchfilter", "file_step", stepno=True,
         params=[
             Param("notchparameters", "Slope, then notch frequencies", "numbers", "4, 50, 100",
                   "First value is the slope, the rest are the frequencies to notch out."),
         ],
         help="Remove mains interference and its harmonics (step 5)."),

    Step("downsample", "Preprocessing", "6. Downsample",
         "app.downsampling", "file_step", stepno=True,
         params=[
             Param("dsfact", "Downsample factor", "int", "8",
                   "Integer ratio. 2048 Hz divided by 8 gives 256 Hz."),
         ],
         help="Reduce the sampling rate by an integer factor (step 6)."),

    Step("segmentation", "Preprocessing", "7. Segment on events",
         "app.segmentation", "file_step", stepno=True,
         params=[
             Param("startcode", "Start trigger codes", "ints",
                   "10, 45, 90, 135, 200, 210, 212, 214, 216, 218, 220, 222, 224",
                   "End codes are taken as start code + 1."),
             Param("segmentparams", "Duration, start latency", "numbers", "70, -2",
                   "Seconds: 2 before start + 2 fade in + 64 block + 2 fade out."),
         ],
         help="Cut the recording into epochs around each event (step 7)."),

    Step("ica_fit", "Preprocessing", "ICA - fit and inspect",
         "app.performICA", "ica_fit", ui=True,
         params=[
             Param("ch_name", "Blink reference channel", "str", "Fp1",
                   "Channel used to score components against eye movement."),
         ],
         help="Fit the ICA matrix, then overlay each component on the data so you "
              "can see which one carries the blinks. The result is kept in memory "
              "for the next step."),

    Step("ica_apply", "Preprocessing", "ICA - remove components",
         "app.applyICA", "ica_apply",
         params=[
             Param("rmidx", "Components to remove", "ints", "",
                   "Component numbers from the overlay screen, comma separated.",
                   required=True),
         ],
         help="Subtract the chosen components. Run the fit step first."),

    Step("interpolate", "Preprocessing", "Interpolate channels",
         "app.interpolate", "file_step", stepno=True,
         params=[
             Param("interp_chnames", "Channels to interpolate", "strs", "",
                   "No more than 5% of channels. Never interpolate from a bad channel.",
                   required=True),
             Param("bad_chnames", "Bad channels to avoid", "strs", "",
                   "Excluded from the neighbours used for interpolation."),
         ],
         help="Replace a channel with the mean of its three good neighbours."),

    Step("reference", "Preprocessing", "8. Global re-reference",
         "app.globalreferencing", "file_step", stepno=True,
         params=[
             Param("badchids", "Channels to leave out", "strs", "",
                   "Eye channels and any channel that was not interpolated."),
         ],
         help="Subtract the mean of the good channels from every channel (step 8)."),

    # --- Postprocessing: steps 9-14 -----------------------------------------
    Step("separate_epochs", "Postprocessing", "9. Split epochs per condition",
         "app.separateepochs", "file_only",
         params=[
             Param("mergekeyflag", "Merge paired events", "bool", "True",
                   "Merges TOP/BOTTOM/RIGHT/LEFT pairs into one file each."),
             Param("mergekeys", "Custom merge map", "json", "",
                   'Leave empty for the default, or give a dict such as '
                   '{"TOP": [210, 212]}.'),
         ],
         help="Write one file per condition (step 9)."),

    Step("merge_epochs", "Postprocessing", "9b. Merge epochs across files",
         "app.mergeepochs", "folder_event",
         params=[
             Param("folderpath", "Folder", "dirpath", "",
                   "Folder holding the per-condition files.", required=True),
             Param("event_label", "Event label", "str", "",
                   "File name prefix, for example TOP.", required=True),
         ],
         help="Concatenate the same condition across subjects."),

    Step("fft", "Postprocessing", "10. Fourier transform",
         "app.frequencytransform", "file_step", stepno=True,
         params=[
             Param("freqbin", "Frequency range", "numbers", "0.01, 50",
                   "Low and high edge in Hz. Expect dips at 50 and 100 Hz."),
         ],
         help="Amplitude spectrum, normalised by the number of samples (step 10)."),

    Step("average_trials", "Postprocessing", "11. Average across trials",
         "app.averagingacrosstrials", "file_step", stepno=True,
         help="Mean over the epoch axis (step 11)."),

    Step("chunking", "Postprocessing", "12. Chunk around harmonics",
         "app.chunking", "file_step", stepno=True,
         params=[
             Param("basefreq", "Base frequency (Hz)", "float", "1.2",
                   "The oddball frequency."),
             Param("window_width", "Window width (Hz)", "float", "0.4",
                   "Width of the window cut around each harmonic."),
         ],
         help="Cut a window around the oddball frequency and its harmonics (step 12)."),

    Step("select_chunks", "Postprocessing", "13. Select harmonics",
         "app.selectingchunks", "file_step", stepno=True,
         params=[
             Param("numharmonics", "Number of harmonics", "int", "3",
                   "Splits the chunks into a baseline file and an oddball file."),
         ],
         help="Keep the oddball harmonics and drop the base frequency ones (step 13)."),

    Step("sum_harmonics", "Postprocessing", "14. Sum harmonics",
         "app.sumofharmonics", "file_step", stepno=True,
         help="Sum over the selected harmonics (step 14)."),

    Step("baseline", "Postprocessing", "Baseline correction",
         "app.baselinefiltering", "file_step", stepno=True,
         help="Subtract the mean of the surrounding bins from each chunk."),

    # --- Merged events (ME) --------------------------------------------------
    Step("merge_events", "Merged events", "Merge events across subjects",
         "app.mergeevents", "folder_event",
         params=[
             Param("folderpath", "Folder", "dirpath", "",
                   "Folder holding the per-subject files.", required=True),
             Param("event_label", "Event label", "str", "",
                   "File name prefix, for example TOP.", required=True),
         ],
         help="Stack subjects into one array. Every ME step below expects this file."),

    Step("me_fft", "Merged events", "Fourier transform (ME)",
         "app.MEfrequencytransform", "file_step", stepno=True,
         params=[
             Param("freqbin", "Frequency range", "numbers", "0.05, 50",
                   "Low and high edge in Hz."),
         ]),

    Step("me_average", "Merged events", "Average across trials (ME)",
         "app.MEaveragingacrosstrials", "file_step", stepno=True),

    Step("me_chunking", "Merged events", "Chunk around harmonics (ME)",
         "app.MEchunking", "file_step", stepno=True,
         params=[
             Param("basefreq", "Base frequency (Hz)", "float", "1.2"),
             Param("window_width", "Window width (Hz)", "float", "0.4"),
         ]),

    Step("me_select_chunks", "Merged events", "Select harmonics (ME)",
         "app.MEselectingchunks", "file_step", stepno=True,
         params=[
             Param("numharmonics", "Number of harmonics", "int", "3"),
         ]),

    Step("me_sum_harmonics", "Merged events", "Sum harmonics (ME)",
         "app.MEsumofharmonics", "file_step", stepno=True),

    Step("me_baseline", "Merged events", "Baseline correction (ME)",
         "app.MEbaselinefiltering", "file_step", stepno=True),

    # --- Display: read only, never writes a file -----------------------------
    Step("view_summary", "Display", "Summary plot (lines)",
         "app.showmesummaryplot", "data_only", ui=True,
         help="The centre panel in its own window."),

    Step("view_summary_alt", "Display", "Summary plot (stems)",
         "app.showmesummaryplot_alt", "data_only", ui=True,
         help="Same viewer, stem plot in the frequency domain."),

    Step("view_ica_overlay", "Display", "ICA components over data",
         "cf.showmeICAoverlayedondata", "ica_overlay", ui=True,
         params=[
             Param("chid", "Channel index", "int", "",
                   "Leave empty to use Fp1."),
         ],
         help="Needs an ICA fit from the Preprocessing menu."),

    Step("view_stft", "Display", "STFT spectrogram",
         "cf.showmeSTFTSpectrogram", "spectrogram", ui=True,
         params=[
             Param("chid", "Channel index", "int", "0"),
             Param("titlestr", "Title", "str", ""),
             Param("freqlim", "Frequency limits", "numbers", ""),
             Param("binsize", "Bin size", "float", ""),
             Param("isoverlap", "Overlap", "float", ""),
         ]),

    Step("view_mw", "Display", "Morlet spectrogram",
         "cf.showmeMWSpectrogram", "mw_spectrogram", ui=True,
         params=[
             Param("chid", "Channel index", "int", "0"),
             Param("epochid", "Epoch index", "int", "0"),
             Param("titlestr", "Title", "str", ""),
             Param("freqlim", "Frequency limits", "numbers", ""),
             Param("binsize", "Bin size", "float", ""),
             Param("isoverlap", "Overlap", "float", ""),
         ]),

    Step("view_topo_2d", "Display", "2D topomap",
         "cf.showme2DTopomap", "topomap", ui=True,
         params=[
             Param("sample", "Sample index", "int", "",
                   "Row of the data to map. Leave empty to average over all rows."),
             Param("title", "Title", "str", "2D EEG Topomap"),
         ]),

    Step("view_topo_3d", "Display", "3D topomap (skull mesh)",
         "cf.showme3DTopomap", "topomap", ui=True,
         params=[
             Param("sample", "Sample index", "int", ""),
             Param("title", "Title", "str", "3D EEG Topography"),
         ],
         help="Opens in your web browser: this one is drawn with plotly."),

    Step("view_topo_3d_head", "Display", "3D topomap (head mesh)",
         "cf.showme3DTopomapnewmesh", "topomap", ui=True,
         params=[
             Param("sample", "Sample index", "int", ""),
             Param("title", "Title", "str", "3D EEG Topography"),
         ],
         help="Opens in your web browser: this one is drawn with plotly."),
]


def steps_for(menu):
    """Steps belonging to one dropdown, in registry order."""
    return [s for s in STEPS if s.menu == menu]


def step_by_key(key):
    for s in STEPS:
        if s.key == key:
            return s
    raise KeyError(f"no step registered under {key!r}")
