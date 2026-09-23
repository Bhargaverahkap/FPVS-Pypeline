"""Every RLW wrapper must take a filepath and return (npy_data, meta_data, filepath)."""
import shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw

work = Path(tempfile.mkdtemp())
fs, n, nch, nep = 250, 256, 6, 4
rng = np.random.default_rng(0)
t = np.arange(n) / fs
labels = ["Fp1", "Fp2", "Cz", "Pz", "EK1", "EOG"]
ang = np.linspace(0, np.pi, nch)

def write(name, data, extra=None):
    meta = {"fs": fs, "xstart": 0.0, "xstep": 1/fs, "filetype": "time_amplitude",
            "history": {},
            "chanlocs": {"labels": np.array(labels, dtype=object),
                         "X": np.sin(ang), "Y": np.cos(ang), "Z": np.full(nch, 0.3),
                         "topo_enabled": np.ones(nch, dtype=int),
                         "SEEG_enabled": np.zeros(nch, dtype=int)},
            "events": {"code": np.array(["go", "go", "stim", "go"], dtype=object),
                       "latency": np.array([0.10, 0.20, 0.30, 0.40]),
                       "epoch": np.array([0, 1, 2, 3])}}
    meta.update(extra or {})
    path = work / name
    rlw.savealldata(data, meta, path)
    return path

base = np.stack([np.sin(2*np.pi*10*t) + 0.1*rng.standard_normal(n) for _ in range(nch)], axis=1)
ecg = np.zeros(n)
for b in range(0, n, 100):
    ecg[b:b+3] = [2, 8, 2]
base[:, 4] = ecg
data = np.repeat(base[:, :, None], nep, axis=2)
main = write("subject A", data)
other = write("subject B", data * 2)

epochdata = [{"data": {"rt": 0.5}}, {"data": {"rt": 0.1}},
             {"data": {"rt": 0.9}}, {"data": {"rt": 0.3}}]
with_epochdata = write("subject C", data, {"epochdata": epochdata})
continuous = write("continuous", data[:, :, :1])

# the arguments each wrapper needs beyond (filepath, stepno)
CALLS = {
 "arrange_index": ((main, 1), {"index_idx": [0]}),
 "average_epochs_sliding": ((main, 1), {"width": 0.05}),
 "average_erpimage": ((main, 1), {"num_lines": 4, "smooth": False}),
 "concatenate_epochs": ((main, 1), {}),
 "crop": ((main, 1), {"x_start": 0.1, "x_size": 50}),
 "cwt": ((main, 1), {"low_frequency": 5, "high_frequency": 15, "num_frequency_lines": 4}),
 "cwt_fast": ((main, 1), {"low_frequency": 5, "high_frequency": 15,
                          "num_frequency_lines": 3, "mother_size": 400}),
 "dc_removal": ((main, 1), {}),
 "derivate_signals": ((main, 1), {}),
 "edit_electrodes_info": ((main, 1), {"chanlocs": [{"labels": "Cz", "X": 9.0}]}),
 "edit_electrodes_seeg": ((main, 1), {"list_labels": ["Cz"], "list_x": [1],
                                      "list_y": [2], "list_z": [3]}),
 "equalize_epochs": (([main, other], 1), {"num_epochs": 2}),
 "events_delete_duplicate": ((main, 1), {}),
 "events_level_trigger": ((main, 1), {"selected_channel": "EK1", "threshold": 5,
                                      "min_isi": 0.05}),
 "fft_filter": ((main, 1), {"low_cutoff": 5, "high_cutoff": 20}),
 "find_ekg": ((main, 1), {}),
 "flip_electrodes": ((main, 1), {"chan_label_pairs": [("Fp1", "Fp2")]}),
 "grand_average": (([main, other], 1), {}),
 "hilbert": ((main, 1), {}),
 "hilbert_bands": ((main, 1), {"freq_start": 8, "freq_end": 12, "freq_lines": 2,
                               "freq_width": 4}),
 "ica_unmix": ((main, 1), {"ica_um": np.eye(nch)}),
 "ica_remix": ((main, 1), {"ica_mm": np.eye(nch)}),
 "ifft": ((main, 1), {}),
 "linear_channel_map": ((main, 1), {"num_lines": 8}),
 "linear_csd": ((main, 1), {}),
 "math": ((main, 1, other), {"operation": "A-B"}),
 "math_constant": ((main, 1), {"operation": "multiply", "constant": 2}),
 "merge_channels": (([main, other], 1), {}),
 "merge_index": (([main, other], 1), {}),
 "ocular_remove": ((main, 1), {"eog_channels": ["EOG"]}),
 "pan_tompkin": ((continuous, 1), {"channel_label": "EK1"}),
 "pca_compute": ((main, 1), {}),
 "pool_channels": ((main, 1), {"channel_labels_wanted": ["Fp1", "Fp2"]}),
 "properties": ((main, 1), {"filetype": "frequency_amplitude"}),
 "rectify_signals": ((main, 1), {}),
 "reject_epochs": ((main, 1), {"rejected_epochs": [1]}),
 "reject_epochs_amplitude": ((main, 1), {"criterion": 100}),
 "rereference_advanced": ((main, 1), {"active_channel_labels": ["Fp1"],
                                      "reference_channel_labels": ["Cz"]}),
 "resample": ((main, 1), {"x_sampling_rate": 125}),
 "scalp_csd": ((main, 1), {}),
 "segmentation_ssep": ((main, 1), {"event_labels": ["go"], "cycle_total": 2,
                                   "cycle_frequency": 10.0}),
 "select_epochdata": ((with_epochdata, 1), {"fieldname": "rt", "logical": "<",
                                            "comparison_value": 0.6}),
 "select_events": ((main, 1), {"event_code": "go", "maximum_latency": 1.0}),
 "snr": ((main, 1), {"operation": "subtract"}),
 "sort_epochdata": ((with_epochdata, 1), {"fieldname": "rt"}),
 "sort_events": ((main, 1), {"event_code": "go"}),
 "stfft": ((main, 1), {"hanning_width": 0.1, "low_frequency": 5,
                       "high_frequency": 15, "num_frequency_lines": 4}),
 "stfft_zhang": ((main, 1), {"hanning_width": 0.1, "low_frequency": 5,
                             "high_frequency": 15, "num_frequency_lines": 3}),
 "suppress_artifact": ((main, 1), {"x_start": 0.1, "x_end": 0.12}),
 "suppress_artifact_event": ((main, 1), {"event_code": "go"}),
 "threshold": ((main, 1), {"threshold_value": 0.5, "threshold_criterion": ">"}),
 "wavelet_filter_build": ((main, 1), {"selected_channel": "Cz", "start_frequency": 8,
                                      "end_frequency": 12, "frequency_step": 2}),
 "weighted_channel_average_template": ((main, 1), {"x": 0.05, "num_channels": 3}),
}

results, failures = {}, []
for name, (args, kwargs) in CALLS.items():
    try:
        out = getattr(rlw, name)(*args, **kwargs)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        continue
    triples = out if isinstance(out, list) else [out]
    for triple in triples:
        if not (isinstance(triple, tuple) and len(triple) == 3):
            failures.append(f"{name}: returned {type(out).__name__}, not a 3-tuple")
            break
        npy_data, meta_data, path = triple
        if not isinstance(npy_data, np.ndarray):
            failures.append(f"{name}: first value is {type(npy_data).__name__}")
        if not isinstance(meta_data, dict):
            failures.append(f"{name}: second value is {type(meta_data).__name__}")
        if not isinstance(path, Path):
            failures.append(f"{name}: third value is {type(path).__name__}")
        elif not (path.with_suffix(".npy").exists() and path.with_suffix(".pkl").exists()):
            failures.append(f"{name}: no files written at {path}")
        elif f"1_{rlw.FILE_STEPS[name][0]} " not in path.name:
            failures.append(f"{name}: unexpected output name {path.name}")
    else:
        results[name] = triples[0][2].name

# the two that consume another step's output
mask_path = results and rlw.wavelet_filter_build(main, 1, selected_channel="Cz",
                                                 start_frequency=8, end_frequency=12,
                                                 frequency_step=2)[2]
try:
    out = rlw.wavelet_filter_apply(main, 2, mask_path, channel_name="Cz")
    assert isinstance(out, tuple) and len(out) == 3 and out[2].with_suffix(".npy").exists()
    results["wavelet_filter_apply"] = out[2].name
except Exception as exc:
    failures.append(f"wavelet_filter_apply: {type(exc).__name__}: {exc}")

try:
    tpath = rlw.weighted_channel_average_template(main, 1, x=0.05, num_channels=3)[2]
    tmpl = rlw.loadalldata(tpath)[1]["channel_template"]
    out = rlw.weighted_channel_average_apply(
        main, 2, template_weights=tmpl["weights"], template_labels=tmpl["labels"])
    assert isinstance(out, tuple) and len(out) == 3 and out[2].with_suffix(".npy").exists()
    results["weighted_channel_average_apply"] = out[2].name
except Exception as exc:
    failures.append(f"weighted_channel_average_apply: {type(exc).__name__}: {exc}")

# the saved file must read back to what was returned, and carry the history
npy_data, meta_data, path = rlw.dc_removal(main, 7)
again, meta_again, *_ = rlw.loadalldata(path)
assert np.allclose(again, npy_data), "the saved data differs from what was returned"
assert "7_dc " in str(meta_again.get("history")), meta_again.get("history")
assert path.name.startswith("7_dc "), path.name

print(f"wrappers exercised: {len(results)} of {len(CALLS) + 2}")
if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
else:
    print("ALL FILE WRAPPERS OK")
shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if failures else 0)
