import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw

fs = 100
meta = {"fs": fs, "xstart": 0.0, "xstep": 1/fs,
        "chanlocs": {"labels": np.array(["Fp1","Trig"], dtype=object)},
        "events": {"code": np.array(["go","go","stop"], dtype=object),
                   "latency": np.array([0.5, 0.9, 0.2]),
                   "epoch": np.array([0, 1, 2])}}
data = np.arange(10*2*3, dtype=float).reshape(10, 2, 3)
ok = []

# select_events: "go" within 0..0.6 -> only epoch 0
out, m = rlw.select_events_data(data, meta, "go", 0, 0.6)
assert out.shape[2] == 1 and np.allclose(out[:, :, 0], data[:, :, 0])
out, _ = rlw.select_events_data(data, meta, "go", 0, 1.0)
assert out.shape[2] == 2
out, _ = rlw.select_events_data(data, meta, "nope")
assert out.shape[2] == 3, "unknown code should change nothing"
ok.append("select_events")

# sort_events: descending latency -> epoch 1 (0.9) before epoch 0 (0.5)
out, m = rlw.sort_events_data(data, meta, "go", "descend")
assert out.shape[2] == 2 and np.allclose(out[:, :, 0], data[:, :, 1])
out, _ = rlw.sort_events_data(data, meta, "go", "ascend", discard_empty=False)
assert out.shape[2] == 3 and np.allclose(out[:, :, 2], data[:, :, 2])
ok.append("sort_events")

# events_delete_duplicate
dmeta = dict(meta)
dmeta["events"] = {"code": np.array(["go","go","go","stop"], dtype=object),
                   "latency": np.array([0.5, 0.5, 0.55, 0.5]),
                   "epoch": np.array([0, 0, 0, 0])}
m = rlw.events_delete_duplicate_data(dmeta)
assert len(rlw.events_to_list(m)) == 3, rlw.events_to_list(m)   # exact dup only
m = rlw.events_delete_duplicate_data(dmeta, exact_latencies=False, tolerance=0.1)
assert len(rlw.events_to_list(m)) == 2, rlw.events_to_list(m)   # 0.55 folds in
ok.append("events_delete_duplicate")

# events_level_trigger: two pulses on the Trig channel
trig = np.zeros((100, 2, 1)); trig[10:15, 1, 0] = 5000; trig[60:65, 1, 0] = 5000
tmeta = dict(meta); tmeta["events"] = {"code": np.array([], dtype=object),
                                       "latency": np.array([]), "epoch": np.array([])}
m = rlw.events_level_trigger_data(trig, tmeta, "Trig", threshold=1000, min_isi=0.1)
evs = rlw.events_to_list(m)
assert len(evs) == 2, evs
assert np.allclose([e["latency"] for e in evs], [0.10, 0.60]), evs
ok.append("events_level_trigger")

# average_epochs_sliding
ramp = np.arange(10, dtype=float).reshape(10, 1, 1)
# width 3 samples -> half window round(3/2) = 2, so the window spans 5 samples,
# the same rounding MATLAB's round(binwidth/2) does
out, _ = rlw.average_epochs_sliding_data(ramp, meta, "average", width=3/fs)
assert np.isclose(out[5, 0, 0], 5.0), out[5, 0, 0]        # mean(3..7)
assert np.isclose(out[0, 0, 0], 1.0), out[0, 0, 0]        # window clipped to 0..2
out, _ = rlw.average_epochs_sliding_data(ramp, meta, "max", width=3/fs)
assert np.isclose(out[5, 0, 0], 7.0), out[5, 0, 0]
ok.append("average_epochs_sliding")

# average_erpimage
many = np.random.default_rng(0).standard_normal((20, 2, 8))
out, m = rlw.average_erpimage_data(many, meta, num_lines=8, smooth=False)
assert out.shape == (20, 8, 2, 1), out.shape
assert np.allclose(out[:, 3, :, 0], many[:, :, 3])
out, m = rlw.average_erpimage_data(many, meta, num_lines=5, smooth=True, smooth_width=3)
assert out.shape == (20, 5, 2, 1), out.shape
ok.append("average_erpimage")

# grand_average
a = np.ones((10, 2, 4)); b = np.full((10, 2, 1), 3.0)
out, _ = rlw.grand_average_data([(a, meta), (b, meta)])
assert out.shape == (10, 2, 1) and np.allclose(out, 2.0), out[0, 0]
out, _ = rlw.grand_average_data([(a, meta), (b, meta)], [3, 1])
assert np.allclose(out, 1.5), out[0, 0]
ok.append("grand_average")

# properties
m = rlw.properties_data(meta, filetype="frequency_amplitude", xstart=-2, xstep=0.5)
assert m["filetype"] == "frequency_amplitude" and m["xstart"] == -2 and m["fs"] == 2
assert meta["xstart"] == 0.0, "the original header must not be touched"
ok.append("properties")

print("passed: " + ", ".join(ok)); print("BATCH4 OK")
