import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw

fs = 100
def meta_for(labels):
    return {"fs": fs, "xstart": 0.0, "xstep": 1/fs,
            "chanlocs": {"labels": np.array(labels, dtype=object),
                         "X": np.arange(len(labels), dtype=float),
                         "topo_enabled": np.ones(len(labels), dtype=int)},
            "events": {"code": np.array(["go"], dtype=object),
                       "latency": np.array([0.1]), "epoch": np.array([0])}}

meta = meta_for(["C3", "C4", "Cz", "Pz"])
data = np.arange(6*4*2, dtype=float).reshape(6, 4, 2)
ok = []

assert rlw.channel_labels(meta) == ["C3", "C4", "Cz", "Pz"]
assert rlw.find_channels(meta, ["cz", "C3"]) == [2, 0]
m = rlw.select_channels(meta, [3, 1])
assert rlw.channel_labels(m) == ["Pz", "C4"] and list(m["chanlocs"]["X"]) == [3.0, 1.0]
m = rlw.append_channel(meta, "new")
assert rlw.channel_labels(m) == ["C3","C4","Cz","Pz","new"] and len(m["chanlocs"]["X"]) == 5
assert rlw.channel_labels(meta) == ["C3","C4","Cz","Pz"], "original must be untouched"
ok.append("channel helpers")

# merge_channels
other = meta_for(["O1", "O2"])
out, m = rlw.merge_channels([(data, meta), (data[:, :2] * 10, other)])
assert out.shape == (6, 6, 2)
assert rlw.channel_labels(m) == ["C3","C4","Cz","Pz","O1","O2"]
assert len(rlw.events_to_list(m)) == 1, "identical events should fold together"
ok.append("merge_channels")

# pool_channels
out, m = rlw.pool_channels(data, meta, ["C3", "C4"])
assert out.shape == (6, 5, 2)
assert np.allclose(out[:, 4], (data[:, 0] + data[:, 1]) / 2)
assert rlw.channel_labels(m)[-1] == "newchan"
out, m = rlw.pool_channels(data, meta, ["C3","C4"], channel_weights=[3, 1],
                           keep_original_channels=False)
assert out.shape == (6, 1, 2)
assert np.allclose(out[:, 0], (3*data[:, 0] + data[:, 1]) / 4)
ok.append("pool_channels")

# flip_electrodes: C3 takes C4's signal and the other way round
out, _ = rlw.flip_electrodes(data, meta, [("C3", "C4")])
assert np.allclose(out[:, 0], data[:, 1]) and np.allclose(out[:, 1], data[:, 1])
ok.append("flip_electrodes")

# edit_electrodes_info
m = rlw.edit_electrodes_info(meta, [{"labels": "Cz", "X": 99.0}])
assert list(m["chanlocs"]["X"]) == [0.0, 1.0, 99.0, 3.0], list(m["chanlocs"]["X"])
ok.append("edit_electrodes_info")

# rereference_advanced
out, m = rlw.rereference_advanced(data, meta, ["C3", "Pz"], ["Cz", "Cz"])
assert out.shape == (6, 2, 2)
assert np.allclose(out[:, 0], data[:, 0] - data[:, 2])
assert rlw.channel_labels(m) == ["C3-Cz", "Pz-Cz"]
ok.append("rereference_advanced")

# linear_channel_map
out, m = rlw.linear_channel_map(data, meta, num_lines=7)
assert out.shape == (6, 7, 1, 2), out.shape
assert np.allclose(out[:, 0, 0, 0], data[:, 0, 0], atol=1e-8)
assert np.allclose(out[:, -1, 0, 0], data[:, -1, 0], atol=1e-8)
ok.append("linear_channel_map")

# weighted channel average
peak = np.zeros((6, 4, 1))
peak[3, :, 0] = [1.0, 4.0, 2.0, 3.0]
w, labels = rlw.weighted_channel_average_template(peak, meta, x=3/fs, num_channels=2)
assert labels == ["C4", "Pz"], labels
assert np.allclose(w, [4/7, 3/7]), w
out, m = rlw.weighted_channel_average_apply(peak, meta, w, labels)
assert out.shape == (6, 1, 1)
assert np.isclose(out[3, 0, 0], (4*(4/7) + 3*(3/7)) / 1.0), out[3, 0, 0]
assert rlw.channel_labels(m) == ["chanavg"]
ok.append("weighted_channel_average template and apply")

print("passed: " + ", ".join(ok)); print("BATCH5 OK")
