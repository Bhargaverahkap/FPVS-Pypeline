import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw

fs = 256
meta = {"fs": fs, "xstart": 0.0, "xstep": 1/fs,
        "chanlocs": {"labels": np.array(["Fp1","Fp2","Cz"], dtype=object)},
        "events": {"code": np.array(["10","20","30"], dtype=object),
                   "latency": np.array([0.1, 0.2, 0.3]),
                   "epoch": np.array([0, 1, 2])}}
data = np.arange(4*3*3, dtype=float).reshape(4, 3, 3)   # (x=4, ch=3, epochs=3)
ok = []

# events round trip
evs = rlw.events_to_list(meta)
assert len(evs) == 3 and evs[1]["code"] == "20" and evs[1]["epoch"] == 1
ok.append("events helpers")

# reject_epochs: epoch 1 goes, its event goes, epoch 2 renumbers to 1
out, m = rlw.reject_epochs(data, meta, [1])
assert out.shape == (4, 3, 2)
assert np.allclose(out[:, :, 1], data[:, :, 2])
evs = rlw.events_to_list(m)
assert [e["code"] for e in evs] == ["10", "30"], evs
assert [e["epoch"] for e in evs] == [0, 1], evs
ok.append("reject_epochs")

# reject_epochs_amplitude: blow up epoch 0 only
loud = data.copy(); loud[:, :, 0] += 1000
out, m = rlw.reject_epochs_amplitude(loud, meta, criterion=500)
assert out.shape[2] == 2 and np.allclose(out[:, :, 0], data[:, :, 1])
# channel restriction: only Cz is loud -> still rejected; restrict to Fp1 -> kept
partly = data.copy(); partly[:, 2, 0] += 1000
out, _ = rlw.reject_epochs_amplitude(partly, meta, 500, selected_channel_labels=["Fp1"])
assert out.shape[2] == 3, out.shape
out, _ = rlw.reject_epochs_amplitude(partly, meta, 500, selected_channel_labels=["Cz"])
assert out.shape[2] == 2, out.shape
# x window: spike at sample 0 only, window starting later keeps the epoch
spike = data.copy(); spike[0, :, 0] += 1000
out, _ = rlw.reject_epochs_amplitude(spike, meta, 500, x_limits=True, x_start=1/fs)
assert out.shape[2] == 3, out.shape
ok.append("reject_epochs_amplitude")

# concatenate_epochs: 3 epochs of 4 samples -> one epoch of 12, latencies shifted
out, m = rlw.concatenate_epochs(data, meta)
assert out.shape == (12, 3, 1), out.shape
assert np.allclose(out[4:8, :, 0], data[:, :, 1])
evs = rlw.events_to_list(m)
dur = 4 / fs
assert np.allclose([e["latency"] for e in evs], [0.1, 0.2 + dur, 0.3 + 2*dur]), evs
assert all(e["epoch"] == 0 for e in evs)
ok.append("concatenate_epochs")

# equalize_epochs
big = np.zeros((4, 3, 5)); small = np.zeros((4, 3, 2))
out = rlw.equalize_epochs([(big, meta), (small, meta)])
assert [d.shape[2] for d, _ in out] == [2, 2]
out = rlw.equalize_epochs([(big, meta)], num_epochs=3, random_selection=True, seed=1)
assert out[0][0].shape[2] == 3
ok.append("equalize_epochs")

# index axis
four = np.zeros((4, 3, 3, 5))
out, m = rlw.arrange_index(four, dict(meta, index_labels=[f"i{i}" for i in range(5)]), [4, 0])
assert out.shape == (4, 3, 3, 2) and m["index_labels"] == ["i4", "i0"]
out, _ = rlw.arrange_index(data, meta, [0])         # 3-D: no-op
assert out.shape == data.shape
merged, m = rlw.merge_index([(data, meta), (data, meta)])
assert merged.shape == (4, 3, 3, 2), merged.shape
assert len(rlw.events_to_list(m)) == 3, "duplicate events should have been dropped"
ok.append("arrange_index, merge_index")

# epochdata select / sort
emeta = dict(meta)
emeta["epochdata"] = [{"data": {"rt": 0.5}}, {"data": {"rt": 0.1}}, {"data": {}}]
out, m = rlw.select_epochdata(data, emeta, "rt", "<", 0.3)
assert out.shape[2] == 1 and np.allclose(out[:, :, 0], data[:, :, 1])
out, m = rlw.sort_epochdata(data, emeta, "rt")
assert out.shape[2] == 2
assert np.allclose(out[:, :, 0], data[:, :, 1]), "0.1 should sort first"
out, _ = rlw.sort_epochdata(data, emeta, "rt", "descend", discard_empty=False)
assert out.shape[2] == 3 and np.allclose(out[:, :, 0], data[:, :, 0])
ok.append("select_epochdata, sort_epochdata")

print("passed: " + ", ".join(ok)); print("BATCH3 OK")
