import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw

fs = 100
def meta_for(labels, events=None):
    m = {"fs": fs, "xstart": 0.0, "xstep": 1/fs,
         "chanlocs": {"labels": np.array(labels, dtype=object),
                      "X": np.zeros(len(labels)), "Y": np.zeros(len(labels)),
                      "Z": np.zeros(len(labels)),
                      "SEEG_enabled": np.zeros(len(labels), dtype=int),
                      "topo_enabled": np.ones(len(labels), dtype=int)}}
    m["events"] = events or {"code": np.array([], dtype=object),
                             "latency": np.array([]), "epoch": np.array([], dtype=int)}
    return m
ok = []
rng = np.random.default_rng(0)

# ICA unmix / remix round trip through a known mixing matrix
meta = meta_for(["A", "B"])
sources = rng.standard_normal((50, 2, 1))
mm = np.array([[2.0, 1.0], [1.0, 3.0]])
um = np.linalg.inv(mm)
mixed = np.einsum("ij,xje->xie", mm, sources)

comps, m, original = rlw.ica_unmix(mixed, meta, um)
assert np.allclose(comps, sources), np.abs(comps - sources).max()
assert rlw.channel_labels(m) == ["IC1", "IC2"]
back, m2 = rlw.ica_remix(comps, m, mm)
assert np.allclose(back, mixed)
assert rlw.channel_labels(m2) == ["A", "B"], "original labels should come back"
# dropping component 1 must remove its contribution
only0, _ = rlw.ica_remix(comps, m, mm, ic_list=[0])
expect = np.einsum("i,xe->xie", mm[:, 0], sources[:, 0, :])
assert np.allclose(only0, expect)
ok.append("ica_unmix, ica_remix")

# PCA: matrices invert each other and decorrelate the channels
corr = rng.standard_normal((200, 1))
data = np.stack([corr[:, 0], corr[:, 0]*0.5 + rng.standard_normal(200)*0.1], axis=1)[:, :, None]
matrix = rlw.pca_compute(data)
assert np.allclose(matrix["ica_mm"] @ matrix["ica_um"], np.eye(2), atol=1e-8)
comps, _, _ = rlw.ica_unmix(data, meta, matrix["ica_um"])
off = np.corrcoef(comps[:, 0, 0], comps[:, 1, 0])[0, 1]
assert abs(off) < 0.2, off
ok.append("pca_compute")

# ocular_remove: a channel that is signal + 0.8*EOG comes back clean
emeta = meta_for(["EEG", "EOG"])
eog = rng.standard_normal((300, 1))
clean = np.sin(np.arange(300)/10)[:, None]
d = np.stack([(clean + 0.8*eog)[:, 0], eog[:, 0]], axis=1)[:, :, None]
out, _ = rlw.ocular_remove(d, emeta, ["EOG"])
before = abs(np.corrcoef(d[:, 0, 0], eog[:, 0])[0, 1])
after = abs(np.corrcoef(out[:, 0, 0], eog[:, 0])[0, 1])
assert after < 1e-10 < before, (before, after)     # the eye signal is gone
assert np.corrcoef(out[:, 0, 0], clean[:, 0])[0, 1] > 0.99
ok.append("ocular_remove")

# suppress_artifact: a spike between two flat stretches is replaced by a line
spike = np.ones((20, 1, 1)); spike[9:12] = 50
out, _ = rlw.suppress_artifact(spike, meta_for(["A"]), x_start=8/fs, x_end=13/fs)
assert np.allclose(out[8:14, 0, 0], 1.0), out[8:14, 0, 0]
ok.append("suppress_artifact")

# suppress_artifact_event: same, but located by an event
ev = {"code": np.array(["stim"], dtype=object), "latency": np.array([0.10]),
      "epoch": np.array([0])}
ramp = np.arange(30, dtype=float)[:, None, None]
dirty = ramp.copy(); dirty[9:12] = 500
out, _ = rlw.suppress_artifact_event(dirty, meta_for(["A"], ev), "stim",
                                     x_start=-0.01, x_end=0.01, interp_method="linear")
assert np.allclose(out[:, 0, 0], ramp[:, 0, 0], atol=1e-6), out[8:13, 0, 0]
ok.append("suppress_artifact_event")

# segmentation_ssep: 2 triggers, 4 cycles at 2 Hz -> epochs of 200 samples
long = np.arange(1000, dtype=float)[:, None, None]
sev = {"code": np.array(["10", "10"], dtype=object), "latency": np.array([1.0, 4.0]),
       "epoch": np.array([0, 0])}
out, m = rlw.segmentation_ssep(long, meta_for(["A"], sev), ["10"],
                               cycle_skip=0, cycle_total=4, cycle_frequency=2.0)
assert out.shape == (200, 1, 2), out.shape
assert out[0, 0, 0] == 100 and out[0, 0, 1] == 400
out, m = rlw.segmentation_ssep(long, meta_for(["A"], sev), ["10"],
                               cycle_skip=2, cycle_total=4, cycle_frequency=2.0)
assert out.shape == (100, 1, 2) and out[0, 0, 0] == 200, (out.shape, out[0, 0, 0])
assert m["xstart"] == 1.0
ok.append("segmentation_ssep")

# edit_electrodes_seeg
m = rlw.edit_electrodes_seeg(meta_for(["A", "B"]), ["B"], [1], [2], [3])
assert list(m["chanlocs"]["X"]) == [0.0, 1.0] and list(m["chanlocs"]["Z"]) == [0.0, 3.0]
assert list(m["chanlocs"]["SEEG_enabled"]) == [0, 1]
assert list(m["chanlocs"]["topo_enabled"]) == [1, 0]
ok.append("edit_electrodes_seeg")

print("passed: " + ", ".join(ok)); print("BATCH6 OK")
