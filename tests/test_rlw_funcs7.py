import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw
ok = []
rng = np.random.default_rng(0)

fs = 250
def meta_for(labels, xyz=None, fs=fs):
    n = len(labels)
    if xyz is None:
        ang = np.linspace(0, np.pi, n)
        xyz = np.stack([np.sin(ang), np.cos(ang), np.full(n, 0.3)], axis=1)
    return {"fs": fs, "xstart": 0.0, "xstep": 1/fs,
            "chanlocs": {"labels": np.array(labels, dtype=object),
                         "X": xyz[:, 0], "Y": xyz[:, 1], "Z": xyz[:, 2],
                         "topo_enabled": np.ones(n, dtype=int),
                         "SEEG_enabled": np.zeros(n, dtype=int)},
            "events": {"code": np.array([], dtype=object),
                       "latency": np.array([]), "epoch": np.array([], dtype=int)}}

# linear_csd: a signal shared by every channel cancels; a local one survives
meta5 = meta_for(["a","b","c","d","e"])
common = np.tile(np.sin(np.arange(40)/5)[:, None], (1, 5))[:, :, None]
out, m = rlw.linear_csd(common, meta5)
assert out.shape == (40, 3, 1) and np.allclose(out, 0, atol=1e-12)
local = common.copy(); local[:, 2, 0] += 10
out, _ = rlw.linear_csd(local, meta5)
assert np.allclose(out[:, 1, 0], 20.0), out[:5, 1, 0]
assert rlw.channel_labels(m) == ["b","c","d"]
ok.append("linear_csd")

# scalp_csd: same shape, a spatially flat map gives ~no CSD, a spike gives some
meta8 = meta_for([f"c{i}" for i in range(8)])
flat = np.ones((20, 8, 1))
out, m = rlw.scalp_csd(flat, meta8)
assert out.shape == (20, 8, 1)
assert np.abs(out).max() < 1e-6, np.abs(out).max()
spiky = np.zeros((20, 8, 1)); spiky[:, 3, 0] = 1.0
out, _ = rlw.scalp_csd(spiky, meta8)
assert np.argmax(np.abs(out[0, :, 0])) == 3, out[0, :, 0]
ok.append("scalp_csd")

# wavelet filter: the mask keeps the requested proportion, and filtering runs
n = 128
t = np.arange(n) / fs
trials = np.stack([np.sin(2*np.pi*10*t) + 0.2*rng.standard_normal(n) for _ in range(4)], axis=1)
wdata = trials[:, None, :]
wmeta = meta_for(["Cz"])
mask, mmeta = rlw.wavelet_filter_build(wdata, wmeta, "Cz", 5, 20, 5, threshold=0.8)
assert mask.shape == (4, n), mask.shape
assert abs(mask.mean() - 0.2) < 0.02, mask.mean()
out, om = rlw.wavelet_filter_apply(wdata, wmeta, mask, mmeta, "Cz")
assert out.shape == (n, 1, 4, 2), out.shape
assert np.allclose(out[:, 0, 1, 1], trials[:, 1]), "the original must be kept as is"
assert np.isfinite(out[:, 0, 0, 0]).all()
ok.append("wavelet_filter_build, wavelet_filter_apply")

# a synthetic ECG: sharp R peaks at 1 Hz on one channel, noise on the others
n = fs * 10
beats = np.arange(0, n, fs)
ecg = np.zeros(n)
for b in beats:
    ecg[b:b+3] = [2.0, 8.0, 2.0]
ecg += 0.05 * rng.standard_normal(n)
noise = 0.5 * rng.standard_normal((n, 2))
d = np.stack([noise[:, 0], ecg, noise[:, 1]], axis=1)[:, :, None]
emeta = meta_for(["EEG1", "EK1", "EEG2"])

peaks = rlw._pan_tompkin_detect(ecg, fs)
assert abs(len(peaks) - len(beats)) <= 1, (len(peaks), len(beats))
# the R peak sits one sample into the pulse; the beat at sample 0 drifts a
# little further because the bandpass has no history to work with there
offsets = np.min(np.abs(peaks[:, None] - beats[None, :]), axis=1)
assert offsets.max() <= 10, offsets
assert np.median(offsets) <= 2, offsets
ok.append("pan_tompkin detector")

out, m = rlw.pan_tompkin(d, emeta, "EK1")
assert out.shape == (n, 4, 1) and rlw.channel_labels(m)[-1] == "HR"
qrs = [e for e in rlw.events_to_list(m) if e["code"] == "QRS"]
assert abs(len(qrs) - len(beats)) <= 1, len(qrs)
hr = out[fs*5, 3, 0]
assert abs(hr - 1.0) < 0.05, hr          # one beat per second
ok.append("pan_tompkin")

_, m = rlw.find_ekg(d, emeta)
found = [e for e in rlw.events_to_list(m) if e["code"] == "EKG"]
assert len(found) >= len(beats) - 3, len(found)
ok.append("find_ekg")

print("passed: " + ", ".join(ok)); print("BATCH7 OK")
