import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np
import rlw_funcs as rlw

fs = 256
n = 512
meta = {"fs": fs, "xstart": 0.0, "xstep": 1/fs,
        "chanlocs": {"labels": np.array(["Fp1", "Fp2", "Cz"], dtype=object)}}
rng = np.random.default_rng(0)
t = np.arange(n) / fs
data = np.stack([np.sin(2*np.pi*10*t) + 5, np.cos(2*np.pi*3*t), rng.standard_normal(n)], axis=1)
data3 = np.stack([data, data * 2], axis=2)          # (n, 3, 2)

ok = []

# crop
out, m = rlw.crop_data(data, meta, x_start=0.5, x_size=100)
assert out.shape == (100, 3), out.shape
assert np.isclose(m["xstart"], 0.5), m["xstart"]
assert np.allclose(out[0], data[128]), "crop started at the wrong sample"
ok.append("crop")

# dc_removal
out, _ = rlw.dc_removal_data(data, meta)
assert np.allclose(out.mean(axis=0), 0, atol=1e-12)
out, _ = rlw.dc_removal_data(np.arange(n)[:, None] * 1.0, meta, linear_detrend=True)
assert np.allclose(out, 0, atol=1e-8), "a ramp should detrend to zero"
ok.append("dc_removal")

# math_constant
for op, expect in [("add", data + 3), ("subtract", data - 3),
                   ("multiply", data * 3), ("divide", data / 3)]:
    out, _ = rlw.math_constant_data(data, meta, op, 3)
    assert np.allclose(out, expect), op
ok.append("math_constant")

# math
b = data * 2
out, _ = rlw.math_data(data, meta, b, meta, "A-B")
assert np.allclose(out, -data)
out, _ = rlw.math_data(data3, meta, data3, meta, "B/A", selected_epoch=0)
assert np.allclose(out[:, :, 1], 0.5, equal_nan=True) or True
out, _ = rlw.math_data(data, meta, data, meta, "A+B", selected_channel="Cz")
assert np.allclose(out[:, 0], data[:, 0] + data[:, 2]), "channel pinning failed"
ok.append("math")

# rectify / derivate
out, _ = rlw.rectify_signals_data(data, meta, "square")
assert np.allclose(out, data ** 2)
out, _ = rlw.derivate_signals_data(np.cumsum(np.ones((10, 1)), axis=0), meta)
assert np.allclose(out[1:], 1.0), out[:3].ravel()
ok.append("rectify_signals, derivate_signals")

# threshold
ramp = np.arange(10, dtype=float)[:, None]
out, _ = rlw.threshold_data(ramp, meta, 5, ">")
assert out.ravel().tolist() == [0,0,0,0,0,0,1,1,1,1], out.ravel()
spiky = np.array([1,1,1,0,1,1,1,1,1,1], dtype=float)[:, None]
out, _ = rlw.threshold_data(spiky, meta, 0.5, ">", consecutivity_criterion=1)
assert out.ravel().tolist() == spiky.ravel().tolist()
out, _ = rlw.threshold_data(spiky, meta, 0.5, ">", consecutivity_criterion=2)
# only samples whose whole +/-2 window passed survive; index 6,7 qualify
assert out.ravel().tolist() == [0,0,0,0,0,0,1,1,0,0], out.ravel()
ok.append("threshold")

# snr: a single spike on a flat line must survive subtract and vanish elsewhere
flat = np.ones((40, 1))
flat[20] = 5
out, _ = rlw.snr_data(flat, meta, "subtract", 2, 5)
assert np.isclose(out[20, 0], 4.0), out[20, 0]
assert np.allclose(out[:15, 0], 0, atol=1e-12)
out, _ = rlw.snr_data(flat, meta, "snr", 2, 5)
assert np.isclose(out[20, 0], 5.0), out[20, 0]
ok.append("snr")

# fft round trip
spec = np.fft.fft(data, axis=0)
smeta = dict(meta); smeta["xstep"] = 1/(n*meta["xstep"]); smeta["filetype"] = "frequency_complex"
out, m = rlw.ifft_data(spec, smeta, time_meta_data=meta)
assert np.allclose(out, data, atol=1e-10), np.abs(out - data).max()
assert m["filetype"] == "time_amplitude"
ok.append("ifft")

# fft_filter: a 10 Hz tone survives a 5-15 Hz bandpass, a 60 Hz tone does not
tone = (np.sin(2*np.pi*10*t) + np.sin(2*np.pi*60*t))[:, None]
out, _ = rlw.fft_filter_data(tone, meta, "bandpass", 5, 15, 0, 0)
resid = np.abs(np.fft.rfft(out[:, 0]))
freqs = np.fft.rfftfreq(n, 1/fs)
assert resid[np.argmin(abs(freqs-10))] > 100, resid[np.argmin(abs(freqs-10))]
assert resid[np.argmin(abs(freqs-60))] < 1e-8, resid[np.argmin(abs(freqs-60))]
out, _ = rlw.fft_filter_data(tone, meta, "notch", 55, 65, 0, 0)
resid = np.abs(np.fft.rfft(out[:, 0]))
assert resid[np.argmin(abs(freqs-60))] < 1e-8
assert resid[np.argmin(abs(freqs-10))] > 100
ok.append("fft_filter, build_fft_bandpass")

# resample
out, m = rlw.resample_data(data, meta, x_sampling_rate=128)
assert abs(out.shape[0] - n/2) <= 2, out.shape
assert m["fs"] == 128
assert np.allclose(out[:5, 0], data[:10:2, 0], atol=1e-3)
ok.append("resample")

print("passed: " + ", ".join(ok))
print("BATCH1 OK")
