import sys
sys.path.insert(0, "/home/user/FPVS-Pypeline")
import numpy as np, rlw_funcs as rlw

fs, n = 256, 512
t = np.arange(n) / fs
meta = {"fs": fs, "xstart": 0.0, "xstep": 1/fs}
ok = []

# a 10 Hz tone that starts halfway: the transforms must find 10 Hz, late
tone = np.zeros(n); tone[n//2:] = np.sin(2*np.pi*10*t[n//2:])
data = tone[:, None, None]                     # (n, 1 channel, 1 epoch)

out, m = rlw.stfft(data, meta, hanning_width=0.25, sliding_step=8,
                   low_frequency=1, high_frequency=30, num_frequency_lines=30,
                   average_epochs=False)
assert out.ndim == 4, out.shape
freqs = m["freqs"]
power = np.abs(out[:, :, 0, 0])
peak_f = freqs[np.argmax(power.max(axis=0))]
assert abs(peak_f - 10) < 1.5, peak_f
early, late = power[:power.shape[0]//3].max(), power[-power.shape[0]//3:].max()
assert late > 5 * early, (early, late)
ok.append("stfft")

out, m = rlw.stfft_zhang(data, meta, hanning_width=0.25, low_frequency=1,
                         high_frequency=30, num_frequency_lines=30, average_epochs=False)
assert out.shape[0] == n, out.shape
power = np.abs(out[:, :, 0, 0])
assert abs(m["freqs"][np.argmax(power.max(axis=0))] - 10) < 1.5
# the circular padding wraps the tail of the tone into the first half-window,
# so compare against the quiet stretch after it
quiet = power[60:n//2 - 40].max()
assert power[-n//3:].max() > 5 * quiet, (quiet, power[-n//3:].max())
ok.append("stfft_zhang")

out, m = rlw.cwt(data, meta, 1, 30, 30, average_epochs=False)
power = np.abs(out[:, :, 0, 0])
assert abs(m["freqs"][np.argmax(power.max(axis=0))] - 10) < 2
assert power[-n//3:].max() > 3 * power[:n//4].max()
ok.append("cwt")

out, m = rlw.cwt_fast(data, meta, 5, 20, 8, mother_size=800, average_epochs=False)
power = out[:, :, 0, 0]
assert abs(m["freqs"][np.argmax(power.max(axis=0))] - 10) < 3, m["freqs"][np.argmax(power.max(axis=0))]
ok.append("cwt_fast")

# hilbert: envelope of an amplitude-modulated tone follows the modulator
env = 1 + 0.5*np.sin(2*np.pi*2*t)
am = (env * np.sin(2*np.pi*40*t))[:, None]
out, _ = rlw.hilbert(am, meta)
assert np.corrcoef(out[50:-50, 0], env[50:-50])[0, 1] > 0.95
ok.append("hilbert")

out, m = rlw.hilbert_bands(am, meta, freq_start=30, freq_end=50, freq_lines=5, freq_width=8)
assert out.shape == (n//2, 5, 1, 1), out.shape
band_of_40 = np.argmin(abs(m["freqs"] - 40))
mod_axis = np.arange(out.shape[0]) * m["xstep"]
peak_rate = mod_axis[1:][np.argmax(out[1:, band_of_40, 0, 0])]
assert abs(peak_rate - 2) < 1.0, peak_rate      # envelope wobbles at 2 Hz
ok.append("hilbert_bands")

# averaging collapses the epoch axis
two = np.concatenate([data, data * 3], axis=2)
out, _ = rlw.stfft(two, meta, num_frequency_lines=10, average_epochs=True)
assert out.shape[-1] == 1, out.shape
ok.append("epoch averaging")

print("passed: " + ", ".join(ok)); print("BATCH2 OK")
