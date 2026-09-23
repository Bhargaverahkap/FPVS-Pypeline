# Python ports of the Letswave 6 RLW_* processing functions.
#
# Letswave stores a dataset as a 6-D array (epoch, channel, index, z, y, x) plus
# a header struct. This project stores the same thing as
#
#     npy_data   ndarray, axis 0 = x (time samples or frequency bins)
#                         axis 1 = channels
#                         axis 2 = epochs, when the data is epoched
#     meta_data  dict, the header: fs, xstart, xstep, chanlocs, events, ...
#
# so every function here takes (npy_data, meta_data) and returns
# (npy_data, meta_data), the same shape of call as bandpassfilter and the rest of
# FPyVS_appylication. The unused y and z dimensions are dropped: they were only
# ever 1 in this pipeline. varargin name/value pairs become keyword arguments.
#
# Each function names the MATLAB file it came from. MATLAB counts from 1 and
# python from 0, so anything that took an index in the original takes a 0-based
# index here, and that is called out where it matters.

import numpy as np
from scipy.signal import detrend as _detrend, fftconvolve
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def xstep(meta_data):
    """Spacing of the x axis: seconds per sample, or Hz per bin after an FFT."""
    if meta_data.get("xstep"):
        return float(meta_data["xstep"])
    fs = meta_data.get("fs")
    if fs:
        return 1.0 / float(fs)
    return 1.0


def xstart(meta_data):
    """Value of the x axis at the first sample or bin."""
    return float(meta_data.get("xstart", 0.0) or 0.0)


def xvector(npy_data, meta_data):
    """The x axis itself, one value per row of npy_data."""
    return xstart(meta_data) + np.arange(np.shape(npy_data)[0]) * xstep(meta_data)


def _refresh_shape(npy_data, meta_data):
    """Keep the size bookkeeping in the header honest after a change."""
    meta_data["shape"] = np.shape(npy_data)
    meta_data["size"] = np.size(npy_data)
    return meta_data


def _as3d(npy_data):
    """View the data as (x, channels, epochs) whatever its original rank."""
    data = np.asarray(npy_data)
    if data.ndim == 1:
        return data[:, None, None], data.ndim
    if data.ndim == 2:
        return data[:, :, None], data.ndim
    return data, data.ndim


def _restore(data, ndim):
    """Undo _as3d."""
    if ndim == 1:
        return data[:, 0, 0]
    if ndim == 2:
        return data[:, :, 0]
    return data


# ---------------------------------------------------------------------------
# RLW_crop
# ---------------------------------------------------------------------------

def crop(npy_data, meta_data, x_start=None, x_size=None):
    """Keep a slice of the x axis. Port of RLW_crop.

    x_start is given in x-axis units (seconds, or Hz for a spectrum), not in
    samples, exactly as in Letswave. x_size is a number of samples. Leave either
    out to keep the axis as it is.
    """
    data, ndim = _as3d(npy_data)
    meta_data = dict(meta_data)

    if x_start is None and x_size is None:
        return _restore(data, ndim), meta_data

    start_bin = 0
    if x_start is not None:
        start_bin = int(round((float(x_start) - xstart(meta_data)) / xstep(meta_data)))
        start_bin = max(0, min(start_bin, data.shape[0] - 1))

    end_bin = data.shape[0] if x_size is None else start_bin + int(x_size)
    end_bin = min(end_bin, data.shape[0])

    data = data[start_bin:end_bin]
    meta_data["xstart"] = xstart(meta_data) + start_bin * xstep(meta_data)
    print(f"cropped the x axis to {data.shape[0]} samples starting at "
          f"{meta_data['xstart']}")
    return _restore(data, ndim), _refresh_shape(data, meta_data)


# ---------------------------------------------------------------------------
# RLW_dc_removal
# ---------------------------------------------------------------------------

def dc_removal(npy_data, meta_data, linear_detrend=False):
    """Subtract each signal's own mean. Port of RLW_dc_removal.

    With linear_detrend the straight line through the signal is removed too,
    which is what MATLAB's detrend(...,'linear') does.
    """
    data, ndim = _as3d(npy_data)
    meta_data = dict(meta_data)

    out = data - np.mean(data, axis=0, keepdims=True)
    if linear_detrend:
        print("applying a linear detrend")
        out = _detrend(out, axis=0, type="linear")
    else:
        print("DC removal")

    return _restore(out, ndim), _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_math_constant
# ---------------------------------------------------------------------------

_CONSTANT_OPS = {
    "add": lambda d, c: d + c,
    "subtract": lambda d, c: d - c,
    "multiply": lambda d, c: d * c,
    "divide": lambda d, c: d / c,
}


def math_constant(npy_data, meta_data, operation="add", constant=0):
    """Apply one constant to every sample. Port of RLW_math_constant.

    operation is add, subtract, multiply or divide.
    """
    try:
        func = _CONSTANT_OPS[str(operation).lower()]
    except KeyError:
        raise ValueError(f"operation must be one of {sorted(_CONSTANT_OPS)}, got {operation!r}")

    out = func(np.asarray(npy_data, dtype=float), constant)
    print(f"math operation using a constant: {operation} {constant}")
    return out, _refresh_shape(out, dict(meta_data))


# ---------------------------------------------------------------------------
# RLW_math
# ---------------------------------------------------------------------------

_MATH_OPS = {
    "A+B": lambda a, b: a + b,
    "A-B": lambda a, b: a - b,
    "B-A": lambda a, b: b - a,
    "A*B": lambda a, b: a * b,
    "A/B": lambda a, b: a / b,
    "B/A": lambda a, b: b / a,
}


def math(npy_data_a, meta_data_a, npy_data_b, meta_data_b, operation="A+B",
         selected_epoch=None, selected_channel=None):
    """Combine two datasets sample by sample. Port of RLW_math.

    operation is one of A+B, A-B, B-A, A*B, A/B, B/A. The metadata of A is kept.

    selected_epoch pins every epoch of A against that one epoch of B (0-based).
    selected_channel does the same with a channel, named as a label. Leave them
    out to pair A and B element for element, which is what Letswave does when
    the selectors are 0.
    """
    try:
        func = _MATH_OPS[operation]
    except KeyError:
        raise ValueError(f"operation must be one of {sorted(_MATH_OPS)}, got {operation!r}")

    a, ndim = _as3d(np.asarray(npy_data_a, dtype=float))
    b, _ = _as3d(np.asarray(npy_data_b, dtype=float))

    if selected_channel is not None:
        labels = [str(x) for x in np.atleast_1d(
            np.squeeze(meta_data_b["chanlocs"]["labels"]))]
        matches = [i for i, name in enumerate(labels)
                   if name.lower() == str(selected_channel).lower()]
        if not matches:
            raise ValueError(f"selected channel {selected_channel!r} not found in B")
        print(f"process channel: {selected_channel}")
        b = b[:, matches[0]:matches[0] + 1, :]

    if selected_epoch is not None:
        print(f"process epoch: {selected_epoch}")
        b = b[:, :, int(selected_epoch):int(selected_epoch) + 1]

    out = func(a, b)   # numpy broadcasts the pinned axes back out
    print(f"math operation {operation}")
    return _restore(out, ndim), _refresh_shape(out, dict(meta_data_a))


# ---------------------------------------------------------------------------
# RLW_rectify_signals
# ---------------------------------------------------------------------------

def rectify_signals(npy_data, meta_data, operation="rectify"):
    """Absolute value or square of every sample. Port of RLW_rectify_signals."""
    data = np.asarray(npy_data, dtype=float)
    if str(operation).lower() == "rectify":
        out = np.abs(data)
    elif str(operation).lower() == "square":
        out = data ** 2
    else:
        raise ValueError(f"operation must be rectify or square, got {operation!r}")
    print(f"{operation} signals")
    return out, _refresh_shape(out, dict(meta_data))


# ---------------------------------------------------------------------------
# RLW_derivate_signals
# ---------------------------------------------------------------------------

def derivate_signals(npy_data, meta_data):
    """First difference along x. Port of RLW_derivate_signals.

    The first sample is left as it was, the way the MATLAB loop starts at 2.
    """
    data = np.asarray(npy_data, dtype=float)
    out = data.copy()
    out[1:] = data[1:] - data[:-1]
    print("derivate signals")
    return out, _refresh_shape(out, dict(meta_data))


# ---------------------------------------------------------------------------
# RLW_threshold
# ---------------------------------------------------------------------------

_THRESHOLD_OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
    "==": np.equal,
}


def threshold(npy_data, meta_data, threshold_value=0, threshold_criterion="<",
              consecutivity_criterion=1):
    """Mark the samples that pass a threshold. Port of RLW_threshold.

    Returns ones and zeros, not the data. With consecutivity_criterion n a
    sample only survives when the whole window of n samples either side of it
    also passed, so isolated crossings drop out.
    """
    try:
        compare = _THRESHOLD_OPS[str(threshold_criterion)]
    except KeyError:
        raise ValueError(f"threshold_criterion must be one of "
                         f"{sorted(_THRESHOLD_OPS)}, got {threshold_criterion!r}")

    data = np.asarray(npy_data, dtype=float)
    out = compare(data, threshold_value).astype(float)
    print(f"a total of {int(out.sum())} bins satisfy the threshold criterion")

    n = int(consecutivity_criterion)
    if n > 1:
        print("consecutivity criterion > 1, applying it")
        window = 2 * n + 1
        kept = np.zeros_like(out)
        # A sample passes when every sample in its window passed too.
        for dx in range(n, out.shape[0] - n):
            block = out[dx - n:dx + n + 1]
            kept[dx] = (block.sum(axis=0) >= window).astype(float)
        out = kept
        print(f"after the consecutivity criterion, {int(out.sum())} bins remain")

    return out, _refresh_shape(out, dict(meta_data))


# ---------------------------------------------------------------------------
# RLW_SNR
# ---------------------------------------------------------------------------

def snr(npy_data, meta_data, operation="subtract", xstart_bins=2, xend_bins=5,
        num_extreme=0):
    """Compare each bin against its neighbours. Port of RLW_SNR.

    For every bin the neighbourhood is the bins xstart_bins..xend_bins away on
    both sides, the bin itself and its closest neighbours excluded. operation
    says what to do with the mean of that neighbourhood:

        subtract   bin - baseline
        snr        bin / baseline
        zscore     (bin - baseline) / standard deviation of the neighbourhood
        percent    (bin - baseline) / baseline

    num_extreme drops that many values from each end of the sorted neighbourhood
    before averaging, which is the "extreme bins to remove" box in Letswave.
    """
    operation = str(operation).lower()
    if operation not in ("subtract", "snr", "zscore", "percent"):
        raise ValueError("operation must be subtract, snr, zscore or percent, "
                         f"got {operation!r}")

    data, ndim = _as3d(np.asarray(npy_data, dtype=float))
    nx = data.shape[0]
    dx1, dx2 = int(xstart_bins), int(xend_bins)

    baseline = np.zeros_like(data)
    spread = np.zeros_like(data)

    for dx in range(nx):
        left = np.arange(max(0, dx - dx2), max(0, dx - dx1) + 1)
        right = np.arange(min(nx - 1, dx + dx1), min(nx - 1, dx + dx2) + 1)
        neighbours = np.unique(np.concatenate([left, right]))
        block = data[neighbours]

        if num_extreme > 0 and block.shape[0] > 2 * num_extreme:
            block = np.sort(block, axis=0)[num_extreme:block.shape[0] - num_extreme]

        baseline[dx] = np.mean(block, axis=0)
        if operation == "zscore":
            spread[dx] = np.std(block, axis=0)

    if operation == "subtract":
        out = data - baseline
    elif operation == "snr":
        out = data / baseline
    elif operation == "zscore":
        out = (data - baseline) / spread
    else:
        out = (data - baseline) / baseline

    print(f"SNR ({operation}) over bins {dx1} to {dx2}, "
          f"{num_extreme} extreme bins removed")
    meta_data = dict(meta_data)
    meta_data["snr"] = {"operation": operation, "xstart_bins": dx1,
                        "xend_bins": dx2, "num_extreme": int(num_extreme)}
    return _restore(out, ndim), _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_iFFT
# ---------------------------------------------------------------------------

def ifft(npy_data, meta_data, time_meta_data=None, force_real=True):
    """Inverse FFT back to the time domain. Port of RLW_iFFT.

    Expects the full complex spectrum, the thing fft(..., half_spectrum=False,
    output="complex") produces. Pass the metadata the signal had before the
    forward transform as time_meta_data so xstart, xstep and the events come
    back; without it the x axis is rebuilt from the bin spacing.
    """
    data = np.asarray(npy_data)
    if np.isrealobj(data):
        print("warning: the input is real, so it is probably not a complex spectrum")

    out = np.fft.ifft(data, axis=0)
    if force_real:
        out = np.real(out)

    meta_data = dict(meta_data)
    if time_meta_data is not None:
        for key in ("xstart", "xstep", "events", "fs"):
            if key in time_meta_data:
                meta_data[key] = time_meta_data[key]
    else:
        # bins are 1/(n*dt) apart, so dt falls out of the bin spacing
        bin_width = xstep(meta_data)
        if bin_width:
            meta_data["xstep"] = 1.0 / (bin_width * data.shape[0])
            meta_data["fs"] = 1.0 / meta_data["xstep"]
        meta_data["xstart"] = 0.0

    meta_data["filetype"] = "time_amplitude"
    print("finished computing the inverse FFT")
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_FFT_filter, with ILW_buildFFTbandpass
# ---------------------------------------------------------------------------

def build_fft_bandpass(n_bins, bin_width, low_cutoff, high_cutoff,
                       low_width=0.0, high_width=0.0):
    """The gain applied to each FFT bin. Port of ILW_buildFFTbandpass.

    The vector is symmetric, so it suits a full (not half) spectrum. A non-zero
    width tapers that edge with half a Hann window instead of cutting it square.
    """
    v = np.ones(int(n_bins))
    size = v.size

    if low_cutoff and low_cutoff > 0:
        dx1 = int(round(low_cutoff / bin_width))
        v[:dx1 + 1] = 0
        v[size - dx1:] = 0
        if low_width and low_width > 0:
            width = int(low_width / bin_width)
            if width > dx1:
                print("warning: cutoff width greater than cutoff frequency, adjusting")
                width = dx1
            if width > 0:
                han = np.hanning(width * 2)
                v[dx1 + 1 - width:dx1 + 1] = han[:width]
                v[size - dx1:size - dx1 + width] = han[width:]

    if high_cutoff and high_cutoff > 0:
        dx2 = int(round(high_cutoff / bin_width))
        v[dx2 + 1:size - dx2 + 1] = 0
        if high_width and high_width > 0:
            width = int(high_width / bin_width)
            if width > 0:
                han = np.hanning(width * 2)
                v[dx2 + 1:dx2 + 1 + width] = han[width:]
                v[size - dx2 + 1 - width:size - dx2 + 1] = han[:width]

    return v


def fft_filter(npy_data, meta_data, filter_type="bandpass", low_cutoff=0.5,
               high_cutoff=30, low_width=0.25, high_width=1.0):
    """Filter in the frequency domain. Port of RLW_FFT_filter.

    Forward FFT, multiply the bins by the gain vector, inverse FFT.
    filter_type is bandpass, lowpass, highpass or notch.
    """
    data = np.asarray(npy_data, dtype=float)
    n = data.shape[0]
    bin_width = 1.0 / (n * xstep(meta_data))

    filter_type = str(filter_type).lower()
    if filter_type == "bandpass":
        vector = build_fft_bandpass(n, bin_width, low_cutoff, high_cutoff,
                                    low_width, high_width)
    elif filter_type == "lowpass":
        vector = build_fft_bandpass(n, bin_width, 0, high_cutoff, 0, high_width)
    elif filter_type == "highpass":
        vector = build_fft_bandpass(n, bin_width, low_cutoff, 0, low_width, 0)
    elif filter_type == "notch":
        vector = 1.0 - build_fft_bandpass(n, bin_width, low_cutoff, high_cutoff,
                                          low_width, high_width)
    else:
        raise ValueError("filter_type must be bandpass, lowpass, highpass or "
                         f"notch, got {filter_type!r}")

    print(f"{filter_type} filtering through the FFT")
    spectrum = np.fft.fft(data, axis=0)
    shape = [1] * spectrum.ndim
    shape[0] = n
    spectrum = spectrum * vector.reshape(shape)
    out = np.real(np.fft.ifft(spectrum, axis=0))

    return out, _refresh_shape(out, dict(meta_data))


# ---------------------------------------------------------------------------
# RLW_resample
# ---------------------------------------------------------------------------

def resample(npy_data, meta_data, x_sampling_rate=None, interpolation_method="spline"):
    """Interpolate onto a new sampling rate. Port of RLW_resample (x axis only).

    Unlike downsampling, which throws samples away, this interpolates, so the
    new rate does not have to be an integer division of the old one.
    interpolation_method is spline, linear or nearest.
    """
    if not x_sampling_rate:
        return np.asarray(npy_data), dict(meta_data)

    kind = {"spline": "cubic", "cubic": "cubic",
            "linear": "linear", "nearest": "nearest"}.get(
        str(interpolation_method).lower(), "cubic")

    data, ndim = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    old_x = xvector(data, meta_data)
    new_step = 1.0 / float(x_sampling_rate)
    new_x = np.arange(old_x[0], old_x[-1] + new_step / 2, new_step)
    new_x = new_x[new_x <= old_x[-1]]

    out = interp1d(old_x, data, axis=0, kind=kind)(new_x)

    meta_data["xstep"] = new_step
    meta_data["fs"] = float(x_sampling_rate)
    print(f"resampled to {x_sampling_rate} Hz using {interpolation_method} "
          f"interpolation, {out.shape[0]} samples")
    return _restore(out, ndim), _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# Time-frequency results
#
# A time-frequency transform adds an axis. Letswave puts frequency on its y
# dimension and time on x; dropping the dimensions this pipeline never uses
# leaves (epoch, channel, frequency, time), which in x-first order is
#
#     (time, frequency, channel, epoch)
#
# so axis 0 stays x and the frequency axis slots in right after it. The
# frequency axis itself is written into the header as ystart, ystep and freqs,
# the same three numbers Letswave keeps.
# ---------------------------------------------------------------------------

def _frequency_lines(low_frequency, high_frequency, num_frequency_lines):
    """The frequency axis Letswave builds: num_frequency_lines from the low edge."""
    step = (float(high_frequency) - float(low_frequency)) / int(num_frequency_lines)
    return low_frequency + np.arange(int(num_frequency_lines)) * step, step


def _dft_at(block, frequencies, fs):
    """DFT of a windowed block evaluated at arbitrary frequencies.

    MATLAB's spectrogram(x, window, noverlap, F, fs) does exactly this through
    the Goertzel algorithm; for the handful of lines Letswave asks for, one
    matrix product is simpler and gives the same numbers.
    """
    n = block.shape[0]
    kernel = np.exp(-2j * np.pi * np.outer(np.arange(n), np.asarray(frequencies)) / fs)
    return np.tensordot(block, kernel, axes=([0], [0]))   # -> (..., freq)


def stfft(npy_data, meta_data, hanning_width=0.25, sliding_step=1,
          low_frequency=1, high_frequency=30, num_frequency_lines=100,
          postprocess="amplitude", average_epochs=True):
    """Short-time FFT. Port of RLW_STFFT.

    A Hann window hanning_width seconds long slides sliding_step samples at a
    time; each position is transformed at num_frequency_lines frequencies
    between low_frequency and high_frequency.

    postprocess is amplitude, power, phase or complex. With average_epochs the
    epochs are averaged, and for phase that average is the phase locking value,
    as in Letswave.

    Returns (time, frequency, channel, epoch); see the note above.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step = xstep(meta_data)
    fs = 1.0 / step

    frequencies, freq_step = _frequency_lines(low_frequency, high_frequency,
                                              num_frequency_lines)
    width = int(hanning_width / step)
    if width < 2:
        raise ValueError("hanning_width is shorter than two samples")
    hop = max(1, int(sliding_step))
    window = np.hanning(width)

    starts = np.arange(0, data.shape[0] - width + 1, hop)
    print(f"STFFT: window {width} samples, hop {hop}, {len(starts)} time points, "
          f"{len(frequencies)} frequency lines")

    out = np.empty((len(starts), len(frequencies)) + data.shape[1:], dtype=complex)
    for i, start in enumerate(starts):
        block = data[start:start + width] * window[:, None, None]
        # (freq, channel, epoch) from (..., freq)
        out[i] = np.moveaxis(_dft_at(block, frequencies, fs), -1, 0)

    out = _postprocess_tf(out, postprocess)

    if average_epochs:
        out = _average_tf_epochs(out, postprocess)
        print("averaged the epochs" + (" as a phase locking value"
                                       if str(postprocess).lower() == "phase" else ""))

    meta_data["filetype"] = f"frequency_time_{str(postprocess).lower()}"
    meta_data["xstart"] = xstart(meta_data) + (width / 2) * step
    meta_data["xstep"] = hop * step
    meta_data["ystart"] = float(frequencies[0])
    meta_data["ystep"] = float(freq_step)
    meta_data["freqs"] = frequencies
    return out, _refresh_shape(out, meta_data)


def _postprocess_tf(spectrum, postprocess):
    """amplitude / power / phase / complex, the four Letswave endings."""
    postprocess = str(postprocess).lower()
    if postprocess == "complex":
        return spectrum
    if postprocess == "amplitude":
        return np.abs(spectrum)
    if postprocess == "power":
        return np.abs(spectrum) ** 2
    if postprocess in ("phase", "angle"):
        return np.angle(spectrum)
    raise ValueError("postprocess must be amplitude, power, phase or complex, "
                     f"got {postprocess!r}")


def _average_tf_epochs(out, postprocess):
    """Average over the epoch axis, or the phase locking value for phases."""
    if str(postprocess).lower() in ("phase", "angle"):
        n = out.shape[-1]
        x = np.sum(np.sin(out), axis=-1)
        y = np.sum(np.cos(out), axis=-1)
        return (np.sqrt(x ** 2 + y ** 2) / n)[..., None]
    return np.mean(out, axis=-1, keepdims=True)


def stfft_zhang(npy_data, meta_data, hanning_width=0.25, low_frequency=1,
                high_frequency=30, num_frequency_lines=100,
                postprocess="amplitude", average_epochs=True):
    """Short-time FFT, Zhang's variant. Port of RLW_STFFT_zhang with sub_tfa_stft.

    Differs from stfft in three ways, all from the original helper: the window
    is centred on every sample rather than hopped, the signal is padded
    circularly so the edges keep their length, and the power is normalised by
    the window energy and the sampling rate. Each windowed block has its mean
    removed before the transform.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step = xstep(meta_data)
    fs = 1.0 / step

    frequencies, freq_step = _frequency_lines(low_frequency, high_frequency,
                                              num_frequency_lines)
    half = int(round(hanning_width * fs / 2))
    h = half * 2 + 1
    window = np.hanning(h)
    energy = float(window @ window)      # U, the window power compensation

    padded = np.concatenate([data[-half:], data, data[:half]], axis=0) if half else data
    padded = _detrend(padded, axis=0, type="linear")

    n_times = data.shape[0]
    print(f"STFFT (Zhang): centred window of {h} samples, {n_times} time points, "
          f"{len(frequencies)} frequency lines")

    out = np.empty((n_times, len(frequencies)) + data.shape[1:], dtype=complex)
    for i in range(n_times):
        block = padded[i:i + h]
        block = block - np.mean(block, axis=0, keepdims=True)
        block = block * window[:, None, None]
        out[i] = np.moveaxis(_dft_at(block, frequencies, fs), -1, 0)

    power = (out * np.conj(out)).real / (fs * energy)
    postprocess = str(postprocess).lower()
    if postprocess == "amplitude":
        result = np.sqrt(power)
    elif postprocess == "power":
        result = power
    elif postprocess in ("phase", "angle"):
        result = np.angle(out)
    elif postprocess == "complex":
        result = out
    else:
        raise ValueError("postprocess must be amplitude, power, phase or complex, "
                         f"got {postprocess!r}")

    if average_epochs:
        result = _average_tf_epochs(result, postprocess)

    meta_data["filetype"] = f"frequency_time_{postprocess}"
    meta_data["ystart"] = float(frequencies[0])
    meta_data["ystep"] = float(freq_step)
    meta_data["freqs"] = frequencies
    return result, _refresh_shape(result, meta_data)


# ---------------------------------------------------------------------------
# RLW_CWT and RLW_CWT_fast
# ---------------------------------------------------------------------------

MORLET_CENTRAL_FREQUENCY = 0.8125   # centfrq('morl') in MATLAB


def _morlet(n_points, scale, central_frequency=MORLET_CENTRAL_FREQUENCY):
    """A complex Morlet wavelet on `n_points`, stretched by `scale`."""
    t = (np.arange(n_points) - (n_points - 1) / 2) / scale
    return (np.exp(2j * np.pi * central_frequency * t) *
            np.exp(-t ** 2 / 2) / np.sqrt(scale))


def cwt(npy_data, meta_data, low_frequency=1, high_frequency=30,
        num_frequency_lines=100, output="amplitude", average_epochs=True,
        central_frequency=MORLET_CENTRAL_FREQUENCY):
    """Continuous wavelet transform. Port of RLW_CWT.

    Scales come from the same formula Letswave uses, centfrq / (xstep * f), so
    the frequency axis matches. The wavelet here is the complex Morlet, which
    is what makes an amplitude or a phase meaningful; MATLAB's real 'morl'
    would give a signed coefficient instead.

    output is amplitude, power, phase or complex. Returns
    (time, frequency, channel, epoch).
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step = xstep(meta_data)

    frequencies, freq_step = _frequency_lines(low_frequency, high_frequency,
                                              num_frequency_lines)
    scales = (central_frequency / step) / frequencies
    n = data.shape[0]
    print(f"CWT over {len(frequencies)} frequency lines, "
          f"scales {scales.min():.2f} to {scales.max():.2f}")

    out = np.empty((n, len(frequencies)) + data.shape[1:], dtype=complex)

    for fi, scale in enumerate(scales):
        # A Morlet has died away by four standard deviations, so the wavelet
        # only needs that much support. fftconvolve pads rather than wrapping,
        # which keeps the two ends of the recording out of each other.
        support = int(min(n, max(8, 8 * scale)))
        if support % 2 == 0:
            support += 1
        wavelet = _morlet(support, scale, central_frequency)
        shape = [1] * data.ndim
        shape[0] = support
        out[:, fi] = fftconvolve(data, wavelet.reshape(shape), mode="same", axes=0)

    out = _postprocess_tf(out, output)
    if average_epochs:
        out = _average_tf_epochs(out, output)

    meta_data["filetype"] = f"frequency_time_{str(output).lower()}"
    meta_data["ystart"] = float(frequencies[0])
    meta_data["ystep"] = float(freq_step)
    meta_data["freqs"] = frequencies
    return out, _refresh_shape(out, meta_data)


def cwt_fast(npy_data, meta_data, low_frequency=1, high_frequency=30,
             num_frequency_lines=100, mother_name="morlet", mother_frequency=5,
             mother_spread=0.15, mother_size=8000, output="amplitude",
             average_epochs=True):
    """Continuous wavelet transform, convolution flavour. Port of RLW_CWT_fast.

    Builds one long mother wavelet once, then for each frequency resamples it to
    the length that puts mother_frequency cycles at that frequency and convolves.
    mother_name is morlet (Gaussian envelope, mother_spread wide) or hanning.
    """
    data, ndim = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step = xstep(meta_data)
    srate = 1.0 / step

    frequencies, freq_step = _frequency_lines(low_frequency, high_frequency,
                                              num_frequency_lines)

    size = int(mother_size)
    idx = (np.arange(size) - size / 2) / (size - 1)
    sin_part = np.sin(idx * 2 * np.pi * mother_frequency)
    cos_part = np.cos(idx * 2 * np.pi * mother_frequency)

    if str(mother_name).lower() == "morlet":
        envelope = np.exp(-(idx ** 2) / (2 * float(mother_spread) ** 2))
    elif str(mother_name).lower() == "hanning":
        envelope = 0.5 * (1 - np.cos((2 * np.pi * np.arange(size)) / size))
    else:
        raise ValueError(f"mother_name must be morlet or hanning, got {mother_name!r}")

    mother_sin = sin_part * envelope
    mother_cos = cos_part * envelope
    print(f"CWT (fast) with a {mother_name} mother wavelet over "
          f"{len(frequencies)} frequency lines")

    n = data.shape[0]
    flat = data.reshape(n, -1)
    out = np.empty((n, len(frequencies), flat.shape[1]))

    for fi, freq in enumerate(frequencies):
        spec_size = max(2, int(round((srate * mother_frequency) / freq)))
        positions = np.linspace(0, size - 1, spec_size)
        kernel_sin = np.interp(positions, np.arange(size), mother_sin)
        kernel_cos = np.interp(positions, np.arange(size), mother_cos)
        weight = np.sqrt(freq / mother_frequency)

        for col in range(flat.shape[1]):
            a = np.convolve(flat[:, col], kernel_sin, mode="same") * weight
            b = np.convolve(flat[:, col], kernel_cos, mode="same") * weight
            out[:, fi, col] = np.sqrt(a ** 2 + b ** 2)

    out = out.reshape((n, len(frequencies)) + data.shape[1:])
    if str(output).lower() == "power":
        out = out ** 2
    if average_epochs:
        out = _average_tf_epochs(out, output)

    meta_data["filetype"] = f"frequency_time_{str(output).lower()}"
    meta_data["ystart"] = float(frequencies[0])
    meta_data["ystep"] = float(freq_step)
    meta_data["freqs"] = frequencies
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_hilbert and RLW_hilbert_bands
# ---------------------------------------------------------------------------

def hilbert(npy_data, meta_data):
    """Envelope of the signal. Port of RLW_hilbert.

    The absolute value of the analytic signal, which is the instantaneous
    amplitude.
    """
    from scipy.signal import hilbert as _analytic

    data = np.asarray(npy_data, dtype=float)
    out = np.abs(_analytic(data, axis=0))
    print("Hilbert transform")
    return out, _refresh_shape(out, dict(meta_data))


def hilbert_bands(npy_data, meta_data, freq_start=50, freq_end=300,
                  freq_lines=100, freq_width=5, freq_transition_width=1):
    """Envelope in each of many narrow bands. Port of RLW_hilbert_bands.

    For every band the signal is bandpass filtered through the FFT, its envelope
    taken, and that envelope transformed again, so the result says how strongly
    each band is modulated and at what rate. Returns
    (modulation frequency, band, channel, epoch); the band axis is written into
    the header as ystart / ystep / freqs.

    Slow: it runs one filter and one Hilbert transform per band.
    """
    from scipy.signal import hilbert as _analytic

    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step = xstep(meta_data)
    n = data.shape[0]
    bin_width = 1.0 / (n * step)

    bands = np.linspace(float(freq_start), float(freq_end), int(freq_lines))
    band_step = bands[1] - bands[0] if len(bands) > 1 else 0.0
    print(f"Hilbert transform over {len(bands)} bands, {freq_width} Hz wide. "
          "This can take a while.")

    spectrum = np.fft.fft(data, axis=0)
    shape = [1] * data.ndim
    shape[0] = n

    envelopes = np.empty((n, len(bands)) + data.shape[1:])
    for bi, centre in enumerate(bands):
        vector = build_fft_bandpass(n, bin_width,
                                    centre - freq_width / 2, centre + freq_width / 2,
                                    freq_transition_width, freq_transition_width)
        filtered = np.real(np.fft.ifft(spectrum * vector.reshape(shape), axis=0))
        envelopes[:, bi] = np.abs(_analytic(filtered, axis=0))

    # How fast each envelope moves: magnitude spectrum, half of it, normalised.
    modulation = np.abs(np.fft.fft(envelopes, axis=0)) / n
    modulation = modulation[:n // 2]

    meta_data["xstart"] = 0.0
    meta_data["xstep"] = bin_width
    meta_data["ystart"] = float(bands[0])
    meta_data["ystep"] = float(band_step)
    meta_data["freqs"] = bands
    meta_data["filetype"] = "time_frequency_amplitude"
    meta_data.pop("events", None)
    return modulation, _refresh_shape(modulation, meta_data)
