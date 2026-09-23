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


# ---------------------------------------------------------------------------
# events
#
# Letswave keeps events as a struct array, one struct per event with code,
# latency and epoch. This project keeps the same three fields as parallel lists
# inside meta_data["events"]. These helpers move between the two shapes so the
# ports below can think one event at a time the way the MATLAB does.
# ---------------------------------------------------------------------------

def events_to_list(meta_data):
    """meta_data["events"] as a list of {code, latency, epoch} dicts."""
    events = meta_data.get("events")
    if not events:
        return []
    codes = np.atleast_1d(np.asarray(events.get("code", []), dtype=object))
    latencies = np.atleast_1d(np.asarray(events.get("latency", []), dtype=object))
    epochs = np.atleast_1d(np.asarray(events.get("epoch", []), dtype=object))
    out = []
    for i in range(len(codes)):
        out.append({
            "code": codes[i],
            "latency": float(latencies[i]) if i < len(latencies) else 0.0,
            "epoch": int(epochs[i]) if i < len(epochs) and epochs[i] is not None else 0,
        })
    return out


def events_from_list(meta_data, events):
    """Write a list of event dicts back into meta_data, in place."""
    meta_data["events"] = {
        "code": np.array([e["code"] for e in events], dtype=object),
        "latency": np.array([e["latency"] for e in events], dtype=float),
        "epoch": np.array([e["epoch"] for e in events], dtype=int),
    }
    return meta_data


def _remap_epochs(meta_data, kept_epochs):
    """Keep the events of the epochs that survived and renumber them.

    kept_epochs is the list of original epoch numbers in their new order, so
    an event on old epoch kept_epochs[i] ends up on epoch i.
    """
    events = events_to_list(meta_data)
    if not events:
        return meta_data
    new_position = {old: new for new, old in enumerate(kept_epochs)}
    kept = []
    for event in events:
        if event["epoch"] in new_position:
            event = dict(event)
            event["epoch"] = new_position[event["epoch"]]
            kept.append(event)
    return events_from_list(meta_data, kept)


# ---------------------------------------------------------------------------
# RLW_arrange_index and RLW_merge_index
#
# Letswave's third dimension, "index", holds things like the separate
# components of a decomposition. This pipeline never fills it, so the data here
# is 3-D and these two functions work on a fourth axis when one exists.
# ---------------------------------------------------------------------------

def arrange_index(npy_data, meta_data, index_idx):
    """Keep, reorder or drop indexes. Port of RLW_arrange_index.

    index_idx is 0-based. With 3-D data there is no index axis and nothing
    happens.
    """
    data = np.asarray(npy_data)
    meta_data = dict(meta_data)
    if data.ndim < 4:
        print("no index axis in this dataset, nothing to arrange")
        return data, meta_data

    index_idx = np.atleast_1d(np.asarray(index_idx, dtype=int))
    out = np.take(data, index_idx, axis=3)
    labels = meta_data.get("index_labels")
    if labels is not None:
        meta_data["index_labels"] = [labels[i] for i in index_idx]
    print(f"number of indexes: {len(index_idx)}")
    return out, _refresh_shape(out, meta_data)


def merge_index(datasets):
    """Stack several datasets along the index axis. Port of RLW_merge_index.

    datasets is a list of (npy_data, meta_data) pairs, all the same shape. The
    header of the first one is kept, the index labels and events of all of them
    are concatenated, and duplicate events are dropped the way Letswave does.
    """
    if not datasets:
        raise ValueError("no datasets to merge")

    first_data, first_meta = datasets[0]
    out = np.asarray(first_data)
    if out.ndim < 4:
        out = out[..., None]
    meta_data = dict(first_meta)

    labels = list(meta_data.get("index_labels")
                  or [f"index {i}" for i in range(out.shape[3])])
    events = events_to_list(meta_data)

    for data, meta in datasets[1:]:
        data = np.asarray(data)
        if data.ndim < 4:
            data = data[..., None]
        if data.shape[:3] != out.shape[:3]:
            raise ValueError("datasets cannot be merged as their sizes do not match: "
                             f"{out.shape[:3]} against {data.shape[:3]}")
        out = np.concatenate([out, data], axis=3)
        labels += list(meta.get("index_labels")
                       or [f"index {i}" for i in range(data.shape[3])])
        events += events_to_list(meta)

    # drop events that are identical in all three fields
    seen, unique = set(), []
    for event in events:
        key = (str(event["code"]), event["latency"], event["epoch"])
        if key not in seen:
            seen.add(key)
            unique.append(event)
    if len(unique) != len(events):
        print(f"deleted {len(events) - len(unique)} duplicate events")

    meta_data["index_labels"] = labels
    meta_data["history"] = {}
    meta_data = events_from_list(meta_data, unique)
    print(f"merged {len(datasets)} datasets into {out.shape[3]} indexes")
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_concatenate_epochs
# ---------------------------------------------------------------------------

def concatenate_epochs(npy_data, meta_data, epoch_idx=None):
    """Lay the chosen epochs end to end into one long epoch.

    Port of RLW_concatenate_epochs. epoch_idx is 0-based; leave it out to take
    every epoch. Event latencies are shifted by the duration of the epochs in
    front of them, as in Letswave.
    """
    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)

    if epoch_idx is None:
        epoch_idx = np.arange(data.shape[2])
    epoch_idx = np.atleast_1d(np.asarray(epoch_idx, dtype=int))
    print(f"concatenating {len(epoch_idx)} epochs")

    picked = data[:, :, epoch_idx]
    out = np.concatenate([picked[:, :, i] for i in range(picked.shape[2])], axis=0)
    out = out[:, :, None]

    duration = data.shape[0] * xstep(meta_data)
    events, kept = events_to_list(meta_data), []
    for position, old_epoch in enumerate(epoch_idx):
        for event in events:
            if event["epoch"] == old_epoch:
                event = dict(event)
                event["latency"] = event["latency"] + duration * position
                event["epoch"] = 0
                kept.append(event)
    meta_data = events_from_list(meta_data, kept)

    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_equalize_epochs
# ---------------------------------------------------------------------------

def equalize_epochs(datasets, num_epochs=None, random_selection=False, seed=None):
    """Cut every dataset down to the same number of epochs.

    Port of RLW_equalize_epochs. datasets is a list of (npy_data, meta_data)
    pairs. With num_epochs left out the smallest dataset sets the count. With
    random_selection the epochs are drawn at random rather than taken from the
    front; pass a seed to make that draw repeatable.
    """
    prepared = [(_as3d(np.asarray(d))[0], dict(m)) for d, m in datasets]

    if not num_epochs:
        num_epochs = min(d.shape[2] for d, _ in prepared)
    num_epochs = int(num_epochs)
    print(f"equalizing to {num_epochs} epochs")

    rng = np.random.default_rng(seed)
    out = []
    for data, meta in prepared:
        if random_selection:
            epoch_idx = np.sort(rng.permutation(data.shape[2])[:num_epochs])
        else:
            epoch_idx = np.arange(min(num_epochs, data.shape[2]))

        picked = data[:, :, epoch_idx]
        meta = _remap_epochs(meta, list(epoch_idx))
        if meta.get("epochdata") is not None:
            meta["epochdata"] = [meta["epochdata"][i] for i in epoch_idx]
        out.append((picked, _refresh_shape(picked, meta)))

    return out


# ---------------------------------------------------------------------------
# RLW_reject_epochs and RLW_reject_epochs_amplitude
# ---------------------------------------------------------------------------

def reject_epochs(npy_data, meta_data, rejected_epochs):
    """Throw away the named epochs. Port of RLW_reject_epochs.

    rejected_epochs is 0-based. Events belonging to a rejected epoch go with it.
    """
    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)

    rejected = set(int(i) for i in np.atleast_1d(rejected_epochs))
    accepted = [i for i in range(data.shape[2]) if i not in rejected]
    print(f"rejecting {len(rejected)} epochs, {len(accepted)} left")

    out = data[:, :, accepted]
    meta_data = _remap_epochs(meta_data, accepted)
    if meta_data.get("epochdata") is not None:
        meta_data["epochdata"] = [meta_data["epochdata"][i] for i in accepted]
    return out, _refresh_shape(out, meta_data)


def reject_epochs_amplitude(npy_data, meta_data, criterion=100, x_limits=False,
                            x_start=None, x_end=None, selected_channel_labels=None):
    """Throw away the epochs that swing too far. Port of RLW_reject_epochs_amplitude.

    An epoch goes when any sample in the window exceeds criterion in absolute
    value. With x_limits the window is x_start..x_end in x-axis units rather
    than the whole epoch, and selected_channel_labels narrows the test to those
    channels.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    if x_limits:
        step, origin = xstep(meta_data), xstart(meta_data)
        dx1 = 0 if x_start is None else max(0, int(round((x_start - origin) / step)))
        dx2 = data.shape[0] if x_end is None else min(
            data.shape[0], int(round((x_end - origin) / step)) + 1)
    else:
        dx1, dx2 = 0, data.shape[0]

    if selected_channel_labels:
        labels = [str(x) for x in np.atleast_1d(
            np.squeeze(meta_data["chanlocs"]["labels"]))]
        wanted = [str(x).lower() for x in np.atleast_1d(selected_channel_labels)]
        channels = [i for i, name in enumerate(labels) if name.lower() in wanted]
        if not channels:
            raise ValueError("none of the selected channel labels were found")
    else:
        channels = list(range(data.shape[1]))

    print(f"amplitude criterion {criterion}, samples {dx1} to {dx2}, "
          f"{len(channels)} channels")

    block = data[dx1:dx2][:, channels]
    peak = np.max(np.abs(block), axis=(0, 1))
    accepted = [i for i in range(data.shape[2]) if peak[i] <= criterion]
    print(f"accepted epochs: {accepted}")

    out = data[:, :, accepted]
    meta_data = _remap_epochs(meta_data, accepted)
    if meta_data.get("epochdata") is not None:
        meta_data["epochdata"] = [meta_data["epochdata"][i] for i in accepted]
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_select_epochdata and RLW_sort_epochdata
#
# Letswave hangs a record on each epoch - reaction time, response, whatever the
# experiment logged - in header.epochdata. Here that is meta_data["epochdata"],
# a list with one dict per epoch.
# ---------------------------------------------------------------------------

_COMPARISONS = {
    "==": lambda a, b: a == b, "=": lambda a, b: a == b,
    "~=": lambda a, b: a != b, "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
}


def _epochdata_values(meta_data, fieldname, n_epochs):
    """The value of one epochdata field per epoch, None where it is missing."""
    records = meta_data.get("epochdata")
    if not records:
        raise ValueError("no epoch data available")
    values = []
    for i in range(n_epochs):
        record = records[i] if i < len(records) else {}
        value = record.get("data", record).get(fieldname) if isinstance(record, dict) else None
        values.append(value)
    return values


def select_epochdata(npy_data, meta_data, fieldname, logical="==", comparison_value=0):
    """Keep the epochs whose epochdata field passes a test.

    Port of RLW_select_epochdata. logical is ==, ~=, >, <, >= or <=. Epochs
    with no value, or a value that is not a number, are dropped, as in Letswave.
    """
    try:
        compare = _COMPARISONS[str(logical)]
    except KeyError:
        raise ValueError(f"logical must be one of {sorted(_COMPARISONS)}, got {logical!r}")

    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)

    values = _epochdata_values(meta_data, fieldname, data.shape[2])
    accepted = [i for i, value in enumerate(values)
                if isinstance(value, (int, float, np.number))
                and compare(value, comparison_value)]
    print(f"found {len(accepted)} epochs meeting the criterion")

    out = data[:, :, accepted]
    meta_data = _remap_epochs(meta_data, accepted)
    meta_data["epochdata"] = [meta_data["epochdata"][i] for i in accepted]
    return out, _refresh_shape(out, meta_data)


def sort_epochdata(npy_data, meta_data, fieldname, sort_direction="ascend",
                   discard_empty=True):
    """Reorder the epochs by an epochdata field. Port of RLW_sort_epochdata.

    Epochs with no usable value are dropped, or appended at the end when
    discard_empty is False.
    """
    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)

    values = _epochdata_values(meta_data, fieldname, data.shape[2])
    sortable = [(i, v) for i, v in enumerate(values)
                if isinstance(v, (int, float, np.number))]
    missing = [i for i, v in enumerate(values)
               if not isinstance(v, (int, float, np.number))]
    print(f"found {len(sortable)} epochs to sort")

    sortable.sort(key=lambda pair: pair[1],
                  reverse=str(sort_direction).lower().startswith("desc"))
    order = [i for i, _ in sortable]
    if missing and not discard_empty:
        print(f"appending discarded epochs: {missing}")
        order += missing
    print(f"sort order: {order}")

    out = data[:, :, order]
    meta_data = _remap_epochs(meta_data, order)
    meta_data["epochdata"] = [meta_data["epochdata"][i] for i in order]
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_select_events and RLW_sort_events
# ---------------------------------------------------------------------------

def select_events(npy_data, meta_data, event_code, minimum_latency=0.0,
                  maximum_latency=1.0, check_minimum_latency=True,
                  check_maximum_latency=True):
    """Keep the epochs that carry a given event. Port of RLW_select_events.

    An epoch survives when it holds at least one event with that code whose
    latency falls inside the window. Either end of the window can be switched
    off with check_minimum_latency / check_maximum_latency.
    """
    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)

    events = events_to_list(meta_data)
    if not events:
        print("no events available, nothing selected")
        return data, meta_data

    matching = [e for e in events
                if str(e["code"]).lower() == str(event_code).lower()]
    if not matching:
        print(f"no events with code {event_code!r}, nothing selected")
        return data, meta_data

    accepted = []
    for epoch in range(data.shape[2]):
        for event in matching:
            if event["epoch"] != epoch:
                continue
            if check_minimum_latency and event["latency"] < minimum_latency:
                continue
            if check_maximum_latency and event["latency"] > maximum_latency:
                continue
            accepted.append(epoch)
            break
    print(f"found {len(accepted)} epochs meeting the criterion")

    out = data[:, :, accepted]
    meta_data = _remap_epochs(meta_data, accepted)
    if meta_data.get("epochdata") is not None:
        meta_data["epochdata"] = [meta_data["epochdata"][i] for i in accepted]
    return out, _refresh_shape(out, meta_data)


def sort_events(npy_data, meta_data, event_code, sort_direction="ascend",
                discard_empty=True):
    """Reorder the epochs by when an event happened. Port of RLW_sort_events.

    Each epoch is keyed on the latency of its first event with that code.
    Epochs without one are dropped, or appended at the end when discard_empty
    is False.
    """
    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)

    events = events_to_list(meta_data)
    matching = [e for e in events
                if str(e["code"]).lower() == str(event_code).lower()]
    if not matching:
        print(f"no events with code {event_code!r}, nothing sorted")
        return data, meta_data

    first_latency = {}
    for event in matching:
        first_latency.setdefault(event["epoch"], event["latency"])

    keyed = sorted(first_latency.items(), key=lambda pair: pair[1],
                   reverse=str(sort_direction).lower().startswith("desc"))
    order = [epoch for epoch, _ in keyed]
    missing = [i for i in range(data.shape[2]) if i not in first_latency]
    if missing and not discard_empty:
        print(f"appending discarded epochs: {missing}")
        order += missing
    print(f"sort order: {order}")

    out = data[:, :, order]
    meta_data = _remap_epochs(meta_data, order)
    if meta_data.get("epochdata") is not None:
        meta_data["epochdata"] = [meta_data["epochdata"][i] for i in order]
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_events_delete_duplicate
# ---------------------------------------------------------------------------

def events_delete_duplicate(meta_data, exact_latencies=True, tolerance=0.1,
                            verbose=False):
    """Drop repeated triggers. Port of RLW_events_delete_duplicate.

    Two events are the same when they share a code and an epoch and their
    latencies match: exactly, or within tolerance seconds when exact_latencies
    is False. The earliest of each group is the one kept. Returns the metadata
    only, as the data is untouched.
    """
    meta_data = dict(meta_data)
    events = events_to_list(meta_data)
    if len(events) < 2:
        return meta_data

    print("finding events with identical latencies" if exact_latencies else
          f"finding events with similar latencies, tolerance {tolerance}")

    order = sorted(range(len(events)), key=lambda i: events[i]["latency"])
    dropped = set()
    for position, i in enumerate(order):
        if i in dropped:
            continue
        for j in order[position + 1:]:
            same_key = (str(events[j]["code"]).lower() == str(events[i]["code"]).lower()
                        and events[j]["epoch"] == events[i]["epoch"])
            if not same_key:
                continue
            gap = abs(events[i]["latency"] - events[j]["latency"])
            if (gap == 0) if exact_latencies else (gap < tolerance):
                dropped.add(j)
                if verbose:
                    print(f"event {i} = event {j}")

    print(f"found {len(dropped)} duplicate events")
    kept = [e for i, e in enumerate(events) if i not in dropped]
    return events_from_list(meta_data, kept)


# ---------------------------------------------------------------------------
# RLW_events_level_trigger
# ---------------------------------------------------------------------------

def events_level_trigger(npy_data, meta_data, selected_channel, threshold=1000,
                         min_isi=1.0, direction="ascending", event_code="trig"):
    """Read triggers off a channel that crosses a level.

    Port of RLW_events_level_trigger. Every sample past the threshold becomes a
    candidate; candidates closer together than min_isi seconds are dropped, so
    one crossing gives one event. The events are added to the metadata and the
    data is returned untouched.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    labels = [str(x) for x in np.atleast_1d(
        np.squeeze(meta_data["chanlocs"]["labels"]))]
    matches = [i for i, name in enumerate(labels)
               if name.lower() == str(selected_channel).lower()]
    if not matches:
        raise ValueError(f"channel label {selected_channel!r} not found")
    chanpos = matches[0]
    print(f"selected channel position: {chanpos}")

    times = xvector(data, meta_data)
    events = events_to_list(meta_data)

    for epoch in range(data.shape[2]):
        trace = data[:, chanpos, epoch]
        if str(direction).lower() == "ascending":
            crossings = np.where(trace > threshold)[0]
        elif str(direction).lower() == "descending":
            crossings = np.where(trace < threshold)[0]
        else:
            raise ValueError("direction must be ascending or descending, "
                             f"got {direction!r}")
        if crossings.size == 0:
            print(f"no triggers found in epoch {epoch}")
            continue

        latencies = times[crossings]
        kept = [latencies[0]]
        for latency in latencies[1:]:
            if latency - kept[-1] >= min_isi:
                kept.append(latency)
        print(f"number of triggers found in epoch {epoch}: {len(kept)}")

        for latency in kept:
            events.append({"code": event_code, "latency": float(latency),
                           "epoch": epoch})

    return events_from_list(meta_data, events)


# ---------------------------------------------------------------------------
# RLW_average_epochs_sliding
# ---------------------------------------------------------------------------

_SLIDING_OPS = {
    "average": lambda block, axis: np.mean(block, axis=axis),
    "stdev": lambda block, axis: np.std(block, axis=axis, ddof=1),
    "max": lambda block, axis: np.max(block, axis=axis),
    "min": lambda block, axis: np.min(block, axis=axis),
    "perc75": lambda block, axis: np.percentile(block, 75, axis=axis),
    "perc25": lambda block, axis: np.percentile(block, 25, axis=axis),
    "maxminmean": lambda block, axis: (
        (np.max(block, axis=axis) - np.mean(block, axis=axis)) *
        (np.mean(block, axis=axis) - np.min(block, axis=axis))),
}


def average_epochs_sliding(npy_data, meta_data, operation="average", width=0.2):
    """Slide a window along x and reduce it. Port of RLW_average_epochs_sliding.

    width is in x-axis units. operation is average, stdev, max, min, perc25,
    perc75 or maxminmean. The window is centred, and it shrinks at the two ends
    rather than wrapping, exactly as in the original.
    """
    try:
        reduce = _SLIDING_OPS[str(operation).lower()]
    except KeyError:
        raise ValueError(f"operation must be one of {sorted(_SLIDING_OPS)}, "
                         f"got {operation!r}")

    data = np.asarray(npy_data, dtype=float)
    meta_data = dict(meta_data)

    half = int(round(round(width / xstep(meta_data)) / 2))
    print(f"operation: {operation}, window width {width} "
          f"({2 * half + 1} samples)")

    out = np.empty_like(data)
    for dx in range(data.shape[0]):
        lo = max(0, dx - half)
        hi = min(data.shape[0], dx + half + 1)
        out[dx] = reduce(data[lo:hi], 0)

    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_average_erpimage
# ---------------------------------------------------------------------------

def average_erpimage(npy_data, meta_data, num_lines=100, x_start=None,
                     x_end=None, smooth=True, smooth_width=5):
    """Stack the epochs into an image. Port of RLW_average_erpimage.

    Every epoch becomes one line of the image. With smooth on, each line is a
    Hann-weighted average of the neighbouring epochs, which is what makes the
    picture readable. num_lines then resamples the epoch axis, so a hundred
    lines can summarise any number of trials.

    Returns (time, line, channel, epoch), with one epoch left.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step, origin = xstep(meta_data), xstart(meta_data)

    dx1 = 0 if x_start is None else max(0, int((x_start - origin) / step))
    dx2 = data.shape[0] if x_end is None else min(
        data.shape[0], int((x_end - origin) / step) + 1)
    print(f"ERP image over samples {dx1} to {dx2}, {num_lines} lines")

    n_epochs = data.shape[2]
    window = None
    if smooth and smooth_width > 1:
        half = int(smooth_width) // 2
        window = np.hanning(2 * half + 1)
        print(f"Hanning window width: {window.size}")
    elif smooth:
        print("Hanning window width should be > 1, no smoothing applied")

    cropped = data[dx1:dx2]                       # (x, channel, epoch)
    smoothed = cropped
    if window is not None:
        half = window.size // 2
        smoothed = np.empty_like(cropped)
        for epoch in range(n_epochs):
            lo = max(0, epoch - half)
            hi = min(n_epochs, epoch + half + 1)
            weights = window[lo - epoch + half:hi - epoch + half]
            smoothed[:, :, epoch] = np.mean(
                cropped[:, :, lo:hi] * weights, axis=2)

    if num_lines == n_epochs:
        print("number of epochs equals number of lines, no resampling needed")
        lines = smoothed
        ystep_out = 1.0
    else:
        print("number of epochs does not equal number of lines, resampling")
        source = np.arange(n_epochs)
        target = np.linspace(0, n_epochs - 1, int(num_lines))
        lines = interp1d(source, smoothed, axis=2, kind="cubic"
                         if n_epochs > 3 else "linear")(target)
        ystep_out = float(target[1] - target[0]) if len(target) > 1 else 1.0

    out = np.moveaxis(lines, 2, 1)[..., None]     # (x, line, channel, 1 epoch)

    meta_data["xstart"] = origin + dx1 * step
    meta_data["ystart"] = 1.0
    meta_data["ystep"] = ystep_out
    meta_data["filetype"] = "time_epochs_amplitude"
    for event in (events_to_list(meta_data) or []):
        event["epoch"] = 0
    meta_data.pop("epochdata", None)
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_grand_average
# ---------------------------------------------------------------------------

def grand_average(datasets, dataset_weights=None):
    """Weighted average across datasets. Port of RLW_grand_average.

    datasets is a list of (npy_data, meta_data) pairs. A dataset with several
    epochs is averaged over its epochs first, then the datasets are combined
    with their weights and divided by the total weight. The header of the first
    dataset is kept.

    The MATLAB passes the epoch count as the dimension argument to mean(), which
    averages along whichever dimension that number happens to name; this takes
    the mean over epochs, which is what the function is for.
    """
    if not datasets:
        raise ValueError("no datasets to average")

    if dataset_weights is None:
        dataset_weights = [1.0] * len(datasets)
    weights = np.asarray(dataset_weights, dtype=float)
    if weights.size != len(datasets):
        raise ValueError("one weight per dataset is needed")

    total = None
    for (data, meta), weight in zip(datasets, weights):
        block, _ = _as3d(np.asarray(data, dtype=float))
        if block.shape[2] > 1:
            block = np.mean(block, axis=2, keepdims=True)
        print(f"dataset: {meta.get('name', '?')} - weight: {weight}")
        total = block * weight if total is None else total + block * weight

    out = total / weights.sum()
    meta_data = dict(datasets[0][1])
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_properties
# ---------------------------------------------------------------------------

def properties(meta_data, filetype=None, xstart=None, xstep=None,
               ystart=None, ystep=None):
    """Edit the axis and filetype fields by hand. Port of RLW_properties.

    Whatever is passed is changed and the rest is left alone, which replaces the
    change_x / change_y flags of the original. Returns the metadata only.
    """
    meta_data = dict(meta_data)
    if filetype is not None:
        print("changing filetype")
        meta_data["filetype"] = filetype
    if xstart is not None or xstep is not None:
        print("changing X-axis info")
        if xstart is not None:
            meta_data["xstart"] = float(xstart)
        if xstep is not None:
            meta_data["xstep"] = float(xstep)
            meta_data["fs"] = 1.0 / float(xstep)
    if ystart is not None or ystep is not None:
        print("changing Y-axis info")
        if ystart is not None:
            meta_data["ystart"] = float(ystart)
        if ystep is not None:
            meta_data["ystep"] = float(ystep)
    return meta_data


# ---------------------------------------------------------------------------
# channels
#
# Letswave keeps one struct per channel in header.chanlocs. This project keeps
# the same fields as parallel arrays inside meta_data["chanlocs"], so picking or
# adding a channel means touching every field at once.
# ---------------------------------------------------------------------------

def channel_labels(meta_data):
    """The channel labels as a plain list of strings."""
    labels = meta_data.get("chanlocs", {}).get("labels", [])
    return [str(x) for x in np.atleast_1d(np.squeeze(np.asarray(labels, dtype=object)))]


def find_channels(meta_data, wanted, required=True):
    """Positions of the named channels, matched without regard to case."""
    labels = [name.lower() for name in channel_labels(meta_data)]
    found = []
    for name in np.atleast_1d(wanted):
        name = str(name).lower()
        if name in labels:
            found.append(labels.index(name))
        elif required:
            raise ValueError(f"channel label {name!r} not found")
    return found


def select_channels(meta_data, channel_idx):
    """A copy of the metadata keeping only those channels, in that order."""
    meta_data = dict(meta_data)
    chanlocs = dict(meta_data.get("chanlocs", {}))
    for key, values in chanlocs.items():
        values = np.atleast_1d(np.squeeze(np.asarray(values, dtype=object)))
        if values.size >= max(channel_idx) + 1:
            chanlocs[key] = np.array([values[i] for i in channel_idx], dtype=object)
    meta_data["chanlocs"] = chanlocs
    return meta_data


def append_channel(meta_data, label, **fields):
    """A copy of the metadata with one more channel on the end."""
    meta_data = dict(meta_data)
    chanlocs = dict(meta_data.get("chanlocs", {}))
    n = len(channel_labels(meta_data))
    for key, values in chanlocs.items():
        values = list(np.atleast_1d(np.squeeze(np.asarray(values, dtype=object))))
        if len(values) == n:
            values.append(label if key == "labels" else fields.get(key, 0))
        chanlocs[key] = np.array(values, dtype=object)
    meta_data["chanlocs"] = chanlocs
    return meta_data


def _chanlocs_from_labels(labels):
    """A minimal chanlocs for channels that are built rather than recorded."""
    labels = list(labels)
    return {"labels": np.array(labels, dtype=object),
            "topo_enabled": np.zeros(len(labels), dtype=int),
            "SEEG_enabled": np.zeros(len(labels), dtype=int)}


# ---------------------------------------------------------------------------
# RLW_merge_channels
# ---------------------------------------------------------------------------

def merge_channels(datasets):
    """Put the channels of several datasets side by side. Port of RLW_merge_channels.

    datasets is a list of (npy_data, meta_data) pairs that agree on everything
    but their channels. The first header is kept, chanlocs and events are
    concatenated, and duplicate events are dropped.
    """
    if not datasets:
        raise ValueError("no datasets to merge")

    out, meta_data = _as3d(np.asarray(datasets[0][0]))[0], dict(datasets[0][1])
    labels = channel_labels(meta_data)
    events = events_to_list(meta_data)

    for data, meta in datasets[1:]:
        block, _ = _as3d(np.asarray(data))
        if (block.shape[0], block.shape[2]) != (out.shape[0], out.shape[2]):
            raise ValueError("datasets cannot be merged as their sizes do not match: "
                             f"{out.shape} against {block.shape}")
        out = np.concatenate([out, block], axis=1)
        labels += channel_labels(meta)
        events += events_to_list(meta)

    seen, unique = set(), []
    for event in events:
        key = (str(event["code"]), event["latency"], event["epoch"])
        if key not in seen:
            seen.add(key)
            unique.append(event)

    meta_data["chanlocs"] = _chanlocs_from_labels(labels)
    meta_data["history"] = {}
    meta_data = events_from_list(meta_data, unique)
    print(f"merged into {out.shape[1]} channels")
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_pool_channels
# ---------------------------------------------------------------------------

def pool_channels(npy_data, meta_data, channel_labels_wanted, channel_weights=None,
                  mixed_channel_label="newchan", keep_original_channels=True):
    """Average several channels into a new one. Port of RLW_pool_channels.

    The pooled channel is the weighted mean of the named channels; with no
    weights they count equally. It is appended to the data, or replaces it when
    keep_original_channels is False.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    idx = find_channels(meta_data, channel_labels_wanted, required=False)
    if not idx:
        raise ValueError("selected channels not found in the dataset")

    weights = (np.ones(len(idx)) if channel_weights is None
               else np.asarray(channel_weights, dtype=float)[:len(idx)])
    print(f"pooling {len(idx)} channels with weights {weights}")

    pooled = np.tensordot(weights, data[:, idx], axes=([0], [1])) / weights.sum()
    pooled = pooled[:, None, :] if pooled.ndim == 2 else pooled

    if keep_original_channels:
        out = np.concatenate([data, pooled], axis=1)
        meta_data = append_channel(meta_data, mixed_channel_label)
    else:
        out = pooled
        meta_data["chanlocs"] = _chanlocs_from_labels([mixed_channel_label])

    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_flip_electrodes
# ---------------------------------------------------------------------------

def flip_electrodes(npy_data, meta_data, chan_label_pairs):
    """Swap the signals of paired electrodes. Port of RLW_flip_electrodes.

    chan_label_pairs is a list of two-element pairs, for example
    [("C3", "C4"), ("P3", "P4")]. The labels stay where they are and the data
    moves, which is how Letswave mirrors a montage left to right.
    """
    data, _ = _as3d(np.asarray(npy_data))
    meta_data = dict(meta_data)
    if not chan_label_pairs:
        return data, meta_data

    order = list(range(data.shape[1]))
    labels = [name.lower() for name in channel_labels(meta_data)]
    for left, right in chan_label_pairs:
        left, right = str(left).lower(), str(right).lower()
        if left in labels and right in labels:
            order[labels.index(left)] = labels.index(right)

    print(f"flipped {len(chan_label_pairs)} electrode pairs")
    out = data[:, order]
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_edit_electrodes_info
# ---------------------------------------------------------------------------

def edit_electrodes_info(meta_data, chanlocs):
    """Overwrite the details of named channels. Port of RLW_edit_electrodes_info.

    chanlocs is a list of dicts, each with a "labels" key naming the channel it
    describes; its other keys replace that channel's fields. Channels that are
    not named are untouched. Returns the metadata only.
    """
    meta_data = dict(meta_data)
    if not chanlocs:
        return meta_data

    existing = dict(meta_data.get("chanlocs", {}))
    labels = [name.lower() for name in channel_labels(meta_data)]

    for record in chanlocs:
        name = str(record.get("labels", "")).lower()
        if name not in labels:
            continue
        position = labels.index(name)
        for key, value in record.items():
            if key == "labels":
                continue
            values = list(np.atleast_1d(np.squeeze(
                np.asarray(existing.get(key, np.zeros(len(labels))), dtype=object))))
            while len(values) < len(labels):
                values.append(0)
            values[position] = value
            existing[key] = np.array(values, dtype=object)

    meta_data["chanlocs"] = existing
    print(f"edited the info of {len(chanlocs)} electrodes")
    return meta_data


# ---------------------------------------------------------------------------
# RLW_rereference_advanced
# ---------------------------------------------------------------------------

def rereference_advanced(npy_data, meta_data, active_channel_labels,
                         reference_channel_labels):
    """Build a custom montage. Port of RLW_rereference_advanced.

    Pairs each active channel with its own reference and returns one channel per
    pair, labelled "active-reference". Unlike globalreferencing, which takes one
    average reference for everything, this is how a bipolar montage is made.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    active = np.atleast_1d(active_channel_labels)
    reference = np.atleast_1d(reference_channel_labels)
    if len(active) != len(reference):
        raise ValueError("there must be one reference channel per active channel")

    active_idx = find_channels(meta_data, active)
    reference_idx = find_channels(meta_data, reference)

    out = data[:, active_idx] - data[:, reference_idx]
    meta_data["chanlocs"] = _chanlocs_from_labels(
        [f"{a}-{r}" for a, r in zip(active, reference)])
    print(f"rereferenced into {out.shape[1]} bipolar channels")
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_linear_channel_map
# ---------------------------------------------------------------------------

def linear_channel_map(npy_data, meta_data, num_lines=100):
    """Interpolate across channels into an image. Port of RLW_linear_channel_map.

    Treats the channel order as a line through the head and interpolates it onto
    num_lines, giving one image per epoch instead of a stack of traces. Returns
    (time, line, 1 channel, epoch).
    """
    from scipy.interpolate import RectBivariateSpline

    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    n_channels = data.shape[1]
    source = np.arange(n_channels)
    target = np.linspace(0, n_channels - 1, int(num_lines))
    print(f"linear channel map: {n_channels} channels onto {num_lines} lines")

    out = np.empty((data.shape[0], int(num_lines), 1, data.shape[2]))
    x = np.arange(data.shape[0])
    for epoch in range(data.shape[2]):
        spline = RectBivariateSpline(x, source, data[:, :, epoch],
                                     kx=min(3, data.shape[0] - 1),
                                     ky=min(3, n_channels - 1))
        out[:, :, 0, epoch] = spline(x, target)

    meta_data["chanlocs"] = _chanlocs_from_labels(["CSD"])
    meta_data["ystart"] = float(target[0])
    meta_data["ystep"] = float(target[1] - target[0]) if len(target) > 1 else 1.0
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_weighted_channel_average_template and _apply
# ---------------------------------------------------------------------------

def weighted_channel_average_template(npy_data, meta_data, x, num_channels=6,
                                      selected_channels=None, epoch=0,
                                      peakdir="max", normalize=True):
    """Pick the channels that carry a peak, and how much each contributes.

    Port of RLW_weighted_channel_average_template. Reads every channel at the
    time x, keeps the num_channels largest (peakdir max, min or absmax) and
    turns their values into weights. Returns (weights, labels), ready for
    weighted_channel_average_apply.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    labels = channel_labels(meta_data)

    idx = (list(range(len(labels))) if selected_channels is None
           else find_channels(meta_data, selected_channels, required=False))
    print(f"{len(idx)} channels with matching labels")

    dx = int(round((float(x) - xstart(meta_data)) / xstep(meta_data)))
    dx = max(0, min(dx, data.shape[0] - 1))
    values = data[dx, idx, min(epoch, data.shape[2] - 1)]

    peakdir = str(peakdir).lower()
    if peakdir == "max":
        order = np.argsort(values)[::-1]
    elif peakdir == "min":
        order = np.argsort(values)
    elif peakdir == "absmax":
        order = np.argsort(np.abs(values))[::-1]
    else:
        raise ValueError(f"peakdir must be max, min or absmax, got {peakdir!r}")

    order = order[:int(num_channels)]
    weights = values[order]
    if normalize:
        weights = weights / weights.sum()

    template_labels = [labels[idx[i]] for i in order]
    print(f"template channels: {template_labels}")
    return weights, template_labels


def weighted_channel_average_apply(npy_data, meta_data, template_weights,
                                   template_labels):
    """Collapse the channels using a template. Port of RLW_weighted_channel_average_apply.

    Returns a single channel, "chanavg", the weighted mean of the template
    channels.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    idx = find_channels(meta_data, template_labels, required=False)
    weights = np.asarray(template_weights, dtype=float)
    if len(idx) != weights.size:
        raise ValueError("not all template channels were found, cannot apply template")

    out = np.tensordot(weights, data[:, idx], axes=([0], [1])) / weights.sum()
    out = out[:, None, :] if out.ndim == 2 else out

    meta_data["chanlocs"] = _chanlocs_from_labels(["chanavg"])
    print("applied the weighted channel average template")
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_ICA_unmix, RLW_ICA_remix and RLW_PCA_compute
#
# Letswave hides the mixing and unmixing matrices in the dataset history and
# digs them back out. Here they are passed in, which is the same information
# without the archaeology; performICA in FPyVS_appylication already produces an
# mne ICA object, and these are for working with the matrices directly.
# ---------------------------------------------------------------------------

def ica_unmix(npy_data, meta_data, ica_um):
    """Turn channels into components. Port of RLW_ICA_unmix.

    ica_um is the unmixing matrix, one row per component. Returns the component
    time courses, labelled IC1, IC2 and so on, together with the channel
    information the components replaced so ica_remix can put it back.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    ica_um = np.asarray(ica_um, dtype=float)
    original_chanlocs = meta_data.get("chanlocs")

    out = np.einsum("ij,xje->xie", ica_um, data)
    meta_data["chanlocs"] = _chanlocs_from_labels(
        [f"IC{i + 1}" for i in range(ica_um.shape[0])])
    meta_data["old_chanlocs"] = original_chanlocs
    print(f"ICA unmix into {out.shape[1]} components")
    return out, _refresh_shape(out, meta_data), original_chanlocs


def ica_remix(npy_data, meta_data, ica_mm, ic_list=None, old_chanlocs=None):
    """Turn components back into channels. Port of RLW_ICA_remix.

    ica_mm is the mixing matrix. ic_list names the components to keep (0-based);
    the columns of the others are zeroed, which is how a blink component is
    removed. old_chanlocs restores the original labels, and ica_unmix leaves a
    copy in the metadata for exactly that.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    ica_mm = np.array(ica_mm, dtype=float, copy=True)
    if ic_list is not None:
        keep = set(int(i) for i in np.atleast_1d(ic_list))
        print(f"selected ICs: {sorted(keep)}")
        for column in range(ica_mm.shape[1]):
            if column not in keep:
                ica_mm[:, column] = 0

    out = np.einsum("ij,xje->xie", ica_mm, data)

    chanlocs = old_chanlocs if old_chanlocs is not None else meta_data.get("old_chanlocs")
    if chanlocs is not None:
        print("found the original channel information")
        meta_data["chanlocs"] = chanlocs
    else:
        print("did not find the original channel information, "
              "computing dummy channel labels")
        meta_data["chanlocs"] = _chanlocs_from_labels(
            [f"C{i + 1}" for i in range(out.shape[1])])
    meta_data.pop("old_chanlocs", None)
    return out, _refresh_shape(out, meta_data)


def pca_compute(npy_data, meta_data=None):
    """Principal components of the channel covariance. Port of RLW_PCA_compute.

    Returns {"ica_mm": mixing, "ica_um": unmixing}, the same pair of matrices
    ica_unmix and ica_remix expect. The eigenvectors are ordered by eigenvalue
    the way the original sorts them, smallest first.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    print("computing PCA")

    # every epoch laid end to end, one row per channel
    flat = np.concatenate([data[:, :, e].T for e in range(data.shape[2])], axis=1)
    covariance = flat @ flat.T

    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    vectors = vectors[:, order]

    return {"ica_mm": vectors, "ica_um": np.linalg.inv(vectors)}


# ---------------------------------------------------------------------------
# RLW_ocular_remove
# ---------------------------------------------------------------------------

def ocular_remove(npy_data, meta_data, eog_channels):
    """Regress the eye channels out of every channel. Port of RLW_ocular_remove.

    For each channel a least squares fit against the EOG channels and a constant
    gives the weight of the eye signal in it, and that much EOG is subtracted.
    Cheaper than ICA, and it leaves the EOG channels themselves in place.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)

    eog_idx = (find_channels(meta_data, eog_channels)
               if not np.issubdtype(np.asarray(eog_channels).dtype, np.number)
               else [int(i) for i in np.atleast_1d(eog_channels)])
    if not eog_idx:
        return data, meta_data
    print(f"regressing out EOG channels {eog_idx}")

    n_samples = data.shape[0] * data.shape[2]
    eog = np.stack([data[:, i, :].reshape(n_samples) for i in eog_idx], axis=1)
    design = np.column_stack([eog, np.ones(n_samples)])

    out = data.copy()
    for channel in range(data.shape[1]):
        y = data[:, channel, :].reshape(n_samples)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        for position, i in enumerate(eog_idx):
            out[:, channel, :] -= coefficients[position] * data[:, i, :]

    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_suppress_artifact and RLW_suppress_artifact_event
# ---------------------------------------------------------------------------

def suppress_artifact(npy_data, meta_data, x_start=-0.005, x_end=0.005):
    """Draw a straight line across a window. Port of RLW_suppress_artifact.

    Replaces everything between x_start and x_end with a line joining the two
    end samples, which is the usual way a stimulation artifact is removed.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step, origin = xstep(meta_data), xstart(meta_data)

    dx1 = int((x_start - origin) / step)
    dx2 = int((x_end - origin) / step)
    dx1, dx2 = max(0, dx1), min(data.shape[0] - 1, dx2)
    print(f"suppressing the artifact between samples {dx1} and {dx2}")
    if dx2 <= dx1:
        return data, meta_data

    out = data.copy()
    span = np.arange(dx2 - dx1 + 1) / (dx2 - dx1)
    left, right = data[dx1], data[dx2]
    out[dx1:dx2 + 1] = left + span[:, None, None] * (right - left)
    return out, _refresh_shape(out, meta_data)


def suppress_artifact_event(npy_data, meta_data, event_code, x_start=-0.005,
                            x_end=0.005, interp_method="spline"):
    """Interpolate over the artifact around each event.

    Port of RLW_suppress_artifact_event. Every event with that code marks a
    window; the samples inside all the windows are thrown away and the trace is
    interpolated back over them from the samples that remain.
    interp_method is spline, linear or nearest.
    """
    kind = {"spline": "cubic", "cubic": "cubic", "linear": "linear",
            "nearest": "nearest"}.get(str(interp_method).lower(), "cubic")

    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step, origin = xstep(meta_data), xstart(meta_data)
    n = data.shape[0]

    blanked = {epoch: set() for epoch in range(data.shape[2])}
    for event in events_to_list(meta_data):
        if str(event["code"]).lower() != str(event_code).lower():
            continue
        latency = event["latency"]
        dx1 = int((latency + x_start - origin) / step)
        dx2 = int((latency + x_end - origin) / step)
        epoch = event["epoch"]
        if epoch in blanked:
            blanked[epoch].update(range(max(0, dx1), min(n, dx2 + 1)))

    out = data.copy()
    for epoch, removed in blanked.items():
        if not removed:
            continue
        keep = np.array([i for i in range(n) if i not in removed])
        if keep.size < 4:
            print(f"epoch {epoch}: too little left to interpolate, skipped")
            continue
        print(f"epoch {epoch}: interpolating over {len(removed)} samples")
        out[:, :, epoch] = interp1d(keep, data[keep, :, epoch], axis=0, kind=kind,
                                    bounds_error=False,
                                    fill_value="extrapolate")(np.arange(n))

    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_segmentation_SSEP
# ---------------------------------------------------------------------------

def segmentation_ssep(npy_data, meta_data, event_labels, cycle_skip=0,
                      cycle_total=1, cycle_frequency=1.0):
    """Cut epochs that hold a whole number of stimulation cycles.

    Port of RLW_segmentation_SSEP. The epoch starts cycle_skip cycles after the
    trigger and runs for cycle_total - cycle_skip cycles, so at
    cycle_frequency 1.2 Hz with 64 cycles every epoch holds exactly 64 periods
    of the oddball and the FFT lands on the bin.
    """
    data, _ = _as3d(np.asarray(npy_data, dtype=float))
    meta_data = dict(meta_data)
    step, origin = xstep(meta_data), xstart(meta_data)

    events = events_to_list(meta_data)
    if not events:
        print("no events, nothing to segment")
        return data, meta_data

    wanted = [str(label).lower() for label in np.atleast_1d(event_labels)]
    matching = [e for e in events if str(e["code"]).lower() in wanted]
    if not matching:
        print("event code not found in dataset")
        return data, meta_data
    print(f"{len(matching)} corresponding events found in dataset")

    x_start = cycle_skip / cycle_frequency
    duration = (cycle_total - cycle_skip) / cycle_frequency
    dxsize = int(round(duration / step))

    epochs, kept_events = [], []
    for position, trigger in enumerate(matching):
        dx1 = int(round((trigger["latency"] + x_start - origin) / step))
        dx2 = dx1 + dxsize
        if dx1 < 0 or dx2 > data.shape[0]:
            print(f"event at {trigger['latency']} falls outside the data, skipped")
            continue
        epochs.append(data[dx1:dx2, :, trigger["epoch"]])

        for event in events:
            if event["epoch"] != trigger["epoch"]:
                continue
            new_latency = event["latency"] - trigger["latency"]
            if x_start <= new_latency <= x_start + duration:
                kept_events.append({"code": event["code"],
                                    "latency": new_latency,
                                    "epoch": len(epochs) - 1})

    if not epochs:
        print("no usable epochs")
        return data, meta_data

    out = np.stack(epochs, axis=2)
    meta_data["xstart"] = x_start
    meta_data = events_from_list(meta_data, kept_events)
    meta_data.pop("epochdata", None)
    print(f"segmented into {out.shape[2]} epochs of {dxsize} samples")
    return out, _refresh_shape(out, meta_data)


# ---------------------------------------------------------------------------
# RLW_edit_electrodes_SEEG
# ---------------------------------------------------------------------------

def edit_electrodes_seeg(meta_data, list_labels, list_x, list_y, list_z):
    """Give named electrodes SEEG coordinates. Port of RLW_edit_electrodes_SEEG.

    Sets X, Y and Z for each named channel and marks it as SEEG rather than
    scalp, so it is left out of the topographies. Returns the metadata only.
    """
    meta_data = dict(meta_data)
    if list_labels is None or len(list_labels) == 0:
        return meta_data

    records = []
    for label, x, y, z in zip(list_labels, list_x, list_y, list_z):
        print(f"setting SEEG electrode coordinate: {label}")
        records.append({"labels": label, "X": float(x), "Y": float(y),
                        "Z": float(z), "SEEG_enabled": 1, "topo_enabled": 0})
    return edit_electrodes_info(meta_data, records)
