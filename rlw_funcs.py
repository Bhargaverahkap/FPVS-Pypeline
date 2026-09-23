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
from scipy.signal import detrend as _detrend
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
