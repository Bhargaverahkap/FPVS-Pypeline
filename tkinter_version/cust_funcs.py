# Tkinter build. The Jupyter build of this module lives in the repository root.
# Only the interactive viewers differ: ipywidgets there, Tkinter windows here.
# Hello Bhargav, welcome to your sandbox


import numpy as np
import trimesh
import plotly.graph_objects as go
from scipy.interpolate import Rbf

from pathlib import Path as _Path
# Data assets (electrode locations, head meshes) stay in the repository root.
_ASSET_DIR = _Path(__file__).resolve().parent.parent

def rebrand_lw6data(filepath):
    """ Rewrite the meta_data from matlab style encryption to python style encryption to make life easier in python

    :param filepath: enter the full filepath starting from the drive it is stored in.
    :return: The newly created meta_data, with the name meta_data
    """
    import numpy as np
    import scipy.io as sci
    meta_data = sci.loadmat(filepath)

    print("the keys in old data are: ", meta_data["header"].dtype.names)

    org_data = {
        "gui_info": ['no gui, no info'],
        "originalfilepath": filepath,
    }

    chanlocs = {
        "labels": np.squeeze(meta_data["header"]["chanlocs"][0][0]["labels"]),
        "topo_enabled" : np.squeeze(meta_data["header"]["chanlocs"][0][0]["topo_enabled"]),
        "SEEG_enabled" : np.squeeze(meta_data["header"]["chanlocs"][0][0]["SEEG_enabled"])
        }

    # A recording with no triggers stores an empty events array, and indexing
    # into that raised "index 0 is out of bounds for axis 0 with size 0".
    # Flattening handles the singleton dimension matlab adds without collapsing
    # a single event down to a scalar, and leaves an empty array empty.
    allevents = np.asarray(meta_data["header"]["events"][0][0]).reshape(-1)

    if allevents.size == 0:
        print("no events found in", filepath)
        code = np.array([], dtype=object)
        latency = np.array([], dtype=float)
        epoch = np.array([], dtype=float)
    else:
        code = np.array([l[0][0] for l in allevents])
        latency = np.array([l[1][0][0] for l in allevents])
        epoch = np.array([l[2][0][0] for l in allevents])

    events = {
        "code": code,
        "latency": latency,
        "epoch": epoch
    }

    fields = {
        "filetype": "Type of file this is stored as",
        "name": "Name of the file as stored in the meta_data",
        "tags": "I have no idea, but this is as it is stored in the meta_data",
        "history": "Record of all changes made to the data as operation stored as suffix: history till then",
        "origins": "Origins of the data, contains gui info and original filepath",
        "datasize": "size of the data",
        "xstart": "I have no idea what this is, but this was present in the original meta_data",
        "ystart": "I have no idea what this is, but this was present in the original meta_data",
        "zstart": "I have no idea what this is, but this was present in the original meta_data",
        "xstep": "I have no idea what this is, but this was present in the original meta_data",
        "ystep": "I have no idea what this is, but this was present in the original meta_data",
        "zstep": "I have no idea what this is, but this was present in the original meta_data",
        "chanlocs": "channel location data, stored as follows",
        "chanlocs.labels": "Channel labels stored as strings",
        "chanlocs.SEEG_enabled": "I have no idea what this is, but was present in the original meta_data",
        "chanlocs.TOPO_enabled": "I have no idea what this is, but was present in the original meta_data",
        "events": "event data as stored as follows",
        "events.code": "event code as described in the triggers document",
        "events.latency": "time delay of current trigger from start of experiment",
        "events.epoch": "event code as described in the triggers document",
        "fs": "Sampling frequency in Hz",
        "activation": "microvolts",
    }

    new_data = {
        "filetype": np.squeeze(meta_data["header"]["filetype"]),
        "name": np.squeeze(meta_data["header"]["name"]),
        "tags": {},
        "history": {},
        "origins": org_data,
        "datasize": np.squeeze(meta_data["header"]["datasize"]),
        "xstart": np.squeeze(meta_data["header"]["xstart"]),
        "ystart": np.squeeze(meta_data["header"]["ystart"]),
        "zstart": np.squeeze(meta_data["header"]["zstart"]),
        "xstep": np.squeeze(meta_data["header"]["xstep"]),
        "ystep": np.squeeze(meta_data["header"]["ystep"]),
        "zstep": np.squeeze(meta_data["header"]["zstep"]),
        "chanlocs": chanlocs,
        "events": events,
        "fs": 2048,
        "activation": "microvolts",
        "fields": fields
    }

    print("the keys in the new data are: ", new_data.keys())
    return assign(new_data)

def assign(data):
    import numpy as np
    """Recursively removes nested NumPy wrappers from dictionaries, lists, and arrays."""
    # If it is a dictionary, unpack all its keys and values
    if isinstance(data, dict):
        return {k: assign(v) for k, v in data.items()}

    # If it is a NumPy array
    if isinstance(data, np.ndarray):
        # Handle 0-dimensional arrays
        if data.ndim == 0:
            return assign(data.item())
        # Handle 1-element or deeply nested text wrappers like array(['Fp1'])
        if data.size == 1:
            return assign(data.ravel()[0])
        # Recursively unpack lists of arrays (like your channel labels)
        return [assign(item) for item in data]

    # Convert NumPy string scalars to native Python strings
    if isinstance(data, np.generic):
        return data.item()

    # Leave clean data types (strings, ints, floats, paths) untouched
    return data

def loadMetadata(filepath):
    import pickle
    if filepath.suffix !='.pkl':
        filepath=filepath.with_suffix('.pkl')

    with open(filepath, "rb") as f:
        meta_data = pickle.load(f)
    return meta_data

def saveMetadata(data, filepath):
    import pickle

    if filepath.suffix != '.pkl':
        filepath = filepath.with_suffix('.pkl')

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def updatemetadataHistory(meta_data,step_prefix):
    if step_prefix is None:
        raise ValueError(f"Write a step prefix")

    current_state = {key: value for key, value in meta_data.items() if key != "history"}
    meta_data["history"][step_prefix] = current_state
    return meta_data

# These functions have become obselete now
# def showmeFFT(signal,fs,freqlim=None,pltsize=None,label=None,titlestr=None,epochid=None):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     if epochid is None:
#         epochid = 0

#     if len(signal.shape) > 1:
#         signal = np.squeeze(signal)

#     if titlestr is None:
#         titlestr = [f"FFT of signal for {label} in epoch {epochid}"]

#     if freqlim is None:
#         freqlim = [0, fs // 2]

#     if pltsize is None:
#         pltsize = [12, 3]

#     n_timepoints = int(len(signal))
#     # Converting signal from sample to freq domain
#     FFT_vals=np.abs(np.fft.fft(signal))
#     FFT_vals = (np.abs(FFT_vals) / n_timepoints)
#     freqs=np.fft.fftfreq(len(FFT_vals),1/fs)
#     idx = (freqs >= freqlim[0]) & (freqs <= freqlim[1])

#     #actual plotting
#     plt.figure(figsize=pltsize)
#     markerline, stemlines, baseline = plt.stem(freqs[idx], FFT_vals[idx])
#     markerline.set_markersize(1)
#     plt.setp(stemlines, linewidth= 1)
#     plt.title(titlestr)
#     plt.xlabel("Freq [Hz]")
#     ticks = np.arange(1.2*(freqlim[0]//1.2), freqlim[1] + 1.2, 1.2)  # include endpoint
#     plt.xticks(ticks,fontsize=8)
#     plt.ylabel("abs Amp")
#     legendstr = [f"epoch:{epochid}"]
#     plt.legend(legendstr,loc="upper right")
#     plt.title(titlestr)
#     plt.show()

# def showmeSignal(signal,duration=None,pltsize=None,titlestr=None,epochid=None):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     if len(signal.shape)>1:
#         signal = np.squeeze(signal)

#     if epochid is None:
#         epochid = 0

#     if duration is None:
#         duration = np.linspace(0,len(signal)-1,len(signal),dtype=int)

#     if pltsize is None:
#         pltsize = [12, 4]

#     if titlestr is None:
#         titlestr = "Channel activation"

#     #actual plotting
#     plt.figure(figsize=pltsize)
#     plt.plot(duration,signal,lw=1)
#     plt.title(titlestr)
#     plt.xlabel("Samples")
#     plt.ylabel("amplitude")
#     legendstr = [f"epoch:{epochid}"]
#     if len(signal.shape) >1:
#         legend_labels = [f"line {i}" for i in range(signal.shape[1])]

#     if len(signal.shape)==1:
#         plt.legend(legendstr,loc="upper right")
#     else:
#         plt.legend(legend_labels,loc="upper left", bbox_to_anchor=(1, 1))
#         plt.subplots_adjust(right=0.75)

#     plt.figure
#     plt.show()

# def showmeSummaryPlot(signal,fs,chname,subjid,pltsize=None,freqlim=None,epochid=None):
#     import numpy as np
#     if len(signal.shape) > 1:
#         signal = np.squeeze(signal)

#     if epochid is None:
#         epochid = 0

#     if freqlim is None:
#         freqlim = [0.1, 20]

#     if pltsize is None:
#         pltsize = [12, 3]

#     sigtitlestr = f"amp vs samples for sub:{subjid} chid:{chname}"
#     ffttitlestr = f"FFT spectrum for sub:{subjid} chid:{chname}"

#     showmeSignal(signal,titlestr=sigtitlestr,epochid=epochid,pltsize=pltsize)
#     showmeFFT(signal,fs,freqlim=freqlim,pltsize=pltsize,titlestr=ffttitlestr,epochid=epochid)

def zscoreChunks(FFT, freqs, f0=None, window=None, exclude_bins=1, buffer=None):
    #funciton under construction
    import numpy as np

    if f0 is None:
        fo=freqs[len(freqs)//2]

    if buffer is None:
        buffer = 5

    if window is None:
        window = .4

    df = freqs[1] - freqs[0]
    half_w = window / 2

    # indices around f0
    if f0 != freqs[len(freqs)/2] :
        idx_signal = [np.where(freqs==f0) , np.where(freqs==f0)+1]
    else:
        idx_signal = np.where(freqs==f0)

    # neighborhood
    noise_idx = (freqs >= fo+buffer*df) & (freqs <= f0-buffer*df)

    # remove center + neighbors
    signal = FFT[idx_signal]
    noise_mean = np.mean(FFT[noise_idx])
    noise_std = np.std(FFT[noise_idx])

    z = (np.mean(signal) - noise_mean) / noise_std
    return z

def givemeNNearestNeighbour(meta_data,chnameid):
    import numpy as np
    labels=meta_data["chanlocs"]["labels"]
    labels = labels.astype(str)
    if isinstance(chnameid, str):
        chid = np.squeeze(np.where(chnameid==labels))
    else:
        chid=chnameid

    #shaping the data
    X=meta_data["chanlocs"]["X"]
    X=X.astype(float)
    Y=meta_data["chanlocs"]["Y"]
    Y=Y.astype(float)
    Z=meta_data["chanlocs"]["Z"]
    Z=Z.astype(float)
    coords = np.column_stack((X, Y, Z))

    #calculating distances
    ref = coords[chid]
    distances = np.linalg.norm(coords-ref,axis=1)

    #Sorting the distances
    sorted_idx = np.argsort(distances, kind='stable')
    sorted_labels = labels[sorted_idx]

    return sorted_idx[1:],sorted_labels[1:]

# This function is also obselete as there is a newer Generation of function (showmesummaryplot in FPyVS_applycation)
# def showmeSignalUI(npy_data,meta_data,xlim=None,ylim=None):
#     import ipywidgets as widgets
#     from IPython.display import display
#     import matplotlib.pyplot as plt
#     import numpy as np

#     epoch = [f"epoch{i}" for i in range(npy_data.shape[2])]
#     labels = np.squeeze(meta_data["chanlocs"]["labels"])
#     labels = labels.astype(str)

#     # 1. Generate Dummy Data (40 Epochs, 60 Series, 100 Timepoints)
#     # Shape: (Epochs, Series, Time)

#     data_matrix = np.moveaxis(npy_data,0,-1) #moves first axis to the last
#     print(data_matrix.shape)
#     # 2. The Plotting Function
#     def plot_data(epoch_idx, selected_series):
#         plt.figure(figsize=(9, 5))

#         if not selected_series:
#             plt.text(0.5, 0.5, "Select series from the list", ha='center', va='center')
#         else:
#             for s_idx in selected_series:
#                 series_data = data_matrix[s_idx, epoch_idx,  :]
#                 plt.plot(series_data, label=labels[s_idx])

#             # plt.title(f"Epoch {epoch_idx} | Comparing {len(selected_series)} Series")
#             plt.xlabel("Time Step")
#             plt.ylabel("Value")
#             plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#             plt.grid(True, linestyle='--', alpha=0.6)
#             if xlim is not None:
#                 plt.xlim(-abs(xlim), abs(xlim))
#             if ylim is not None:
#                 plt.ylim(ylim[0],ylim[1])

#         plt.show()

#     # 3. The Widgets
#     # Slider for "scrolling" through 40 epochs
#     epoch_slider = widgets.IntSlider(
#         value=0, min=0, max=npy_data.shape[2]-1, step=1,
#         description='Epoch:',
#         continuous_update=True,  # Update plot while dragging
#         layout={'width': '1000px'}
#     )

#     # Multi-select for the 60 series
#     series_selector = widgets.SelectMultiple(
#         options=[(labels[i], i) for i in range(npy_data.shape[1])],
#         value=[0, 1, 2, 3],  # Default selection (first 4)
#         description='Channel:',
#         layout={'height': '550px', 'width': '150px'}
#     )

#     # 4. Linking it all together
#     out = widgets.interactive_output(
#         plot_data,
#         {'epoch_idx': epoch_slider, 'selected_series': series_selector}
#     )

#     # 5. Organizing the Layout
#     # Putting the epoch slider on top and the list next to the plot
#     ui = widgets.VBox([
#         epoch_slider,
#         widgets.HBox([series_selector, out])
#     ])

#     display(ui)

def showmeSignalUIandInterp(npy_data, meta_data, xlim=None, ylim=None): #discarded function but kept incase someone wants to use it for parts
    # Tkinter build: same layout as the Jupyter build (epoch slider, channel list,
    # three action buttons, live status text) drawn with Tk widgets instead of
    # ipywidgets. Blocks until "Finalize & Close" or the window is closed.
    import numpy as np
    import _tk_helpers as tkh
    import tkinter as tk
    from tkinter import ttk

    labels = np.squeeze(meta_data["chanlocs"]["labels"]).astype(str)
    data_matrix = np.moveaxis(npy_data, 0, -1)

    to_interpolate = []
    bad_but_ignore = []

    win = tkh.PlotWindow("Channel inspection and interpolation")

    epoch_idx = tk.IntVar(value=0)

    def plot_data(*_):
        ax = win.clear()
        selected_series = tkh.selection(series_selector)
        if not selected_series:
            ax.text(0.5, 0.5, "Select channels to view", ha='center', va='center')
        else:
            for s_idx in selected_series:
                # Visual cue: dashed for bad-no-interp, solid for interpolate
                style = '-'
                alpha = 1.0
                if s_idx in to_interpolate:
                    style = '-'
                elif s_idx in bad_but_ignore:
                    style = '--'
                    alpha = 0.6

                ax.plot(data_matrix[s_idx, int(epoch_idx.get()), :], label=labels[s_idx],
                        linestyle=style, alpha=alpha)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax.set_title(f"Epoch {int(epoch_idx.get())}")
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        win.draw()

    ttk.Label(win.controls, text="Epoch").pack(anchor="w")
    tk.Scale(
        win.controls,
        from_=0,
        to=npy_data.shape[2] - 1,
        orient=tk.HORIZONTAL,
        variable=epoch_idx,
        command=plot_data,
        length=200,
    ).pack(anchor="w")

    series_selector = tkh.make_listbox(
        win.controls,
        "Channels",
        [labels[i] for i in range(npy_data.shape[1])],
        height=16,
        default=(0, 1, 2, 3),
    )
    series_selector.bind("<<ListboxSelect>>", plot_data)

    status = ttk.Label(win.controls, justify="left", text="")

    def update_status():
        status.config(text=(
            f"Interpolate: {[labels[i] for i in sorted(to_interpolate)]}\n"
            f"Bad (Ignore): {[labels[i] for i in sorted(bad_but_ignore)]}"
        ))

    def on_interp_clicked():
        for idx in tkh.selection(series_selector):
            if idx not in to_interpolate: to_interpolate.append(idx)
            if idx in bad_but_ignore: bad_but_ignore.remove(idx)  # Ensure it's only in one list
        update_status()
        plot_data()

    def on_bad_clicked():
        for idx in tkh.selection(series_selector):
            if idx not in bad_but_ignore: bad_but_ignore.append(idx)
            if idx in to_interpolate: to_interpolate.remove(idx)  # Ensure it's only in one list
        update_status()
        plot_data()

    def on_done_clicked():
        win.win.destroy()
        print("--- FINAL CHANNEL REPORT ---")
        print(f"To Interpolate: {sorted(to_interpolate)} ({[labels[i] for i in sorted(to_interpolate)]})")
        print(f"Bad (Ignore):   {sorted(bad_but_ignore)} ({[labels[i] for i in sorted(bad_but_ignore)]})")

    ttk.Button(win.controls, text="Interpolate", command=on_interp_clicked).pack(anchor="w", pady=(10, 0))
    ttk.Button(win.controls, text="Mark Bad (No Interp)", command=on_bad_clicked).pack(anchor="w")
    status.pack(anchor="w", pady=(8, 8))
    ttk.Button(win.controls, text="Finalize & Close", command=on_done_clicked).pack(anchor="w")

    update_status()
    plot_data()
    win.wait()

    # Returning both lists so you can handle them differently in your pipeline
    badch = np.union1d(to_interpolate, bad_but_ignore)
    for i in range(len(to_interpolate)):
        srt_idx,srt_labels = givemeNNearestNeighbour(meta_data, to_interpolate[i])
        srtd_idx = np.where(~np.isin(srt_labels, badch))[0]
        srtd_idx = srtd_idx[:3]
        badch = np.append(badch, srt_labels[srtd_idx])
        npy_data[:,to_interpolate[i],:] = np.mean(npy_data[:,srtd_idx,:], axis=1)


    return npy_data,{"interpolate": to_interpolate, "bad_no_interp": bad_but_ignore}

def givemeUniqueTuples(data,tolerance = None):
    if tolerance is None:
        tolerance = 5

    unique_pairs = []
    for s, e in data:
        is_duplicate = False

        for us, ue in unique_pairs:
            if abs(s - us) <= tolerance and abs(e - ue) <= tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_pairs.append((s, e))

    return sorted(unique_pairs)

def showmeTopomap(data,meta_data):

    import mne
    import plotly.graph_objects as go
    from scipy.interpolate import Rbf
    import os

    # 1. Fetch the dedicated fsaverage template (not the sample data)
    # This will return the path to the 'fsaverage' folder itself
    fs_dir = mne.datasets.fsaverage.data_path()

    # 2. Set subjects_dir to the parent directory of fsaverage
    # MNE functions expect subjects_dir to be the folder CONTAINING 'fsaverage'
    subjects_dir = os.path.dirname(fs_dir)
    subject = 'fsaverage'

    # 3. Define the surface path
    # This will now correctly point to .../mne_data/MNE-fsaverage-data/fsaverage/bem/outer_skin.surf
    surf_path = os.path.join(fs_dir, 'bem', 'outer_skin.surf')

    # 4. Load the surface
    head_surf = mne.read_surface(surf_path)
    vertices, faces = head_surf

    # Convert from mm to meters to match standard EEG coords if necessary
    # fsaverage surfaces are often in mm
    vertices /= 1000.0

    # 2. YOUR DATA (68 Channels)
    # Ensure x_elec, y_elec, z_elec are in meters
    x_elec = meta_data["chanlocs"]["X"].astype(float)
    y_elec = meta_data["chanlocs"]["Y"].astype(float)
    z_elec = meta_data["chanlocs"]["Z"].astype(float)
    activations = data
    # 3. INTERPOLATE ACTIVATIONS ONTO THE FSAVERAGE MESH
    # We train the RBF on your 68 electrode points
    rbf_func = Rbf(x_elec, y_elec, z_elec, activations, function='multiquadric', smooth=0.01)

    # We calculate the intensity for every vertex on the fsaverage head
    interpolated_values = rbf_func(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # 4. CREATE THE PLOTLY FIGURE
    fig = go.Figure()

    # Add the fsaverage Head Mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=interpolated_values,
        colorscale='Jet',
        opacity=1.0,
        name='fsaverage Scalp',
        showscale=True,
        colorbar=dict(title="Activation")
    ))

    # Add the original electrode markers
    fig.add_trace(go.Scatter3d(
        x=x_elec, y=y_elec, z=z_elec,
        mode='markers',
        marker=dict(size=4, color='white', line=dict(width=2, color='black')),
        name='Electrodes'
    ))

    # 5. VIEW SETTINGS
    # fsaverage is oriented so +Y is Nasion, +X is Right Ear
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))  # Perspective view
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()

def showme3DTopomap(activations,meta_data, title="3D EEG Topography"):
    """
    Plots an interactive 3D topographical map on the fsaverage head model.

    Parameters:
    activations: array-like, shape (n_channels,)
    x_elec, y_elec, z_elec: arrays, shape (n_channels,) in meters
    """
    import numpy as np
    import trimesh
    import plotly.graph_objects as go
    from scipy.interpolate import Rbf

    x_p = meta_data["chanlocs"]["X"].astype(float)
    y_p = meta_data["chanlocs"]["Y"].astype(float)
    z_p = meta_data["chanlocs"]["Z"].astype(float)

    #1. Create skull mesh
    mesh = trimesh.load(_ASSET_DIR / "skull_1.obj", force = 'mesh')
    vertices = mesh.vertices.copy()
    vertices[:,[1, 2]] = vertices[:,[2, 1]]
    faces = mesh.faces.copy()


    #2. Center skull mesh
    skull_center = vertices.mean(axis=0)
    print("skull_center_1: ", skull_center)
    # skull_center[2] = skull_center[]
    vertices = vertices - skull_center

    # vertices = vertices / np.max(np.linalg.norm(vertices, axis=1)) # normalizing skull scale

    # vertices[:, 1] *= -1 #IF YOU WANT TO FLIP F/B
    # vertices[:, 0] *= -1 #IF YOU WANT TO FLIP L/R
    # vertices[:, 2] *= -1 #IF YOU WANT TO FLIP U/D
    # elec_center = np.array([x_p.mean(), y_p.mean(), z_p.mean()])

    # --- Apply shift ---
    # Compute electrode radius
    elec_radius = np.mean(np.sqrt(x_p ** 2 + y_p ** 2 + z_p ** 2))
    x_p /= elec_radius
    y_p /= elec_radius
    z_p /= elec_radius

    # --- Scale electrodes to skull ---
    skull_radius = np.mean(np.linalg.norm(vertices, axis=1))
    scale_factor = skull_radius / elec_radius
    print("skull_factor: ", scale_factor)
    #
    x_p *= scale_factor
    y_p *= scale_factor
    z_p *= scale_factor

    # --- Compute shift ---
    # 2.1. Center electrode coordinates
    elec_center = np.array([
        x_p.mean(),
        y_p.mean(),
        z_p.mean()
    ])

    # marker_scaling = 2.02  # 80% outward
    marker_scaling =1.6
    shift = (skull_center - elec_center)
    print("skull_center: ", skull_center)
    print("elec_center: ", elec_center)

    x_p = marker_scaling*(x_p - shift[0])
    y_p = marker_scaling*(y_p - shift[1])
    z_p = marker_scaling*(z_p - shift[2])

    # 3. Interpolation
    # Map the 68 activation points to the thousands of vertices on the head mesh
    # 'smooth' helps prevent "spiky" look if one channel is noisy
    rbf_func = Rbf(
        x_p, y_p, z_p,
        activations,
        function='multiquadric',
        smooth=0.02
    )

    #interpolating values
    interp_values = rbf_func(
        -1*vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2]
    )

    # Shifting/scaling the chan markers to align with the skull
    # Change these values when if you are changing the 'skull.obj' file

    #settings for the head mesh
    # # x_p *= -1
    # y_p *= -1
    y_p -= scale_factor* .13
    # z_p += scale_factor*2
    # 6


    # Settings for the skull mesh
    # 4. Create Plotly Figure
    fig = go.Figure()

    # The Colored Head Mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=interp_values,
        colorscale='Jet',
        opacity=1.0,
        name='Head Surface',
        showscale=True,
        colorbar=dict(title="Activation", thickness=20)
    ))

    # The Electrode Markers
    fig.add_trace(go.Scatter3d(
        x=x_p, y=y_p, z=z_p,
        mode='markers',
        marker=dict(size=3, color='white', line=dict(width=1, color='black')),
        name='Electrodes'
    ))

    # 5. Scene Formatting
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.5)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        # width = 700,  # pixels
        height = 500  # pixels
    )
    return fig

def showme3DTopomapnewmesh(activations,meta_data, title="3D EEG Topography"):
    """
    Plots an interactive 3D topographical map on the fsaverage head model.

    Parameters:
    activations: array-like, shape (n_channels,)
    x_elec, y_elec, z_elec: arrays, shape (n_channels,) in meters
    """
    import numpy as np
    import trimesh
    import plotly.graph_objects as go
    from scipy.interpolate import Rbf

    x_p = meta_data["chanlocs"]["X"].astype(float)
    y_p = meta_data["chanlocs"]["Y"].astype(float)
    z_p = meta_data["chanlocs"]["Z"].astype(float)

    #1. Create skull mesh
    mesh = trimesh.load(_ASSET_DIR / "head_openneck.obj", force = 'mesh')
    vertices = mesh.vertices.copy()
    vertices[:,[1, 2]] = vertices[:,[2, 1]]
    faces = mesh.faces.copy()


    #2. Center skull mesh
    skull_center = vertices.mean(axis=0)
    vertices = vertices - skull_center
    # vertices = vertices / np.max(np.linalg.norm(vertices, axis=1)) # normalizing skull scale

    # vertices[:, 1] *= -1 #IF YOU WANT TO FLIP F/B
    # vertices[:, 0] *= -1 #IF YOU WANT TO FLIP L/R
    # vertices[:, 2] *= -1 #IF YOU WANT TO FLIP U/D
    # elec_center = np.array([x_p.mean(), y_p.mean(), z_p.mean()])

    # --- Apply shift ---
    # Compute electrode radius
    elec_radius = np.mean(np.sqrt(x_p ** 2 + y_p ** 2 + z_p ** 2))

    # --- Scale electrodes to skull ---
    skull_radius = np.mean(np.linalg.norm(vertices, axis=1))
    scale_factor = skull_radius / elec_radius
    print("skull_factor: ", scale_factor)
    #
    x_p *= scale_factor
    y_p *= -1*scale_factor
    z_p *= scale_factor
    # --- Compute shift ---
    # 2.1. Center electrode coordinates
    elec_center = np.array([
        x_p.mean(),
        y_p.mean(),
        z_p.mean()
    ])

    # marker_scaling = 2.02  # 80% outward
    marker_scaling =1.15
    shift = skull_center - elec_center
    print("skull_center: ", skull_center)
    x_p = marker_scaling*(x_p - shift[0])
    y_p = marker_scaling*(y_p - shift[1])
    z_p = marker_scaling*(z_p - shift[2])

    # 3. Interpolation
    # Map the 68 activation points to the thousands of vertices on the head mesh
    # 'smooth' helps prevent "spiky" look if one channel is noisy
    rbf_func = Rbf(
        x_p, y_p, z_p,
        activations,
        function='multiquadric',
        smooth=0.02
    )

    #interpolating values
    interp_values = rbf_func(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2]
    )

    # Shifting/scaling the chan markers to align with the skull
    # Change these values when if you are changing the 'skull.obj' file

    #settings for the head mesh
    # # x_p *= -1
    # y_p *= -1
    # y_p += .385
    # z_p += .3


    # Settings for the skull mesh
    y_p += .1*y_p
    # z_p +=.42
    # 4. Create Plotly Figure
    fig = go.Figure()

    # The Colored Head Mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=interp_values,
        colorscale='Jet',
        opacity=1.0,
        name='Head Surface',
        showscale=True,
        colorbar=dict(title="Activation", thickness=20)
    ))

    # The Electrode Markers
    fig.add_trace(go.Scatter3d(
        x=x_p, y=y_p, z=z_p,
        mode='markers',
        marker=dict(size=3, color='white', line=dict(width=1, color='black')),
        name='Electrodes'
    ))

    # Add a simple 'Nose' marker for orientation (at +Y)
    # fig.add_trace(go.Scatter3d(
    #     x=[0], y=[1.05], z=[0],
    #     mode='text'
    #     text=["FRONT"],
    #     textfont=dict(color="black", size=10),
    #     name='Orientation'
    # ))

    # 5. Scene Formatting
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.5)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        # width = 700,  # pixels
        height = 500  # pixels
    )
    return fig

def showme2DTopomap(activations, meta_data, titlestr="2D EEG Topomap"):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import Rbf

    if titlestr is None:
        titlestr = "2D EEG Topomap"

    # --- Load electrode positions ---
    x = np.array(meta_data["chanlocs"]["X"], dtype=float)
    y = np.array(meta_data["chanlocs"]["Y"], dtype=float)
    z = np.array(meta_data["chanlocs"]["Z"], dtype=float)
    labels = meta_data["chanlocs"]["labels"]

    # Filter out external/unwanted channels
    indices = [i for i, val in enumerate(labels) if val.upper() not in ['EXG7', 'EXG8']]

    x = x[indices]
    y = y[indices]
    z = z[indices]

    # Also filter the activations to match the channel count
    activations = np.array(activations)[indices]

    # --- Normalize to unit sphere ---
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x = x / r
    y = y / r

    # --- Interpolation ---
    # Fit the Rbf model on actual data points (x, y) and their values (activations)
    rbf = Rbf(-1*x, y, activations, function='multiquadric', smooth=0.02)

    #adding this so that the channel markers extend beyond the head circle, which will depict a more accurate picture
    y = (y-.06)*1.05
    x = x*1.05

    # Generate grid
    grid_x, grid_y = np.mgrid[-1.3:1.3:300j, -1.3:1.3:300j]

    # Evaluating the fitted model on the grid
    grid_z = rbf(grid_x, grid_y)

    # Mask everything outside the head boundary (radius = 1.0)
    mask = (grid_x) ** 2 + (grid_y+.06) ** 2 > 1.05 ** 2 #Off center masking to include the bottom of the skull, not displayed in the top view
    grid_z[mask] = np.nan

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw contour on layer zorder=1
    im = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap='jet', zorder=1)

    # --- Vector Boundary Clipping (CLEAN EDGES) ---
    # Creates a perfect geometric circle matching your head outline (radius=1)
    # clip_circle = plt.Circle((0, 0), 1.0, transform=ax.transData)
    # im.set_clip_path(clip_circle)

    # --- Head outline (zorder=4) ---
    head = plt.Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=2, zorder=4)
    ax.add_patch(head)

    # --- Ears (zorder=4) ---
    ear_left = plt.Circle((-1.02, 0), 0.08, edgecolor='black', facecolor='none', linewidth=2, zorder=4)
    ear_right = plt.Circle((1.02, 0), 0.08, edgecolor='black', facecolor='none', linewidth=2, zorder=4)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)

    # --- Nose (zorder=4) ---
    nose_x = [0, -0.08, 0.08, 0]
    nose_y = [1.08, 1.0, 1.0, 1.08]
    ax.plot(nose_x, nose_y, color='black', linewidth=2, zorder=4)

    # --- Electrodes (zorder=5) ---
    ax.scatter(x, y, c='black', s=10, zorder=5)

    # --- Grid and Axis Configuration (ADDED) ---
    # ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axis_on()

    # Grid ticks spanning across the boundary limits
    # ax.set_xticks(np.arange(-1.2, 1.3, 0.4))
    # ax.set_yticks(np.arange(-1.2, 1.3, 0.4))

    # Clean up outer spine boxes
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Formatting ---
    ax.set_aspect('equal')
    ax.set_title(titlestr)

    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.show()

def showmeICAoverlayedondata(npy_data, ica_data ,labels, chid = None, subjid = None):
    # Tkinter build: the ipywidgets slider is replaced by a Tk scale, the plot is
    # the same one drawn by the Jupyter build.
    import numpy as np
    import _tk_helpers as tkh
    import tkinter as tk
    from tkinter import ttk

    if chid is None:
        chid = np.where(labels == 'Fp1')[0]

    chname = labels[chid]
    signal = npy_data[:, chid]

    def normalize(x): return (x - np.mean(x)) / np.std(x)

    signal_norm = normalize(signal)

    win = tkh.PlotWindow(f"ICA overlay {subjid}" if subjid is not None else "ICA overlay")

    ica_id = tk.IntVar(value=0)

    def overlay_data(*_):
        ax = win.clear()
        ICA_id = int(ica_id.get())
        ICA_norm = normalize(ica_data[ICA_id, :])

        ax.plot(signal_norm, label=f"EEG Channel {chname}", color='blue')
        ax.plot(ICA_norm, label=f"ICA Component {ICA_id}", color='red', alpha=0.7)
        ax.set_title(f"ICA {ICA_id} vs Channel {chname} for {subjid}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized Amplitude")
        ax.legend([f"ICA component {ICA_id}"], loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        win.draw()

    ttk.Label(win.controls, text="ICA component").pack(anchor="w")
    tk.Scale(
        win.controls,
        from_=0,
        to=ica_data.shape[0] - 1,
        orient=tk.HORIZONTAL,
        variable=ica_id,
        command=overlay_data,
        length=200,
    ).pack(anchor="w")
    ttk.Button(win.controls, text="Close", command=win.win.destroy).pack(anchor="w", pady=(12, 0))

    overlay_data()
    win.wait()

def showmeSTFTSpectrogram(npy_data, chid, fs = 256 , titlestr = None,freqlim = None, binsize = None,isoverlap = None):
    # create a spectrogram of the channel activations for all time.
    # input the npy_data and channel id, function assumes sampling frequency is 256 Hz

    # 256 sample STFT with
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    import numpy as np
    if titlestr == None:
        titlestr = f' STFT Spectrogram for channel {chid}'

    if isoverlap == None:
        isoverlap = False

    if freqlim == None:
        freqlim = [0,50]

    if binsize == None:
        binsize = 512

    chsignal = npy_data[:, chid]
    if isoverlap:
        f, t, Sxx = signal.spectrogram(chsignal, fs,nperseg=binsize,noverlap=binsize/2)
    else:
        f, t, Sxx = signal.spectrogram(chsignal, fs,nperseg=binsize,noverlap=0)

    freqidx = np.where((f>=freqlim[0]) & (f<=freqlim[1]))[0]
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f[freqidx], 10 * np.log10(Sxx[freqidx,:]) , cmap='RdBu_r')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [nsamples]')
    plt.title(titlestr)
    plt.colorbar(label='Power[dB]')
    plt.show()
    return f,t

def showmeMWSpectrogram(npy_data, chid, epochid , fs = 256 , titlestr = None, freqlim = None, binsize = None,isoverlap = None):
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    if freqlim == None:
        freqlim = [0,50]
    if titlestr == None:
        titlestr = f' MW Spectrogram for channel {chid}'
    if len(npy_data.shape) == 2:
        epochid = 0

    signal = np.moveaxis(npy_data, 0, -1)
    signalOI = signal[chid, epochid, :]
    freqs = np.linspace(1, 60, 100)

    # More cycles = better frequency resolution (especially low freq)
    n_cycles = freqs / 2.0

    power = mne.time_frequency.tfr_array_morlet(
        signalOI[np.newaxis, np.newaxis, :],  # shape: (channels, epochs, time)
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power'
    )
    power = power[0, 0]  # remove extra dims
    # Time axis
    times = np.arange(signalOI.shape[0]) / fs

    # Plot
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, freqs, 10 * np.log10(power),
                   shading='gouraud', cmap='RdBu_r')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if titlestr == None:
        plt.title("Morlet Wavelet Spectrogram")
    else:
        plt.title(titlestr)

    plt.colorbar(label="Power (dB)")
    plt.ylim(1, 50)
    plt.show()
#
# def createSkullmesh():
#     import open3d as o3d
#     import mcubes
#     import nrrd
#     import numpy as np
#
#     # 1. Load your CT scan (.nrrd file)
#     data, header = nrrd.read('skull.nrrd')
#
#     # 2. Thresholding (Isolate bone)
#     # Bone density is typically between 200-1000+ HU
#     binary_mask = data > 300
#
#     # 3. Running Marching Cubes to extract mesh
#     vertices, faces = mcubes.marching_cubes(binary_mask, 0)
#
#     # 4. Create and save mesh
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     mesh.triangles = o3d.utility.Vector3iVector(faces)
#
#     # Clean up the mesh
#     mesh.remove_duplicated_vertices()
#     mesh.remove_degenerate_triangles()
#
#     o3d.io.write_triangle_mesh('skull_mesh.stl', mesh)
#     print("Mesh saved!")
#     return(mesh)

