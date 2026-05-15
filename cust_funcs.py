# Hello Bhargav, welcome to your sandbox
def rebrand_lw6data(filepath):
    """ Rewrite the lw6_data from matlab style encryption to python style encryption to make life easier in python

    :param filepath: enter the full filepath starting from the drive it is stored in.
    :return: The newly created lw6_data, with the name meta_data
    """
    import numpy as np
    import scipy.io as sci
    meta_data = sci.loadmat(filepath)
    #
    # keys_new = meta_data["header"].dtype.names
    # keys_new.extend(["fs","activation"])

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

    allevents = meta_data["header"]["events"][0][0][0]
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
        "name": "Name of the file as stored in the lw6_data",
        "tags": "I have no idea, but this is as it is stored in the lw6_data",
        "history": "Record of all changes made to the data as operation stored as suffix: history till then",
        "origins": "Origins of the data, contains gui info and original filepath",
        "datasize": "size of the data",
        "xstart": "I have no idea what this is, but this was present in the original lw6_data",
        "ystart": "I have no idea what this is, but this was present in the original lw6_data",
        "zstart": "I have no idea what this is, but this was present in the original lw6_data",
        "xstep": "I have no idea what this is, but this was present in the original lw6_data",
        "ystep": "I have no idea what this is, but this was present in the original lw6_data",
        "zstep": "I have no idea what this is, but this was present in the original lw6_data",
        "chanlocs": "channel location data, stored as follows",
        "chanlocs.labels": "Channel labels stored as strings",
        "chanlocs.SEEG_enabled": "I have no idea what this is, but was present in the original lw6_data",
        "chanlocs.TOPO_enabled": "I have no idea what this is, but was present in the original lw6_data",
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
    return new_data

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

def openMetadata(filepath):
    import pickle
    with open(filepath, "rb") as f:
        meta_data = pickle.load(f)
    return meta_data

def saveMetadata(meta_data, metafilepath):
    import pickle
    with open(metafilepath, "wb") as f:
        pickle.dump(meta_data, f)

def updatemetadataHistory(meta_data,step_prefix):
    if step_prefix is None:
        raise ValueError(f"Write a step prefix")

    current_state = {key: value for key, value in meta_data.items() if key != "history"}
    meta_data["history"][step_prefix] = current_state
    return meta_data

def showmeFFT(signal,fs,freqlim=None,pltsize=None,label=None,titlestr=None,epochid=None):
    import numpy as np
    import matplotlib.pyplot as plt
    if epochid is None:
        epochid = 0

    if len(signal.shape) > 1:
        signal = np.squeeze(signal)

    if titlestr is None:
        titlestr = [f"FFT of signal for {label} in epoch {epochid}"]

    if freqlim is None:
        freqlim = [0, fs // 2]

    if pltsize is None:
        pltsize = [12, 3]

    # Converting signal from sample to freq domain
    FFT_vals=np.abs(np.fft.fft(signal))
    freqs=np.fft.fftfreq(len(FFT_vals),1/fs)
    idx = (freqs >= freqlim[0]) & (freqs <= freqlim[1])

    #actual plotting
    plt.figure(figsize=pltsize)
    plt.stem(freqs[idx], FFT_vals[idx],lw=1)
    plt.title(titlestr)
    plt.xlabel("Freq [Hz]")
    ticks = np.arange(1.2*(freqlim[0]//1.2), freqlim[1] + 1.2, 1.2)  # include endpoint
    plt.xticks(ticks,fontsize=8)
    plt.ylabel("abs Amp")
    legendstr = [f"epoch:{epochid}"]
    plt.legend(legendstr,loc="upper right")
    plt.title(titlestr)
    plt.show()

def showmeSignal(signal,duration=None,pltsize=None,titlestr=None,epochid=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if len(signal.shape)>1:
        signal = np.squeeze(signal)

    if epochid is None:
        epochid = 0

    if duration is None:
        duration = np.linspace(0,len(signal)-1,len(signal),dtype=int)

    if pltsize is None:
        pltsize = [12, 4]

    if titlestr is None:
        titlestr = "Channel activation"

    #actual plotting
    plt.figure(figsize=pltsize)
    plt.plot(duration,signal,lw=1)
    plt.title(titlestr)
    plt.xlabel("Samples")
    plt.ylabel("amplitude")
    legendstr = [f"epoch:{epochid}"]
    if len(signal.shape) >1:
        legend_labels = [f"line {i}" for i in range(signal.shape[1])]

    if len(signal.shape)==1:
        plt.legend(legendstr,loc="upper right")
    else:
        plt.legend(legend_labels,loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.75)

    plt.figure
    plt.show()

# def showmeInterpSignal(signal,duration=None,pltsize=None,titlestr=None,epochid=None,legendlabels=None): #under development
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     if legendlabels is None:
#         legendlabels = [f"epochid:{epochid}"]
#
#     if len(signal.shape)>1:
#         signal = np.squeeze(signal)
#
#     if epochid is None:
#         epochid = 0
#
#     if duration is None:
#         duration = np.linspace(0,len(signal)-1,len(signal),dtype=int)
#
#     if pltsize is None:
#         pltsize = [12, 3]
#
#     if titlestr is None:
#         titlestr = f"Channel activation, epoch:{epochid}"
#
#     #actual plotting
#     plt.figure(figsize=pltsize)
#     plt.plot(duration,signal,lw=1)
#     plt.title(titlestr)
#     plt.xlabel("Samples")
#     plt.ylabel("amplitude")
#     if signal.shape[1] >1:
#         legend_labels = [f"{legendlabels[i]}" for i in range(signal.shape[1])]
#
#     if signal.shape[1]==1:
#         plt.legend(legendstr,loc="upper right")
#     else:
#         plt.legend(legend_labels,loc="upper right")
#     plt.show()

def showmeSummaryPlot(signal,fs,chname,subjid,pltsize=None,freqlim=None,epochid=None):
    import numpy as np
    import matplotlib.pyplot as plt
    if len(signal.shape) > 1:
        signal = np.squeeze(signal)

    if epochid is None:
        epochid = 0

    if freqlim is None:
        freqlim = [0.1, 20]

    if pltsize is None:
        pltsize = [12, 3]

    sigtitlestr = f"amp vs samples for sub:{subjid} chid:{chname}"
    ffttitlestr = f"FFT spectrum for sub:{subjid} chid:{chname}"

    showmeSignal(signal,titlestr=sigtitlestr,epochid=epochid,pltsize=pltsize)
    showmeFFT(signal,fs,freqlim=freqlim,pltsize=pltsize,titlestr=ffttitlestr,epochid=epochid)

def zscoreChunks(FFT, freqs, f0=None, window=None, exclude_bins=1, buffer=None): #funciton under construction
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

def givemeNNearestNeighbour(lw6_data,chnameid):
    import numpy as np
    labels=lw6_data["chanlocs"]["labels"]
    labels = labels.astype(str)
    if isinstance(chnameid, str):
        chid = np.squeeze(np.where(chnameid==labels))
    else:
        chid=chnameid

    #shaping the data
    X=lw6_data["chanlocs"]["X"]
    X=X.astype(float)
    Y=lw6_data["chanlocs"]["Y"]
    Y=Y.astype(float)
    Z=lw6_data["chanlocs"]["Z"]
    Z=Z.astype(float)
    coords = np.column_stack((X, Y, Z))

    #calculating distances
    ref = coords[chid]
    distances = np.linalg.norm(coords-ref,axis=1)

    #Sorting the distances
    sorted_idx = np.argsort(distances, kind='stable')
    sorted_labels = labels[sorted_idx]

    return sorted_idx[1:],sorted_labels[1:]

def showmeSignalUI(mat_data,lw6_data,xlim=None,ylim=None):
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib.pyplot as plt
    import numpy as np

    epoch = [f"epoch{i}" for i in range(mat_data.shape[2])]
    labels = np.squeeze(lw6_data["chanlocs"]["labels"])
    labels = labels.astype(str)

    # 1. Generate Dummy Data (40 Epochs, 60 Series, 100 Timepoints)
    # Shape: (Epochs, Series, Time)

    data_matrix = np.moveaxis(mat_data,0,-1) #moves first axis to the last
    print(data_matrix.shape)
    # 2. The Plotting Function
    def plot_data(epoch_idx, selected_series):
        plt.figure(figsize=(9, 5))

        if not selected_series:
            plt.text(0.5, 0.5, "Select series from the list", ha='center', va='center')
        else:
            for s_idx in selected_series:
                series_data = data_matrix[s_idx, epoch_idx,  :]
                plt.plot(series_data, label=labels[s_idx])

            # plt.title(f"Epoch {epoch_idx} | Comparing {len(selected_series)} Series")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True, linestyle='--', alpha=0.6)
            if xlim is not None:
                plt.xlim(-abs(xlim), abs(xlim))
            if ylim is not None:
                plt.ylim(-abs(ylim),abs(ylim))

        plt.show()

    # 3. The Widgets
    # Slider for "scrolling" through 40 epochs
    epoch_slider = widgets.IntSlider(
        value=0, min=0, max=mat_data.shape[2]-1, step=1,
        description='Epoch:',
        continuous_update=True,  # Update plot while dragging
        layout={'width': '1000px'}
    )

    # Multi-select for the 60 series
    series_selector = widgets.SelectMultiple(
        options=[(labels[i], i) for i in range(mat_data.shape[1])],
        value=[0, 1, 2, 3],  # Default selection (first 4)
        description='Channel:',
        layout={'height': '550px', 'width': '150px'}
    )

    # 4. Linking it all together
    out = widgets.interactive_output(
        plot_data,
        {'epoch_idx': epoch_slider, 'selected_series': series_selector}
    )

    # 5. Organizing the Layout
    # Putting the epoch slider on top and the list next to the plot
    ui = widgets.VBox([
        epoch_slider,
        widgets.HBox([series_selector, out])
    ])

    display(ui)

def showmeSignalUIandInterp(mat_data, lw6_data, xlim=None, ylim=None):
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib.pyplot as plt
    import numpy as np
    # Setup data and labels
    labels = np.squeeze(lw6_data["chanlocs"]["labels"]).astype(str)
    data_matrix = np.moveaxis(mat_data, 0, -1)

    # Storage for different categories of bad channels
    to_interpolate = []
    bad_but_ignore = []

    # 1. Widgets
    epoch_slider = widgets.IntSlider(
        value=0, min=0, max=mat_data.shape[2] - 1,
        description='Epoch:', layout={'width': '800px'}
    )

    series_selector = widgets.SelectMultiple(
        options=[(labels[i], i) for i in range(mat_data.shape[1])],
        value=[0, 1, 2, 3],
        description='Channels:', layout={'height': '400px', 'width': '200px'}
    )

    # UI Buttons and Labels
    interp_btn = widgets.Button(description="Interpolate", button_style='danger', icon='magic')
    bad_btn = widgets.Button(description="Mark Bad (No Interp)", button_style='warning', icon='ban')
    done_btn = widgets.Button(description="Finalize & Close", button_style='success', icon='check')

    status_html = widgets.HTML(value="""
        <div style='line-height: 1.5;'>
            <b>Interpolate:</b> <span id='interp_list'>[]</span><br>
            <b>Bad (Ignore):</b> <span id='bad_list'>[]</span>
        </div>
    """)

    # 2. Plotting Logic
    def plot_data(epoch_idx, selected_series):
        plt.figure(figsize=(10, 5))
        if not selected_series:
            plt.text(0.5, 0.5, "Select channels to view", ha='center', va='center')
        else:
            for s_idx in selected_series:
                # Visual cue: Dashed for bad-no-interp, thick for interpolate
                style = '-'
                alpha = 1.0
                if s_idx in to_interpolate:
                    style = '-'  # Solid but maybe specific color
                elif s_idx in bad_but_ignore:
                    style = '--'  # Dashed line for "just bad"
                    alpha = 0.6

                plt.plot(data_matrix[s_idx, epoch_idx, :], label=labels[s_idx], linestyle=style, alpha=alpha)

        plt.title(f"Epoch {epoch_idx}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.show()

    out = widgets.interactive_output(plot_data, {'epoch_idx': epoch_slider, 'selected_series': series_selector})

    # 3. Button Functionality
    def update_status():
        status_html.value = f"""
            <div style='line-height: 1.5;'>
                <b style='color:red;'>Interpolate:</b> {[labels[i] for i in sorted(to_interpolate)]}<br>
                <b style='color:orange;'>Bad (Ignore):</b> {[labels[i] for i in sorted(bad_but_ignore)]}
            </div>
        """

    def on_interp_clicked(b):
        for idx in series_selector.value:
            if idx not in to_interpolate: to_interpolate.append(idx)
            if idx in bad_but_ignore: bad_but_ignore.remove(idx)  # Ensure it's only in one list
        update_status()

    def on_bad_clicked(b):
        for idx in series_selector.value:
            if idx not in bad_but_ignore: bad_but_ignore.append(idx)
            if idx in to_interpolate: to_interpolate.remove(idx)  # Ensure it's only in one list
        update_status()

    def on_done_clicked(b):
        ui.close()
        out.close()
        print("--- FINAL CHANNEL REPORT ---")
        print(f"To Interpolate: {sorted(to_interpolate)} ({[labels[i] for i in sorted(to_interpolate)]})")
        print(f"Bad (Ignore):   {sorted(bad_but_ignore)} ({[labels[i] for i in sorted(bad_but_ignore)]})")

    interp_btn.on_click(on_interp_clicked)
    bad_btn.on_click(on_bad_clicked)
    done_btn.on_click(on_done_clicked)

    # 4. Layout
    button_box = widgets.VBox([interp_btn, bad_btn, status_html, done_btn])
    ui = widgets.VBox([
        epoch_slider,
        widgets.HBox([series_selector, out, button_box])
    ])

    display(ui)

    # Returning both lists so you can handle them differently in your pipeline
    badch = np.union1d(to_interpolate, bad_but_ignore)
    for i in range(len(to_interpolate)):
        srt_idx,srt_labels = givemeNNearestNeighbour(lw6_data, to_interpolate[i])
        srtd_idx = np.where(~np.isin(srt_labels, badch))[0]
        srtd_idx = srtd_idx[:3]
        badch = np.append(badch, srt_labels[srtd_idx])
        mat_data[:,to_interpolate[i],:] = np.mean(mat_data[:,srtd_idx,:], axis=1)


    return mat_data,{"interpolate": to_interpolate, "bad_no_interp": bad_but_ignore}

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

def showmeTopomap(data,lw6_data):

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
    x_elec = lw6_data["chanlocs"]["X"].astype(float)
    y_elec = lw6_data["chanlocs"]["Y"].astype(float)
    z_elec = lw6_data["chanlocs"]["Z"].astype(float)
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

def showme3DTopomap(activations,lw6_data, title="3D EEG Topography"):
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
    import os
    from scipy.spatial import cKDTree

    x_elec = lw6_data["chanlocs"]["X"].astype(float)
    y_elec = lw6_data["chanlocs"]["Y"].astype(float)
    z_elec = lw6_data["chanlocs"]["Z"].astype(float)

    #1. Create skull mesh
    mesh = trimesh.load('skull.obj', force='mesh')
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()


    #2. Center skull mesh
    skull_center = vertices.mean(axis=0)
    vertices = vertices - skull_center
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1)) # normalizing skull scale
    # vertices[:, 1] *= -1 #IF YOU WANT TO FLIP F/B
    # vertices[:, 0] *= -1 #IF YOU WANT TO FLIP L/R
    # vertices[:, 2] *= -1 #IF YOU WANT TO FLIP U/D
    # elec_center = np.array([x_p.mean(), y_p.mean(), z_p.mean()])

    # --- Apply shift ---

    # print(shift)
    x_p = x_elec
    y_p = y_elec
    z_p = z_elec

    # Compute electrode radius
    elec_radius = np.mean(np.sqrt(x_p ** 2 + y_p ** 2 + z_p ** 2))

    # --- Scale electrodes to skull ---
    skull_radius = np.mean(np.linalg.norm(vertices, axis=1))
    scale_factor = skull_radius / elec_radius

    x_p *= scale_factor
    y_p *= scale_factor
    z_p *= scale_factor
    # --- Compute shift ---
    # 2.1. Center electrode coordinates
    elec_center = np.array([
        x_elec.mean(),
        y_elec.mean(),
        z_elec.mean()
    ])
    marker_scaling = 1.72  # 80% outward

    x_p = (x_p - elec_center[0]) * marker_scaling + elec_center[0]
    y_p = (y_p - elec_center[1]) * marker_scaling + elec_center[1]
    z_p = (z_p - elec_center[2]) * marker_scaling + elec_center[2]
    shift = skull_center - elec_center
    # print(shift)
    # x_p += shift[0]
    # x_p *= -1
    # x_p -= shift[0]

    # 3. Interpolation
    # Map the 68 activation points to the thousands of vertices on the head mesh
    # 'smooth' helps prevent "spiky" look if one channel is noisy
    rbf_func = Rbf(
        x_p, y_p, z_p,
        activations,
        function='multiquadric',
        smooth=0.02
    )

    # Shifting/scaling the chan markers to align with the skull
    # Change these values when if you are changing the 'skull.obj' file
    y_p += -.385
    y_p *= -1
    z_p += .3

    #interpolating values
    interp_values = rbf_func(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2]
    )

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
    #     mode='text',
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

def showme2DTopomap(activations, lw6_data, titlestr="2D EEG Topomap"):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import Rbf

    if titlestr == None:
        titlestr = "2D EEG Topomap"
    # --- Load electrode positions ---

    x = lw6_data["chanlocs"]["X"].astype(float)
    y = lw6_data["chanlocs"]["Y"].astype(float)
    z = lw6_data["chanlocs"]["Z"].astype(float)

    # --- Normalize to unit sphere ---
    r = np.sqrt(x**2 + y**2 + z**2)
    x = x / r
    y = y / r

    # --- Match your 3D orientation (IMPORTANT) ---
    y *= -1   # same flip you used in 3D

    # --- Interpolation ---
    rbf = Rbf(x, y, activations, function='multiquadric', smooth=0.02)

    grid_x, grid_y = np.mgrid[-1.3:1.3:300j, -1.3:1.3:300j]
    # grid_x, grid_y = np.mgrid[-1:1:300j, -1:1:300j]
    grid_z = rbf(grid_x, grid_y)

    # Optional: softer mask instead of hard cutoff
    mask = grid_x ** 2 + grid_y ** 2 > 1.3 ** 2
    grid_z[mask] = np.nan

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap='jet')

    # --- Head outline ---
    head = plt.Circle((0,0), 1, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(head)

    # --- Ears ---
    ear_left = plt.Circle((-1.05, 0), 0.08, edgecolor='black', facecolor='none', linewidth=2)
    ear_right = plt.Circle((1.05, 0), 0.08, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)

    # --- Nose ---
    nose_x = [0 , -0.08, 0.08, 0]
    nose_y = [1.0, 1.08, 1.08, 1.0]
    ax.plot(nose_x, nose_y, color='black', linewidth=2)

    # --- Electrodes ---
    ax.scatter(x, y, c='black', s=10, zorder=3)

    # --- Formatting ---
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(titlestr)

    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.show()

def showmeICAoverlayedondata(mat_data, ica_data ,labels, chid = None):
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib.pyplot as plt
    import numpy as np

    if chid  == None:
        chid = np.where(labels == 'Fp1')[0]

    chname = labels[chid]
    # EEG signal (one channel)
    # mat_data = np.moveaxis(mat_data,0,-1) #moves first axis to the last
    signal = mat_data[ :, chid ]

    # Normalize function (important for visual comparison)
    def normalize(x): return (x - np.mean(x)) / np.std(x)

    signal_norm = normalize(signal) #Normalize the signal

    def overlay_data(ICA_id):
        plt.figure(figsize=(10, 5))

        # ICA component
        ICA_signal = ica_data[ICA_id, :]
        ICA_norm = normalize(ICA_signal)

        # Plot
        plt.plot(signal_norm, label=f"EEG Channel {chname}", color='blue')
        plt.plot(ICA_norm, label=f"ICA Component {ICA_id}", color='red', alpha=0.7)

        plt.title(f"ICA {ICA_id} vs Channel {chname}")
        plt.xlabel("Time")
        plt.ylabel("Normalized Amplitude")
        legendstr = [f"ICA component {ICA_id}"]
        plt.legend(legendstr, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.show()

    # Slider
    ICA_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=ica_data.shape[0] - 1,
        step=1,
        description='ICA Comp:',
        continuous_update=True,
        layout={'width': '1000px'}
    )

    out = widgets.interactive_output(
        overlay_data,
        {'ICA_id': ICA_slider}
    )

    display(widgets.VBox([ICA_slider, out]))

def showmeICAoverlayedondataWithreturn(mat_data, ica_data, ch_id = None):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np

    if ch_id == None:
        ch_id = 0

    # 1. Setup Data

    signal = mat_data[ch_id, :]

    def normalize(x): return (x - np.mean(x)) / np.std(x)

    signal_norm = normalize(signal)

    # This list will hold the ICA IDs you decide to remove
    ids_to_remove = []

    # 2. Define UI Elements
    ICA_slider = widgets.IntSlider(
        value=0, min=0, max=ica_data.shape[0] - 1,
        description='ICA Comp:', layout={'width': '500px'}
    )

    add_btn = widgets.Button(description="Add to Removal List", button_style='warning')
    done_btn = widgets.Button(description="Finalize & Stop", button_style='success')
    status_label = widgets.Label(value="Selected IDs: []")

    # 3. Plotting Logic
    def overlay_data(ICA_id):
        plt.figure(figsize=(10, 4))
        ICA_signal = ica_data[ICA_id, :]
        ICA_norm = normalize(ICA_signal)

        plt.plot(signal_norm, label=f"EEG Channel {ch_id}", color='blue', alpha=0.5)
        plt.plot(ICA_norm, label=f"ICA Component {ICA_id}", color='red')
        plt.title(f"Comparing ICA {ICA_id} to Channel {ch_id}")
        plt.legend()
        plt.show()

    out = widgets.interactive_output(overlay_data, {'ICA_id': ICA_slider})

    # 4. Button Logic
    def on_add_clicked(b):
        current_id = ICA_slider.value
        if current_id not in ids_to_remove:
            ids_to_remove.append(current_id)
            status_label.value = f"Selected IDs: {sorted(ids_to_remove)}"

    def on_done_clicked(b):
        # Stop displaying the plot and UI
        ui_container.close()
        out.close()
        print(f"Final List of ICA components to remove: {sorted(ids_to_remove)}")
        # You can now use 'ids_to_remove' for the next step of your analysis

    add_btn.on_click(on_add_clicked)
    done_btn.on_click(on_done_clicked)

    # 5. Display
    ui_container = widgets.VBox([
        ICA_slider,
        widgets.HBox([add_btn, done_btn]),
        status_label
    ])
    display(ui_container, out)

    # Note: In Jupyter, this function returns immediately.
    # The 'ids_to_remove' list will be populated as you click.
    return ids_to_remove

def showmeSTFTSpectrogram(mat_data, chid, fs = 256 , titlestr = None):
    # create a spectrogram of the channel activations for all time.
    # input the mat_data and channel id, function assumes sampling frequency is 256 Hz

    # 256 sample STFT with
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    import numpy as np
    # %matplotlib widgets #keeping the plot
    if titlestr == None:
        titlestr = f' STFT Spectrogram for channel {chid}'

    chsignal = mat_data[ :, chid]
    f, t, Sxx = signal.spectrogram(chsignal, fs)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud' , cmap='RdBu_r')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [nsamples]')
    plt.title(titlestr)
    plt.colorbar(label='Power[dB]')
    plt.show()

def showmeMWSpectrogram(mat_data, chid, epochid , fs = 256 , titlestr = None, freqlim = None):
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    if freqlim == None:
        freqlim = [0,50]
    if titlestr == None:
        titlestr = f' MW Spectrogram for channel {chid}'
    signal = np.moveaxis(mat_data, 0, -1)
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
    plt.title("Morlet Wavelet Spectrogram")
    plt.colorbar(label="Power (dB)")
    plt.ylim(1, 50)
    plt.show()

def createSkullmesh():
    import open3d as o3d
    import mcubes
    import nrrd
    import numpy as np

    # 1. Load your CT scan (.nrrd file)
    data, header = nrrd.read('skull.nrrd')

    # 2. Thresholding (Isolate bone)
    # Bone density is typically between 200-1000+ HU
    binary_mask = data > 300

    # 3. Running Marching Cubes to extract mesh
    vertices, faces = mcubes.marching_cubes(binary_mask, 0)

    # 4. Create and save mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Clean up the mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    o3d.io.write_triangle_mesh('skull_mesh.stl', mesh)
    print("Mesh saved!")
    return(mesh)

def loadData(filepath):
    from pathlib import Path, PurePath
    import numpy as np

    if not isinstance(filepath, PurePath):
        filepath = Path(filepath)

    matfilepath = filepath.with_suffix('.mat')
    lw6filepath = filepath.with_suffix('.lw6')
    mat_data = np.load(filepath)