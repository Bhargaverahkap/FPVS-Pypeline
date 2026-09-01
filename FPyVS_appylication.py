# Hello, this script is in service of the resulting (GUI based) application that will be eventually made.
# Add all steps/functionalities here as a function with no dependencies on FPVS_pycage

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import mne
from mne.preprocessing import ICA
import importlib
import cust_funcs as cf
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from functools import partial
import matplotlib.ticker as ticker
importlib.reload(cf)

def loadalldata(filepath):
    metafilepath = filepath.with_suffix(".pkl")
    npyfilepath = filepath.with_suffix(".npy")
    meta_data = cf.loadMetadata(metafilepath)
    npy_data = np.load(npyfilepath)
    return npy_data, meta_data, npyfilepath, metafilepath

def savealldata(npy_data, meta_data, filepath):
    npyfilepath = filepath.with_suffix(".npy")
    metafilepath = filepath.with_suffix(".pkl")
    cf.saveMetadata(meta_data, metafilepath)
    np.save(npyfilepath, npy_data)
    
def extractdatafrombdf(filepath,summarychid = None):
    if filepath.suffix != ".bdf":
        filepath = filepath.with_suffix(".bdf")

    raw = mne.io.read_raw_bdf(filepath, preload=True)

    py_data, times = raw.get_data(return_times = True)
    status_data = raw.get_data(picks=["Status"])[0]

    # 3. Strip the baseline offset (Bit 16 shift) using bitwise operations
    # This isolates the actual 8-bit or 16-bit parallel port lines
    corrected_status = np.bitwise_and(status_data.astype(int), 255)


    # 4. Inject the clean triggers back into a temporary raw copy
    raw_corrected = raw.copy()
    raw_corrected._data[raw.ch_names.index("Status")] = corrected_status

    # 5. Extract events from the cleaned channel
    events_all = mne.find_events(raw_corrected, stim_channel="Status", shortest_event=1)
    fs = raw_corrected.info['sfreq']

    org_data = {
        "gui_info": ['no gui, no info'],
        "originalfilepath": filepath,
    }
    chanlocs = {
        "labels": raw_corrected.ch_names,
        "topo_enabled": np.zeros([len(raw_corrected.ch_names)],bool = True),
        "SEEG_enabled": np.zeros([len(raw_corrected.ch_names)],bool = True),
    }

    events = {
        "code": [events_all[i][2] for i in len(events_all)],
        "sample": [events_all[i][0] for i in len(events_all)],
        "latency": [events_all[i][0]/fs for i in len(events_all)]
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
    }
    #There are some values that are hardcoded, line the *step and *start, i dont know what they are for, so keeping it as is, ask gloria/cedric
    meta_data = {
        "filetype": 'time_amplitude',
        "name": filepath.stem,
        "tags": {},
        "history": {},
        "origins": org_data,
        "datasize": py_data.shape,
        "xstart": 0.0,
        "ystart": 0.0,
        "zstart": 0.0,
        "xstep": 0.00048828125,
        "ystep": 1,
        "zstep": 1,
        "chanlocs": chanlocs,
        "events": events,
        "fs": fs,
        "fields": fields
    }
    savealldata(py_data,meta_data,filepath)

    print("the keys in the new data are: ", meta_data.keys())
    
def convertMATtoPY(filepath):
    """
     This is the script to convert .mat and .lw6 files to .npy and .pkl files to better access and store files in the python environment
    """
    matfilepath = filepath.with_suffix('.mat')
    lw6filepath = filepath.with_suffix('.lw6') 
    
    with h5py.File(matfilepath, 'r') as f:
        # List all variables
        mat_data = f['data'][:]

    # This section cleans the data and meta data to make it more python friendly
    npy_data = np.squeeze(mat_data)
    
    meta_data = cf.rebrand_lw6data(lw6filepath)
    #renaming the mat/lw6 filenames for saving the intermediary steps.
    metafilepath = lw6filepath.with_suffix(".pkl")
    npyfilepath = matfilepath.with_suffix(".npy")
    savealldata(npy_data, meta_data, filepath)
    print("data is saved as: ", metafilepath)
    print("\n")
    
    return npy_data, meta_data, filepath

def renamechannels(filepath,stepno, targets = None, replacements = None):

    if targets == None:
        targets = ["EXG1", "EXG2", "EXG3", "EXG4"]

    if replacements == None:
        replacements = ["I1", "I2", "PO9", "PO10"]

    npy_data , meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    labels = (meta_data["chanlocs"]["labels"])
    mapping = dict(zip(targets, replacements))
    labels = [mapping.get(label, label) for label in labels]
    meta_data["chanlocs"]["labels"] = labels

    extn_str = f"{stepno}_chanlabels "
    filepath = filepath.with_name(extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data, extn_str)
    savealldata(npy_data, meta_data, filepath)
    print("channels ", targets ," are renamed as ",replacements ," and saved in and as: ", filepath)
    print("\n")

    return npy_data, meta_data, filepath

def electrodelocationchange(filepath, stepno, electrodelocationfilepath=None):

    if electrodelocationfilepath == None:
        electrodelocationfilepath = r"biosemi_locations_64_10-20_fixP9P10_add4.xyz"

    
    extn_str = f"{stepno}_chanlocs "
    electrode_data = np.loadtxt(electrodelocationfilepath, dtype=str, max_rows=68)
    npy_data , meta_data, npyfilepath, metafilepath = loadalldata(filepath)

    Xcord = electrode_data[:, 1]
    Xcord = np.append(Xcord, ['0', '0', '0', '0',
                              '0'])  # for some reason, there are 70 channels with only 68 entries so appending zeros
    Xcord = Xcord.astype(float)

    Ycord = electrode_data[:, 2]
    Ycord = np.append(Ycord, ['0', '0', '0', '0',
                              '0'])  # for some reason, there are 70 channels with only 68 entries so appending zeros
    Ycord = Ycord.astype(float)

    Zcord = electrode_data[:, 3]
    Zcord = np.append(Zcord, ['0', '0', '0', '0',
                              '0'])  # for some reason, there are 70 channels with only 68 entries so appending zeros
    Zcord = Zcord.astype(float)

    def cart_to_sph(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = 0 if x == 0 else np.arctan2(y, x)  # azimuth #if x=y=z=0 make angle also zeros
        phi = 0 if r == 0 else np.arcsin(z / r)  # elevation  #if x=y=z=0 make angle also zeros
        return r, theta, phi

    r = np.zeros(len(Xcord))
    theta = np.zeros(len(Xcord))
    phi = np.zeros(len(Xcord))
    theta_besa = np.zeros(len(Xcord))
    phi_besa = np.zeros(len(Xcord))

    for chid in range(len(Xcord)):
        r[chid], theta[chid], phi[chid] = cart_to_sph(Xcord[chid], Ycord[chid], Zcord[chid])
        theta_besa[chid] = .5 * np.pi - theta[chid]
        phi_besa[chid] = .5 * np.pi - phi[chid]


    X = np.array([np.array([x]) for x in Xcord], dtype=object).T
    Y = np.array([np.array([y]) for y in Ycord], dtype=object).T
    Z = np.array([np.array([z]) for z in Zcord], dtype=object).T
    r = np.array([np.array([x]) for x in r], dtype=object).T
    phi = np.array([np.array([y]) for y in phi], dtype=object).T
    theta = np.array([np.array([z]) for z in theta], dtype=object).T
    phi_besa = np.array([np.array([x]) for x in phi_besa], dtype=object).T
    theta_besa = np.array([np.array([y]) for y in theta_besa], dtype=object).T

    meta_data["chanlocs"].update({
        "radius":r,
        "sph_phi" : phi,
        "sph_theta": theta,
        "sph_phi_besa": phi_besa,
        "sph_theta_besa": theta_besa,
        "X": X,
        "Y": Y,
        "Z": Z
    })

    meta_data["fields"].update({
        "chanlocs.radius": "radius of channel from 0,0,0",
        "chanlocs.sph_phi": "phi angle of channel",
        "chanlocs.sph_theta": "theta angle of channel",
        "chanlocs.sph_phi_besa": "phi angle of channel mapped onto the brain",
        "chanlocs.sph_theta_besa": "theta angle of channel mapped onto the brain",
    })

    filepath = filepath.with_name( extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data, extn_str)
    savealldata(npy_data, meta_data, filepath)
    print("elec. locations are changed! and saved as ", filepath)
    print("\n")
    return npy_data, meta_data, filepath

def downsampling(filepath, stepno, dsfact = None, showsummaryflag = True, summarychid = None):
    
    extn_str = f"{stepno}_ds "
    
    npy_data , meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    if dsfact is None:
        dsfact = meta_data["fs"]//256

    ndpts = npy_data.shape[0]  # number of data points
    nch = npy_data.shape[1]  # number of channels

    ds_data = np.zeros((ndpts // dsfact, nch))

    dsampind = range(0, ndpts, dsfact)
    ds_data[:, :nch] = npy_data[dsampind, :nch]

    # saving the data file
    npy_data = ds_data.copy()
    del ds_data
    old_fs = meta_data["fs"]
    new_fs = int(meta_data["fs"]) // dsfact
    meta_data["fs"] = new_fs
    filepath = filepath.with_name( extn_str + filepath.stem)
    savealldata(npy_data, meta_data, filepath)
    
    print("data downsampled from", old_fs,"hz to", new_fs,"hz") 
    print("data saved as: ", filepath)
    print("\n")
    return npy_data, meta_data, filepath

def delete_channels(filepath,stepno,deletechnames = None):
    
    extn_str = f"{stepno}_chan-select "
    
    if deletechnames == None:
        deletechnames = ['EXG5', 'EXG6', 'Status']
    
    deletechnames = np.array(deletechnames)
    deletechnames = np.char.strip(np.char.lower(deletechnames.astype(str)))
    # print(deletechnames)
    npy_data , meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    labels = np.array(meta_data["chanlocs"]["labels"],dtype = str)
    labels = np.char.strip(np.char.lower(labels.astype(str)))
    indices = np.where(np.isin( labels, deletechnames))[0]
    indices = indices.astype(int)
    mask = np.ones(len(labels), dtype=bool)
    mask[indices] = False
    # deleting data
    npy_data = npy_data[:, mask]
    meta_data["deleted_chnames"] = [meta_data["chanlocs"]["labels"][i] for i in indices]

    for key in meta_data["chanlocs"].keys():
        arr = meta_data["chanlocs"][key]
        meta_data["chanlocs"][key] = np.delete(arr, indices)

    meta_data["shape"] = npy_data.shape
    meta_data["size"] = npy_data.size

    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    filepath = filepath.with_name(extn_str + filepath.stem)
    savealldata(npy_data, meta_data, filepath)
    print("channels being deleted are: ", meta_data["deleted_chnames"])
    print("data saved in ",filepath)
    print("\n")

    return npy_data, meta_data, filepath

def bandpassfilter(filepath,stepno,filterparameters = None, showsummaryflag = True, summarychid = None):
    
    extn_str = f"{stepno}_but "
    if filterparameters == None:
        filterparameters = [4, 0.1, 100] #order of the filter, lowpass and highpass 

    npy_data , meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    nch = npy_data.shape[1]

    # Design Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = filterparameters[1] / nyquist
    high = filterparameters[2] / nyquist
    b, a = butter(filterparameters[0], [low, high], btype='band')
    filt_data = np.zeros(npy_data.shape)

    # Apply filter
    for chid in range(0, nch, 1):
        signal = npy_data[:, chid]
        filt_data[:, chid] = filtfilt(b, a, signal)  # bandpass filtering

    meta_data["bandpass filter param"] = filterparameters
    meta_data["fields"].update({"bandpass filter param": "filter order,low cutoff, high cutoff"})

    npy_data = filt_data.copy()
    del filt_data

    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    filepath = filepath.with_name( extn_str + filepath.stem)
    savealldata(npy_data, meta_data, filepath)

    print("data has been band pass filtered", [filterparameters[1], filterparameters[2]], "order: ", filterparameters[0])
    print("data saved as ", filepath)
    print("\n")
    return npy_data, meta_data, filepath

def notchfilter(filepath,stepno,notchparameters = None, showsummaryflag = True, summarychid = None):
    extn_str = f"{stepno}_notchfilt "
    
    if notchparameters == None:
        notchparameters = [4,50,100]

    npy_data , meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    nch = npy_data.shape[1]
    an = []
    bn = []
    for numfreq in range((len(notchparameters)-1)):
        bn_coeff, an_coeff = iirnotch(notchparameters[numfreq+1],notchparameters[0],fs)
        an.append(an_coeff)
        bn.append(bn_coeff)
        
    for chid in range(0, nch, 1):
        signal = npy_data[:, chid]
        for numfilter in range(len(an)):
            signal = filtfilt(bn[numfilter],an[numfilter], signal)
        npy_data[:, chid] = signal

    meta_data["notchfilter param"] = notchparameters
    meta_data["fields"].update({"notchfilter param": "slope, frequencies to filter..."})

    cf.updatemetadataHistory(meta_data, extn_str)
    filepath = filepath.with_name( extn_str + filepath.stem)

    savealldata(npy_data, meta_data, filepath)

    print("notch filtering done to filter out frequencies:", notchparameters[1:],"hz, slope of the filter is: ", notchparameters[0])
    print("data saved as: ", filepath)
    print("\n")

    return npy_data, meta_data, filepath

def performICA(filepath, ch_name = None):

    if ch_name == None:
        ch_name = 'Fp1'

    npy_data, meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    # Design Butterworth bandpass filter
    lowcut = 1
    highcut = 30
    order = 4

    
    fs = int((meta_data["fs"]))
    slope = 2
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    fs = meta_data["fs"]

    dataforICA = np.zeros(npy_data.shape)  # removing nonessential dims
    b, a = butter(order,[low, high], btype='band')
    bn, an = iirnotch(50, slope, fs)

    for chid in range(npy_data.shape[1]):
        sig = filtfilt(b, a, npy_data[:, chid])
        dataforICA[:, chid] = filtfilt(bn, an, sig)
        # dataforICA[:, chid] = filtfilt(bn2, an2, sig)

    dataforICA = np.moveaxis(dataforICA, -1, 0)  # shuffling dims to reflect (nCh,nData)

    # extract channel names from the lw6 data file and create an information base for the mne package
    ch_names = meta_data["chanlocs"]["labels"].tolist()
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(dataforICA, info)

    montage_dict = {}
    for ch in range(len(ch_names)):
        name = ch_names[ch]
        montage_dict[name] = [meta_data["chanlocs"]["X"][ch], meta_data["chanlocs"]["Y"][ch],
                              meta_data["chanlocs"]["Z"][ch]]

    print("length of dict:", len(montage_dict))

    montage = mne.channels.make_dig_montage(ch_pos=montage_dict, coord_frame='head')
    raw.set_montage(montage)
    # perform ICA
    ica = ICA(n_components=0.999, random_state=97, max_iter='auto')
    ica.fit(raw)

    # print(ica)
    sources = ica.get_sources(raw)
    ica_data = sources.get_data()
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=ch_name)
    return ica, raw, ica_data, eog_indices

def overlayICAondata(filepath, ica_data):
    npyfilepath = filepath.with_suffix(".npy")
    metafilepath = filepath.with_suffix(".pkl")
    npy_data = np.load(npyfilepath)
    meta_data = cf.loadMetadata(metafilepath)
    labels = np.array(meta_data["chanlocs"]["labels"], dtype=object)
    subjid = npyfilepath.stem.split()[-1]
    cf.showmeICAoverlayedondata(npy_data, ica_data, labels=labels,subjid=subjid)

def applyICA(filepath, ica, raw, rmidx):
    npy_data, meta_data = loadalldata(filepath)
    print("Removing ICA components:", rmidx)
    ica.exclude = rmidx
    raw_clean = ica.apply(raw.copy())
    npy_data = raw_clean.get_data()

    meta_data["ICA"] = rmidx
    meta_data["shape"] = npy_data.shape
    meta_data["size"] = npy_data.size
    extn_str = "ica_filt "
    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    filepath = filepath.with_name( extn_str + filepath.stem)
    npy_data = np.moveaxis(npy_data, -1, 0)
    savealldata(npy_data, meta_data, filepath)
    print("ICA performed, data saved as ", filepath)
    print("\n")

    return npy_data, meta_data, filepath

def segmentation(filepath, stepno, startcode = None, segmentparams = [None,None], showsummaryflag = True, summarychid = None):

    extn_str = f"{stepno}_ep "
    errorflag = 0
    if startcode == None:
        startcode = np.array([10, 45, 90, 135, 200, 210, 212, 214, 216, 218, 220, 222, 224])
    
    endcode = startcode + 1
    if segmentparams == [None,None]:
        segmentparams = np.array([70, -2])

    if len(segmentparams) != 2:
        raise InvalidLengthError(f"expected an array of length instead got length{len(segmentparams)}, reneter your segmentation parameters as [duration(in seconds), start latency]")
        
    npy_data, meta_data, npyfilepath, metafilepath =loadalldata(filepath)
    fs = meta_data["fs"]
    
    startlatencyfromstart = segmentparams[1]
    endlatencyfromstart = segmentparams[0] + segmentparams[1]

    startlatencyfromend = segmentparams[0]
    endlatencyfromend = np.array([0])

    eventid = np.array(meta_data["events"]["code"])
    timestamps = np.array(meta_data["events"]["latency"], dtype=float)
    duration = segmentparams[0] * fs
    
    unq_eventid = np.unique(eventid)
    mask = np.array([x.isnumeric() for x in unq_eventid])
    unq_eventid = unq_eventid[mask]
    unq_eventid = unq_eventid.astype(int)
    unq_eventid.sort()
    unq_eventid = unq_eventid.astype(str)

    startcode_ind = np.where(np.isin(unq_eventid.astype(int), startcode))[0]
    startevents = unq_eventid[startcode_ind]
    startevents.sort()

    endcode_ind = np.where(np.isin(unq_eventid.astype(int), endcode))[0]
    endevents = unq_eventid[endcode_ind]
    endevents.sort()

    S0id = [None] * len(startevents)
    Snid = [None] * len(endevents)

    startendsample = [[None] for i in range(len(startevents))]
    eventreps = np.zeros(len(endevents))

    for eventno in range(len(startevents)):
        Ks = (fs * timestamps[[i for i, x in enumerate(eventid) if x == startevents[eventno]]])
        S0id[eventno] = Ks.astype(int)

        Ke = (fs * timestamps[[i for i, x in enumerate(eventid) if x == endevents[eventno]]])
        Snid[eventno] = Ke.astype(int)

        starts = np.array(S0id[eventno])
        stops = np.array(Snid[eventno])
        used_stops = np.zeros(len(stops), dtype=bool)

        # First: match starts to closest valid stops
        pairs = []
        for s in starts:
            diffs = stops - s
            # valid stops should be ~70s after start
            valid = np.where((diffs > 0) & (np.abs(diffs - segmentparams[0]*fs) < fs))[0]
            if len(valid) > 0:
                idx = valid[0]
                pairs.append((int(s + startlatencyfromstart * fs), int(stops[idx])))
                used_stops[idx] = True
            else:
                # missing stop → infer
                pairs.append((int(s + startlatencyfromstart * fs), int(s + endlatencyfromstart * fs)))

        # Second: handle stops pairs that were never used (missing starts)
        for i, stop_used in enumerate(used_stops):
            if not stop_used:
                e = stops[i]
                pairs.append((int(e - startlatencyfromend*fs), int(e)))

        # Uncomment these lines when debugging
        # print(f"for event {startevents[eventno]}")
        # for first, second in pairs: 
        #     # Use abs() if you want the absolute distance, or 'first - second' for exact difference
        #     difference = abs(first - second)
        #     print(f"The difference between  start {first} and end {second} is {difference}")
            
        pairs = cf.givemeUniqueTuples(pairs, tolerance=2 * fs)
        # for i in range(len(pairs)):
        # print(f"Event {startevents[eventno]}: Found {np.sum(used_stops)} clean matches, inferred {len(pairs) - np.sum(used_stops)} missing links.")
        startendsample[(eventno)] = pairs
        eventreps[eventno] = len(pairs)
        
    # print(S0id) 

    for idx, pairs_i in enumerate(startendsample):
        if len(pairs_i) not in [2, 5]:
            print(f"Warning: Event {startevents[idx]} has {len(pairs_i)} reps")
            errorflag = 1

    # so far we have removed the obtained the time stamp, sample stamp and event code of all the stim that have been presented.
    # I have then removed the time and sample stamp of all those events that are not relevant to the experiment's analysis like 21/22/50/55
    # trimming the data points

    nch = npy_data.shape[1]
    ep_data = np.zeros((duration , nch, int(sum(eventreps))))
    ## add functionality that makes this compatible with all expts, not just this one ie. self calculating the 41 in this case
    count = 0
    eventrepdata = {}
    startendsampid = np.zeros([int(sum(eventreps)), 2])

    for eventno in range(len(startcode)):
        numrep = []
        for iterid in range(int(eventreps[eventno])):
            dsamp_start = startendsample[eventno][iterid][0]
            dsamp_end = startendsample[eventno][iterid][1]
            startendsampid[count, 0] = dsamp_start
            startendsampid[count, 1] = dsamp_end

            eventrep_dat = np.arange(dsamp_start, dsamp_end, dtype="int")
            # print("range:", dsamp_start, "-", dsamp_end, " length:", dsamp_end - dsamp_start)
            ep_data[:, :nch, count] = npy_data[eventrep_dat, :nch]
            numrep.append(count)
            count += 1
            # final data will have dimensions like so:(ndpts, nch, nevents)
        print("Event label being split: ", startevents[eventno], " number of reps: ", len(numrep))
        eventrepdata[str(startevents[eventno])] = np.array(numrep)

    if errorflag == 0:
        npy_data = ep_data.copy()
        meta_data["eventrepid"] = eventrepdata

    meta_data = cf.updatemetadataHistory(meta_data, extn_str)
    filepath = filepath.with_name( extn_str + filepath.stem)
    savealldata(npy_data, meta_data, filepath)

    print(f"data has been segmented and saved as {filepath}")
    print("\n")
    return npy_data, meta_data, filepath

def interpolate(filepath,stepno, interp_chnames, bad_chnames = None, showsummaryflag = True, summarychid = None):

    npy_data, meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    
    labels = (meta_data["chanlocs"]["labels"])
    interp_chids = np.where(np.isin(labels, interp_chnames))[0]
 
    # interpolation
    badch = np.union1d(interp_chnames, bad_chnames)
    interp_specs = {}
    for i in range(len(interp_chnames)):
        srt_idx, srt_labels = cf.givemeNNearestNeighbour(meta_data, interp_chids[i])
        srtd_idx = np.where(~np.isin(srt_labels, badch))[0]
        srtd_idx = srtd_idx[:3]
        badch = np.append(badch, srt_labels[srtd_idx])
        npy_data[:, interp_chids[i], :] = np.mean(npy_data[:, srtd_idx, :], axis=1)
        new_entry = {interp_chnames[i]: srt_labels[srtd_idx]}
        interp_specs.update(new_entry)
        print(badch, "is interpolated using ", [labels[i] for i in srtd_idx])

    meta_data["interpolation"] = interp_specs
    extn_str = f"{len(interp_chnames)}_interp"

    filepath = npyfilepath.with_name(extn_str + filepath.stem)
    savealldata(npy_data, meta_data, filepath)
    print("channels are interpolated and saved as ", filepath)
    print("\n")
    return npy_data, meta_data, filepath, badch

def globalreferencing(filepath,stepno, badchids= None, showsummaryflag = True, summarychid = None):
    extn_str = f"{stepno}_ref "
    npy_data, meta_data,npyfilepath,metafilepath = loadalldata(filepath)
    labels = meta_data["chanlocs"]["labels"]
    goodchids = np.where(~np.isin(labels, badchids))[0]
    glreference = np.mean(npy_data[:, goodchids, :], axis=1,keepdims=True)
    npy_data = npy_data - glreference
    meta_data["ref_chids"] = goodchids
    filepath = filepath.with_name(extn_str + filepath.stem)
    savealldata(npy_data, meta_data, filepath)
    
    print(r"global referencing done and saved as ", filepath)
    print("\n")

    return npy_data, meta_data, filepath

def separateepochs(filepath, mergekeys = None, mergekeyflag = True, showsummaryflag = True):
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    subjid = metafilepath.stem.split()[-1]
    if len(subjid) != 8:  # Prevents mistaking something else as the subject id as we know that subject id is usually 8 characters long
        subjid = metafilepath.stem.split()[-2]

    # default value is that some events need to be merged. if this is set to 0 then no event is merged
    if mergekeys == None:
        mergekeys = {
            "TOP": [210, 212],
            "BOTTOM": [214, 216],
            "RIGHT": [218, 220],
            "LEFT": [222, 224]
        }

    slices = meta_data["eventrepid"]
    
    if mergekeyflag == True:
        keys = mergekeys.keys()
        delkeys = []
        for i in keys:
            a = slices[str(mergekeys[i][0])]
            b = slices[str(mergekeys[i][1])]
            ab = np.union1d(a, b)
            slices[i] = ab
            delkeys.append(str(mergekeys[i][0]))
            delkeys.append(str(mergekeys[i][1]))
            
        list(map(slices.pop, delkeys))
    
    filepaths = []
    for label, indices in slices.items():
        indices = np.array(indices).astype(int)
        data = npy_data[:, :, indices]
        # print(start:end)
        filepath = metafilepath.parent / f"{label} {subjid}.npy"
        filepaths.append(filepath)
        np.save(filepath, data)
        meta_data['event'] = label
        cf.saveMetadata(meta_data,filepath.with_suffix('.pkl'))        
        # print(f"{label}: {start}:{end} for size {data.shape[2]}")
        print(f"event {label} Saved as: {filepath}: ")
        print("\n")

    return filepath

def mergeepochs(folderpath,event_label):
    print(f"Merging epochs from the event: {event_label}")

    files = sorted([f for f in folderpath.glob(f"{event_label}*.npy")
                    if not f.stem.endswith("MERGED") and not f.name.startswith(f"{event_label}_")
                    ])

    # Excludes the merged files and preexisting files that have event_label_ (e.g. "10_")
    # in front of them

    files = files[:]
    data_list = []
    subj_merged = []
    for file in files:
        strings = file.stem
        strings = strings.split(" ")
        subjid = strings[-1]
        if len(subjid) != 8:  # Prevents taking '.pkl' as the subject id as we know that subject id is usually 8 characters long
            continue
        data = np.load(file)  # loads array saved earlier
        print(subjid, " with matrix size as ", data.shape)
        data_list.append(data)
        subj_merged.append(subjid)
        # print(file)
        
    tempfilepath = file
    tempfilepath = tempfilepath.with_suffix(".pkl")
    tempmeta_data = cf.loadMetadata(tempfilepath)

    npy_data = np.concatenate(data_list, axis=2)
    print(npy_data.shape)
    meta_data = {
        "event_label": event_label,
        "subjids": subj_merged,
        "shape": npy_data.shape,
        "size": npy_data.size,
        "fs": 256,
        "chanlocs": tempmeta_data["chanlocs"],
        "history": {},
        "fields": tempmeta_data["fields"],
        "bandpass filter param": meta_data["bandpass filter param"]
    }
    meta_data["fields"].update({"shape":"ndata,nchannels,nepochs,nsubjects"})
    filepath = folderpath/ f"{event_label} MERGED"
    savealldata(npy_data, meta_data,filepath)
    print(f"{event_label} for all files are merged and saved as: {filepath}")
    print("\n")
    return npy_data, meta_data, filepath

def frequencytransform(filepath,stepno,freqbin = None, showsummaryflag = True, summarychid = None):
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    if freqbin == None:
        freqbin = [0.01 ,50]

    freq_min = freqbin[0]
    freq_max = freqbin[1]
    
    extn_str = f"{stepno}_fft "
    fs = meta_data["fs"]
    freqs = np.fft.fftfreq(npy_data.shape[0], 1 / fs)
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]
    n_timepoints = npy_data.shape[0]
    FFT_data = np.zeros([len(freq_idx), npy_data.shape[1], npy_data.shape[2]])

    for chid in range(npy_data.shape[1]):
        for epochid in range(npy_data.shape[2]):
            FFT_signal = np.abs(np.fft.fft((npy_data[:, chid, epochid])))
            FFT_signal = (np.abs(FFT_signal) / n_timepoints)
            FFT_data[:, chid, epochid] = FFT_signal[idx]

    meta_data["freq_range"] = freqbin
    meta_data["freq_res"] = freqs[1]-freqs[0]
    npy_data = FFT_data.copy()
    cf.updatemetadataHistory(meta_data,extn_str)
    filepath = filepath.with_name(extn_str + filepath.stem)
    del FFT_data
    savealldata(npy_data, meta_data, filepath)
    print(f"Data is transformed to the frequency domain and saved as {filepath}") 
    print('\n')
    return npy_data, meta_data, filepath

def averagingacrosstrials(filepath,stepno, showsummaryflag = True, summarychid = None):
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    npy_data = np.mean(npy_data, axis=2)
    extn_str = f"{stepno}_avg "
    filepath = filepath.with_name(extn_str + filepath.stem)    
    cf.updatemetadataHistory(meta_data,extn_str)
    savealldata(npy_data, meta_data, filepath)
    print(f"Data is averaged across trials and saved as {filepath}") 
    print('\n')
    return npy_data, meta_data, filepath

def chunking(filepath,stepno,basefreq = 1.2 ,window_width = .4, showsummaryflag = True, summarychid = None):
    extn_str = f"{stepno}_chunk"
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    freq_min = meta_data["freq_range"][0]
    freq_max = meta_data["freq_range"][1]
    # freq_res = (freq_max-freq_min)/fs
    freq_res = meta_data["freq_res"]
    freqs = np.arange(freq_min, freq_max, freq_res)

    chunkWidth = window_width + freq_res
    meta_data["chunk_width"] = chunkWidth
    harmonics = np.arange(basefreq, freq_max, basefreq)
    harmonics = harmonics[harmonics <= 50]
    chunk_data = np.zeros((int(np.floor(chunkWidth / freq_res)), npy_data.shape[1], len(harmonics)))
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]

    for chid in range(npy_data.shape[1]):
        for fhid in range(len(harmonics)):
            idx = np.where(
                (freq_idx > (harmonics[fhid] - chunkWidth / 2)) & (freq_idx <= (harmonics[fhid] + chunkWidth / 2)))[0]
            chunk_data[:, chid, fhid] = npy_data[idx, chid]

    npy_data = chunk_data.copy()
    filepath = filepath.with_name(extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    savealldata(npy_data, meta_data, filepath)
    print(f"Data is chunked at frequencies of interest and saved as {filepath}") 
    print('\n')
    return npy_data, meta_data, filepath

def selectingchunks(filepath, stepno,numharmonics = None, showsummaryflag = True, summarychid = None):
    if numharmonics == None:
        numharmonics = 3
    br_extn_str = f"{stepno}_baseline"
    odd_extn_str = f"{stepno}_oddball"
    #So far, this selects only the first three harmonics of the baseline and oddball responses
    
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    bl_chunks = np.arange(4, npy_data.shape[2], 5)
    bl_chunkmask = np.zeros(npy_data.shape[2], dtype=bool)
    bl_chunkmask[bl_chunks] = "True"
    odd_chunkmask = ~bl_chunkmask

    br_data = npy_data[:, :, bl_chunkmask]
    odd_data = npy_data[:, :, odd_chunkmask]
    # Select the first 3 for baseline and 12 chunks for oddball
    
    br_data = br_data[:, :, 0:numharmonics]
    odd_data = odd_data[:, :, 0:numharmonics]

    br_meta_data = meta_data.copy()
    odd_meta_data = meta_data.copy()

    br_meta_data["nChunks"] = numharmonics
    br_meta_data["eventType"] = "Baseline"

    odd_meta_data["nChunks"] = numharmonics
    odd_meta_data["eventType"] = "Oddball"

    br_meta_data = cf.updatemetadataHistory(br_meta_data, br_extn_str)
    odd_meta_data = cf.updatemetadataHistory(odd_meta_data, odd_extn_str)
    filepath_br = filepath.with_name(br_extn_str + filepath.stem)
    filepath_odd = filepath.with_name(odd_extn_str + filepath.stem)
    savealldata(br_data, br_meta_data, filepath_br)
    savealldata(odd_data, odd_meta_data, filepath_odd)
    print(f"Interesting chunks of the data are selected and saved as {filepath}") 
    print('\n')
    return br_data, br_meta_data, odd_data, odd_meta_data, filepath_br, filepath_odd

def sumofharmonics(filepath,stepno, showsummaryflag = True, summarychid = None):
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    npy_data = np.sum(npy_data, axis=2)
    filepath = filepath.with_name(f"{stepno}_sum " + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,f"{stepno}_sum")
    metafilepath = metafilepath.with_name(f"{stepno}_sum " + metafilepath.stem)
    savealldata(npy_data, meta_data, filepath)
    print("Harmonics of npy_data is summed and the new shape is:", npy_data.shape)
    print("\n")
    savealldata(npy_data, meta_data, filepath)

    return npy_data, meta_data, filepath

def baselinefiltering(filepath,stepno):
    npy_data, meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    mid_point = (npy_data.shape[0]//2)-1
    idx = np.arange(npy_data.shape[0])
    idx = np.delete(idx,[mid_point-1,mid_point,mid_point+1])
    baseline = np.mean(npy_data[idx,:],axis = 0)
    npy_data = npy_data - baseline
    extn_str = f"{stepno}_bsl "
    filepath  = filepath.with_name(extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    savealldata(npy_data, meta_data, filepath)
    return npy_data,meta_data, filepath
    print(f"data is baseline filtered and saved as {filepath}")
    print('\n')
    
def mergeevents(folderpath,event_label):
    files = sorted([f for f in folderpath.glob(f"{event_label}*.npy")
                    if not f.stem.endswith("MERGED") and not f.name.startswith(f"{event_label}_")
                    ])
    # Excludes the merged files and preexisting files that have event_label_ (e.g. "10_")
    # in front of them

    files = files[:]
    subj_merged = []
    data_list = []
    for file in files:
        strings = file.stem
        strings = strings.split(" ")
        subjid = strings[-1]
        data = np.load(file)  # loads array saved earlier
        print(subjid, " with matrix size as ", data.shape)
        data_list.append(data)
        subj_merged.append(subjid)

    tempfilepath = file.with_suffix(".pkl")
    tempmeta_data = cf.loadMetadata(tempfilepath)
    npy_data = np.stack(data_list,axis = -1)
    print("the shape of your new data is: ",npy_data.shape)
    
    meta_data = {
        "event_label": event_label,
        "subjids": subj_merged,
        "shape": npy_data.shape,
        "size": npy_data.size,
        "fs": tempmeta_data["fs"],
        "chanlocs": tempmeta_data["chanlocs"],
        "fields": tempmeta_data["fields"]
    }
    
    meta_data["fields"].update({"shape":"ndata,nchannels,nepochs,nsubjects"})
    filepath = folderpath/ f"{event_label} ME MERGED.npy"
    savealldata(npy_data, meta_data,filepath)
    print(f"all files for event {event_label} are merged and saved as: {filepath}")
    print("\n")
    return npy_data, meta_data, filepath

def MEfrequencytransform(filepath,stepno,freqbin = None, showsummaryflag = True, summarychid = None):
    if freqbin == None:
        freqbin = [0.05, 50]

    freq_min = freqbin[0]
    freq_max = freqbin[1]
    
    extn_str = f"{stepno}_fft "
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    meta_data["freq_range"] = [freq_min, freq_max]

    freqs = np.fft.fftfreq(npy_data.shape[0], 1 / fs)
    n_timepoints = npy_data.shape[0]
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]
    FFT_data = np.zeros([len(freq_idx), npy_data.shape[1], npy_data.shape[2],npy_data.shape[3]])

    for subjid in range(npy_data.shape[-1]):
        for chid in range(npy_data.shape[1]):
            for epochid in range(npy_data.shape[2]):
                FFT_signal = np.abs(np.fft.fft((npy_data[:, chid, epochid, subjid])))
                FFT_signal = (np.abs(FFT_signal) / n_timepoints)
                FFT_data[:, chid, epochid, subjid] = FFT_signal[idx]

    meta_data["freq_range"] = freqbin
    meta_data["freq_res"] = freqs[1]-freqs[0]    
    if "history" not in meta_data:
        meta_data["history"] = {}
    cf.updatemetadataHistory(meta_data,extn_str)
    filepath = filepath.with_name(extn_str + filepath.stem)
    npy_data = FFT_data.copy()
    del FFT_data
    savealldata(npy_data, meta_data, filepath)
    return npy_data, meta_data, filepath

def MEaveragingacrosstrials(filepath,stepno, showsummaryflag = True, summarychid = None):
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    npy_data = np.mean(npy_data[:,:,:,:], axis=-2,keepdims = True)
    extn_str = f"{stepno}_avg "
    filepath = filepath.with_name(extn_str + filepath.stem)    
    cf.updatemetadataHistory(meta_data,extn_str)
    savealldata(npy_data, meta_data, filepath)
    print(f"Data is averaged across trials and saved as {filepath}") 
    print('\n')
    return npy_data, meta_data, filepath

def MEchunking(filepath,stepno,basefreq = 1.2,window_width = .4, showsummaryflag = True, summarychid = None): 
    extn_str = f"{stepno}_chunk "
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    freq_min = meta_data["freq_range"][0]
    freq_max = meta_data["freq_range"][1]
    # freq_res = (freq_max-freq_min)/fs
    print(npy_data.shape)
    freq_res = meta_data["freq_res"]
    freqs = np.arange(freq_min, freq_max, freq_res)
    print((window_width//freq_res)//2)
    if (window_width//freq_res)//2 == 0:
        chunkWidth = window_width + freq_res
    else:
        chunkWidth = window_width
    
    meta_data["chunk_width"] = chunkWidth
    harmonics = np.arange(basefreq, freq_max, basefreq)
    harmonics = harmonics[harmonics <= 50]
    print(harmonics)
    chunk_data = np.zeros((int(np.floor(chunkWidth / freq_res)), npy_data.shape[1], len(harmonics),npy_data.shape[3]))
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]
    for subjid in range(npy_data.shape[-1]):
        for chid in range(npy_data.shape[1]):
            for fhid in range(len(harmonics)):
                idx = np.where(
                    (freq_idx > (harmonics[fhid] - chunkWidth / 2)) & (freq_idx <= (harmonics[fhid] + chunkWidth / 2)))[0]
                # print(chunk_data[:, chid,fhid,subjid].shape," and npy_data shape ", npy_data[idx, chid,:,subjid].shape)
                chunk_data[:, chid,fhid,subjid] = np.array(npy_data[idx, chid,:,subjid].squeeze())

    npy_data = chunk_data.copy()
    filepath = filepath.with_name(extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    savealldata(npy_data, meta_data, filepath)
    print(f"Data is chunked at frequencies of interest and saved as {filepath}") 
    print('\n')
    return npy_data, meta_data, filepath

def MEselectingchunks(filepath, stepno,numharmonics = None, showsummaryflag = True, summarychid = None):
    if numharmonics == None:
        numharmonics = 3
    extn_str = f"{stepno}_ep-select "
    #So far, this selects only the first three harmonics of the baseline and oddball responses
    
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    bl_chunks = np.arange(4, npy_data.shape[2], 5)
    print("bl_chunks are:", (bl_chunks+1)*1.2)
    bl_chunkmask = np.zeros(npy_data.shape[2], dtype=bool)
    bl_chunkmask[bl_chunks] = "True"
    odd_chunkmask = ~bl_chunkmask
    print(1.2*(np.where(odd_chunkmask)[0]+1))
    br_data = npy_data[:, :, bl_chunkmask,:]
    odd_data = npy_data[:, :, odd_chunkmask,:]
    # Select the first 3 for baseline and 12 chunks for oddball
    
    br_data = br_data[:, :, 0:numharmonics,:]
    odd_data = odd_data[:, :, 0:numharmonics,:]

    br_meta_data = meta_data.copy()
    odd_meta_data = meta_data.copy()

    br_meta_data["nChunks"] = numharmonics
    br_meta_data["eventType"] = "Baseline"

    odd_meta_data["nChunks"] = numharmonics
    odd_meta_data["eventType"] = "Oddball"

    br_meta_data = cf.updatemetadataHistory(br_meta_data, extn_str)
    odd_meta_data = cf.updatemetadataHistory(odd_meta_data, extn_str)

    br_filepath = filepath.with_name(extn_str + filepath.stem)
    odd_filepath = filepath.with_name(extn_str + filepath.stem)

    savealldata(br_data, br_meta_data, br_filepath)
    savealldata(odd_data, odd_meta_data, odd_filepath)
    return br_data, br_meta_data, br_filepath, odd_data, odd_meta_data, odd_filepath

def MEsumofharmonics(filepath,stepno, showsummaryflag = True, summarychid = None):
    npy_data, meta_data, npyfilepath,metafilepath = loadalldata(filepath)
    npy_data = np.sum(npy_data, axis=2,keepdims = True)
    extn_str = f"{stepno}_sum "
    filepath = filepath.with_name(extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,extn_str )
    metafilepath = metafilepath.with_name(extn_str + metafilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)
    np.save(npyfilepath, npy_data)
    print("Harmonics of npy_data is summed and the new shape is:", npy_data.shape)
    print("\n")
    savealldata(npy_data, meta_data, filepath)
    return npy_data, meta_data, filepath

def MEbaselinefiltering(filepath,stepno):
    npy_data, meta_data, npyfilepath, metafilepath = loadalldata(filepath)
    mid_point = (npy_data.shape[0]//2)-1
    idx = np.arange(npy_data.shape[0])
    idx = np.delete(idx,[mid_point-1,mid_point,mid_point+1])
    baseline = np.mean(npy_data[idx,:,:,:],axis = 0,keepdims = True)
    print(baseline.shape)
    npy_data = npy_data - baseline
    extn_str = f"{stepno}_bsl "
    filepath  = filepath.with_name(extn_str + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    savealldata(npy_data, meta_data, filepath)
    return npy_data,meta_data, filepath

def showmesummaryplot(npy_data, meta_data):
    fs = meta_data["fs"]
    labels = np.squeeze(meta_data["chanlocs"]["labels"])
    labels = labels.astype(str)
    n_timepoints = npy_data.shape[0]

    # 1. New Frequency Range Widget
    freq_slider = widgets.IntRangeSlider(
        value=[0, 20],
        min=0,
        max=int(fs / 2),  # Nyquist limit
        step=1,
        description='Freq (Hz):',
        continuous_update=False,  # Updates plot only when you release the mouse
        layout={'width': '300px'}
    )

    #New and improved plotting function, handles both the 2D and 3D npy_data
    def plot_data(domain, selected_series, freq_range, epoch_idx=0):
        plt.figure(figsize=(10, 5))
    
        if not selected_series:
            plt.text(0.5, 0.5, "Select channels from the list", ha='center', va='center')
            plt.show()
            return
    
        time_vector = np.arange(n_timepoints) / fs
    
        for s_idx in selected_series:
            # Safely handle both 2D and 3D shapes
            if len(npy_data.shape) == 3:
                series_data = npy_data[:, s_idx, epoch_idx]
            else:
                series_data = npy_data[:, s_idx]
            
            if domain == 'Time':
                plt.plot(time_vector, series_data, label=labels[s_idx])
                plt.xlabel("Time (s)")
                plt.ylabel(r"Amplitude ($\mu$V)")
                
            elif domain == 'Frequency':
                # Compute FFT
                fft_vals = np.fft.rfft(series_data)
                fft_freqs = np.fft.rfftfreq(n_timepoints, d=1/fs)
                scaled_fft_magnitude = (np.abs(fft_vals) / n_timepoints) * 2

                # Get limits from slider / configuration
                freq_min, freq_max = freq_range
                idx = np.where((fft_freqs >= freq_min) & (fft_freqs <= freq_max))

                plt.plot(fft_freqs[idx], scaled_fft_magnitude[idx], label=labels[s_idx])
                plt.xlabel("Frequency (Hz)")
                plt.ylabel(r"Magnitude ($\mu$V)")
                plt.xlim(freq_min, freq_max)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        # Apply the 1.2 Hz tick marks specifically when in the Frequency domain
        if domain == 'Frequency':
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1.2))
            # Rotate labels slightly if they crowd each other at wide ranges
            plt.xticks(rotation=45) 
            
        title_suffix = f" | Epoch: {epoch_idx}" if len(npy_data.shape) == 3 else ""
        plt.title(f"Domain: {domain}{title_suffix}")
        plt.show()
            
    # 3. Base UI Widgets
    domain_switch = widgets.ToggleButtons(
        options=['Time', 'Frequency'],
        value='Time',
        description='Domain:',
        button_style='info'
    )

    series_selector = widgets.SelectMultiple(
        options=[(labels[i], i) for i in range(npy_data.shape[1])],
        value=[0, 1, 2, 3],  
        description='Ch Name:',
        layout={'height': '225px' if len(npy_data.shape) == 3 else '400px', 'width': '150px'}
    )

    # 4. Handle 2D vs 3D Layout Setup
    widget_dict = {
        'domain': domain_switch,
        'selected_series': series_selector,
        'freq_range': freq_slider
    }

    if len(npy_data.shape) == 3:
        epoch_selector = widgets.Select(
            options=[(f"Ep: {i}", i) for i in range(npy_data.shape[2])],
            value=0,
            description='Epoch:',
            layout={'height': '225px', 'width': '150px'}
        )
        widget_dict['epoch_idx'] = epoch_selector
        controls = widgets.VBox([epoch_selector, series_selector])
    else:
        controls = widgets.VBox([series_selector])

    out = widgets.interactive_output(plot_data, widget_dict)

        # 5. Dynamic Slider Visibility (Shows slider ONLY during Frequency Mode)
    def toggle_slider_visibility(change):
        if change['new'] == 'Frequency':
            freq_slider.layout.display = 'block'
        else:
            freq_slider.layout.display = 'none'

    domain_switch.observe(toggle_slider_visibility, names='value')
    freq_slider.layout.display = 'none'  # Hidden by default because default is 'Time'

    # 6. Render Layout
    # Create a small spacer to separate the switch and the slider nicely
    spacer = widgets.Box(layout=widgets.Layout(width='40px'))

    # CHANGE THIS: Use HBox instead of VBox to place them side by side
    top_bar = widgets.HBox([domain_switch, spacer, freq_slider])
    
    main_layout = widgets.VBox([
        top_bar, 
        widgets.HBox([controls, out])
    ])
    display(main_layout)

def showmesummaryplot_alt(npy_data, meta_data):
    """
    fs: Sampling frequency in Hz (default 1000 Hz) to scale the frequency axis.
    """
    fs = meta_data["fs"]
    labels = np.squeeze(meta_data["chanlocs"]["labels"])
    labels = labels.astype(str)
    n_timepoints = npy_data.shape[0]

    # 1. New Frequency Range Widget
    freq_slider = widgets.IntRangeSlider(
        value=[0, 20],
        min=0,
        max=int(fs / 2),  # Nyquist limit
        step=1,
        description='Freq (Hz):',
        continuous_update=False,  # Updates plot only when you release the mouse
        layout={'width': '300px'}
    )

    #New and improved plotting function, handles both the 2D and 3D npy_data
    def plot_data(domain, selected_series, freq_range, epoch_idx=0):
        plt.figure(figsize=(10, 5))
    
        if not selected_series:
            plt.text(0.5, 0.5, "Select channels from the list", ha='center', va='center')
            plt.show()
            return
    
        time_vector = np.arange(n_timepoints) / fs
    
        for s_idx in selected_series:
            # Safely handle both 2D and 3D shapes
            if len(npy_data.shape) == 3:
                series_data = npy_data[:, s_idx, epoch_idx]
            else:
                series_data = npy_data[:, s_idx]
            
            if domain == 'Time':
                plt.plot(time_vector, series_data, label=labels[s_idx])
                plt.xlabel("Time (s)")
                plt.ylabel(r"Amplitude ($\mu$V)")
            
            elif domain == 'Frequency':
                # Compute FFT
                fft_vals = np.fft.rfft(series_data)
                fft_freqs = np.fft.rfftfreq(n_timepoints, d=1/fs)
                scaled_fft_magnitude = (np.abs(fft_vals) / n_timepoints) * 2

                # Get limits from slider / configuration
                freq_min, freq_max = freq_range
                idx = np.where((fft_freqs >= freq_min) & (fft_freqs <= freq_max))
                
                # Automatically grab a unique color for this specific channel
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                channel_color = color_cycle[s_idx % len(color_cycle)]
                
                # Clean implementation passing raw colors safely
                plt.stem(
                    fft_freqs[idx], 
                    scaled_fft_magnitude[idx], 
                    label=labels[s_idx],
                    linefmt='-',                    # Solid line style
                    markerfmt='o',                  # Circle marker style
                    basefmt=' ',                    # Keeps baseline hidden
                )
                
                # Use plt.setp to safely apply the tuple color to the lines and markers
                # This bypasses the strict string format parser limitations
                stems = plt.gca().containers[-1]     # Grabs the last added stem group
                plt.setp(stems.stemlines, color=channel_color)
                plt.setp(stems.markerline, color=channel_color)
                
                plt.xlabel("Frequency (Hz)")
                plt.ylabel(r"Magnitude ($\mu$V)")
                plt.xlim(freq_min, freq_max)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        # Apply the 1.2 Hz tick marks specifically when in the Frequency domain
        if domain == 'Frequency':
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1.2))
            # Rotate labels slightly if they crowd each other at wide ranges
            plt.xticks(rotation=45) 
            
        title_suffix = f" | Epoch: {epoch_idx}" if len(npy_data.shape) == 3 else ""
        plt.title(f"Domain: {domain}{title_suffix}")
        plt.show()
            
    # 3. Base UI Widgets
    domain_switch = widgets.ToggleButtons(
        options=['Time', 'Frequency'],
        value='Time',
        description='Domain:',
        button_style='info'
    )

    series_selector = widgets.SelectMultiple(
        options=[(labels[i], i) for i in range(npy_data.shape[1])],
        value=[0, 1, 2, 3],  
        description='Ch Name:',
        layout={'height': '225px' if len(npy_data.shape) == 3 else '400px', 'width': '150px'}
    )

    # 4. Handle 2D vs 3D Layout Setup
    widget_dict = {
        'domain': domain_switch,
        'selected_series': series_selector,
        'freq_range': freq_slider
    }

    if len(npy_data.shape) == 3:
        epoch_selector = widgets.Select(
            options=[(f"Ep: {i+1}", i) for i in range(npy_data.shape[2])],
            value=0,
            description='Epoch:',
            layout={'height': '225px', 'width': '150px'}
        )
        widget_dict['epoch_idx'] = epoch_selector
        controls = widgets.VBox([epoch_selector, series_selector])
    else:
        controls = widgets.VBox([series_selector])

    out = widgets.interactive_output(plot_data, widget_dict)

        # 5. Dynamic Slider Visibility (Shows slider ONLY during Frequency Mode)
    def toggle_slider_visibility(change):
        if change['new'] == 'Frequency':
            freq_slider.layout.display = 'block'
        else:
            freq_slider.layout.display = 'none'

    domain_switch.observe(toggle_slider_visibility, names='value')
    freq_slider.layout.display = 'none'  # Hidden by default because default is 'Time'

    # 6. Render Layout
    # Create a small spacer to separate the switch and the slider nicely
    spacer = widgets.Box(layout=widgets.Layout(width='40px'))

    # CHANGE THIS: Use HBox instead of VBox to place them side by side
    top_bar = widgets.HBox([domain_switch, spacer, freq_slider])
    
    main_layout = widgets.VBox([
        top_bar, 
        widgets.HBox([controls, out])
    ])

    display(main_layout)