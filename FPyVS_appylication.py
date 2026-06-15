# Hello, this script is in service of the resulting (GUI based) application that will be eventually made.
# Add all steps/functionalities here as a function with no dependencies on FPVS_pycage

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import mne
from mne.preprocessing import ICA
import importlib
import cust_funcs as cf
importlib.reload(cf)

def loadalldata(filepath):
    metafilepath = filepath.with_suffix(".pkl")
    matfilepath = filepath.with_suffix(".npy")
    meta_data = cf.loadMetadata(metafilepath)
    mat_data = np.load(matfilepath)
    return mat_data, meta_data, matfilepath, metafilepath

def savealldata(mat_data, meta_data, filepath):
    matfilepath = filepath.with_suffix(".npy")
    metafilepath = filepath.with_suffix(".pkl")
    cf.saveMetadata(meta_data, metafilepath)
    np.save(matfilepath, mat_data)

# 
# def splitBDFdata(filepath):
#     import mne
# 
#     if filepath.suffix != '.bdf':
#         raise ValueError("The file is not a .bdf file, please enter the filepath for a .bdf file and try again")
# 
#     raw = mne.io.read_raw_bdf(filepath, preload=True)
# 
#     matdata, times = raw.get_data(return_times=True)

def renamechannels(filepath,stepno,  targets = None, replacements = None):

    stepno = stepno.astype(str)
    if targets == None:
        targets = ["EXG1", "EXG2", "EXG3", "EXG4"]

    if replacements == None:
        replacements = ["I1", "I2", "PO9", "PO10"]

    mat_data , meta_data, matfilepath, metafilepath = loadalldata(filepath)
    labels = (meta_data["chanlocs"]["labels"])
    mapping = dict(zip(targets, replacements))
    labels = [mapping.get(label, label) for label in labels]
    meta_data["chanlocs"]["labels"] = labels

    extnstr = stepno + "_chanlabels "
    metafilepath = filepath.with_name(extnstr + metafilepath.stem + metafilepath.suffix)
    metafilepath = metafilepath.with_suffix(".pkl")
    matfilepath = filepath.with_name(extnstr + matfilepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data, extnstr)
    savealldata(mat_data, meta_data, filepath)
    print("channels ", targets ," are renamed as ",replacements ," and saved in and as: ", metafilepath)
    print("\n")

    return mat_data,meta_data,matfilepath,metafilepath

def electrodelocationchange(filepath, stepno, electrodelocationfilepath=None):

    if electrodelocationfilepath == None:
        electrodelocationfilepath = r"D:\Goffaux lab\letswave6_TR\\resources\electrodes\spherical_locations\\biosemi_locations_64_10-20_fixP9P10_add4.xyz"

    stepno = stepno.astype(str)
    electrode_data = np.load(electrodelocationfilepath)
    mat_data , meta_data, matfilepath, metafilepath = loadalldata(filepath)

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

    extnstr = stepno + "_chanlocs "
    filepath = filepath.with_name( extnstr + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data, extnstr)
    savealldata(mat_data, meta_data, filepath)
    print("elec. locations are changed! and saved as ", metafilepath)
    print("\n")
    return mat_data, meta_data, matfilepath, metafilepath

def downsampling(filepath, stepno, dsfact = None):
    stepno = stepno.astype(str)
    mat_data , meta_data, matfilepath, metafilepath = loadalldata(filepath)
    if dsfact is None:
        dsfact = meta_data["fs"]//256

    extnstr = stepno + "_ds "
    ndpts = mat_data.shape[0]  # number of data points
    nch = mat_data.shape[1]  # number of channels

    ds_data = np.zeros((ndpts // dsfact, nch))

    dsampind = range(0, ndpts, dsfact)
    ds_data[:, :nch] = mat_data[dsampind, :nch]

    # saving the data file
    mat_data = ds_data.copy()
    del ds_data

    meta_data["fs"] = int(meta_data["fs"]) // dsfact
    filepath = filepath.with_name( extnstr + filepath.stem)
    savealldata(mat_data, meta_data, filepath)
    return mat_data, meta_data, filepath

def delete_channels(filepath,stepno,deletechnames = None):
    stepno = stepno.astype(str)
    if deletechnames == None:
        deletechnames = ['EXG5', 'EXG6', 'Status']

    mat_data , meta_data, matfilepath, metafilepath = loadalldata(filepath)
    labels = meta_data["chanlocs"]["labels"]
    indices = np.where(np.isin(deletechnames, labels))[0]
    indices = indices.astype(int)
    mask = np.ones(len(labels), dtype=bool)
    mask[indices] = False
    # deleting data
    mat_data = mat_data[:, mask]

    for key in meta_data["chanlocs"].keys():
        arr = meta_data["chanlocs"][key]
        meta_data["chanlocs"][key] = np.delete(arr, indices)

    meta_data["shape"] = mat_data.shape
    meta_data["size"] = mat_data.size
    meta_data["deleted_chnames"] = [labels[i] for i in indices]

    extnstr = stepno + "_chan_select "
    meta_data = cf.updatemetadataHistory(meta_data,extnstr)
    metafilepath = metafilepath.with_name(extnstr + metafilepath.stem)
    matfilepath = matfilepath.with_name(extnstr + matfilepath.stem)
    savealldata(mat_data, meta_data, filepath)

    print("channels being deleted are: ", meta_data["deleted_chnames"])
    print("\n data saved in ",metafilepath)
    print("\n")

    return mat_data, meta_data, matfilepath, metafilepath

def bandpassfilter(filepath,stepno,filterparameters = None):
    if filterparameters == None:
        filterparameters = [0.05, 100, 4] #lowpass, highpass and order of the filter

    stepno = stepno.astype(str)
    mat_data , meta_data, matfilepath, metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    nch = mat_data.shape[1]

    # Design Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = filterparameters[0] / nyquist
    high = filterparameters[1] / nyquist
    b, a = butter(filterparameters[2], [low, high], btype='band')
    filt_data = np.zeros(mat_data.shape)

    # Apply filter
    for chid in range(0, nch, 1):
        signal = mat_data[:, chid]
        filt_data[:, chid] = filtfilt(b, a, signal)  # bandpass filtering

    meta_data["bandpass filter param"] = filterparameters
    meta_data["fields"].update({"bandpass filter param": "low cutoff, high cutoff, filter order"})

    mat_data = filt_data.copy()
    del filt_data

    extnstr = stepno + "_fft "
    meta_data = cf.updatemetadataHistory(meta_data,extnstr)
    filepath = filepath.with_name( extnstr + filepath.stem)
    savealldata(mat_data, meta_data, filepath)

    print("data has been band pass filtered", [filterparameters[0], filterparameters[1]], "order: ", filterparameters[2])
    print("data saved as ", filepath)
    print("\n")

def notchfilter(filepath,stepno,notchparameters = None):
    if notchparameters == None:
        notchparameters = [4,50,100]


    mat_data , meta_data, matfilepath, metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    nch = mat_data.shape[1]
    an = []
    bn = []
    for numfreq in range(len(notchparameters)-1):
        an[numfreq] , bn[numfreq] = iirnotch(notchparameters[numfreq+1],notchparameters[0],fs)

    for chid in range(0, nch, 1):
        signal = mat_data[:, chid]
        for numfilter in range(len(an)):
            signal = filtfilt(bn[numfilter],an[numfilter], signal)
        mat_data[:, chid] = signal

    meta_data["notchfilter param"] = notchparameters
    meta_data["fields"].update({"notchfilter param": "slope, frequencies to filter"})

    extnstr = stepno + "_notchfilter "
    cf.updatemetadataHistory(meta_data, extnstr)
    filepath = filepath.with_name( extnstr + filepath.stem)

    savealldata(mat_data, meta_data, filepath)

    print("notch filtering done to filter out frequencies:", notchparameters[1:]," slope of the filter is: ", notchparameters[0])
    return mat_data, meta_data, matfilepath, metafilepath

def performICA(filepath, ch_name = None):

    if ch_name == None:
        ch_name = 'Fp1'

    mat_data, meta_data, matfilepath, metafilepath = loadalldata(filepath)
    # Design Butterworth bandpass filter
    lowcut = 1
    highcut = 10
    order = 4

    fs = int((meta_data["fs"]))
    slope = 2
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    fs = int((meta_data["fs"]))

    dataforICA = np.zeros(mat_data.shape)  # removing nonessential dims
    b, a = butter(order,[low, high], btype='band')
    bn, an = iirnotch(50, slope, fs)

    for chid in range(mat_data.shape[1]):
        sig = filtfilt(b, a, mat_data[:, chid])
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
    matfilepath = filepath.with_suffix(".npy")
    metafilepath = filepath.with_suffix(".pkl")
    mat_data = np.load(matfilepath)
    meta_data = cf.loadMetadata(metafilepath)
    labels = np.array(meta_data["chanlocs"]["labels"], dtype=object)
    subjid = matfilepath.stem.split()[-1]
    cf.showmeICAoverlayedondata(mat_data, ica_data, labels=labels,subjid=subjid)

def applyICA(filepath, ica, raw, rmidx):
    mat_data, meta_data = loadalldata(filepath)
    print("Removing ICA components:", rmidx)
    ica.exclude = rmidx
    raw_clean = ica.apply(raw.copy())
    mat_data = raw_clean.get_data()

    meta_data["ICA"] = rmidx
    meta_data["shape"] = mat_data.shape
    meta_data["size"] = mat_data.size
    extnstr = "ica_filt "
    meta_data = cf.updatemetadataHistory(meta_data,extnstr)
    filepath = filepath.with_name( extnstr + filepath.stem)
    mat_data = np.moveaxis(mat_data, -1, 0)
    savealldata(mat_data, meta_data, filepath)
    print("ICA performed, data saved as ", filepath)
    return mat_data, meta_data, filepath

def segmentation(filepath, stepno, startcode = None, segmentparams = None):
    stepno = stepno.astype(str)
    if startcode is None:
        startcode = np.array([10, 45, 90, 135, 200, 210, 212, 214, 216, 218, 220, 222, 224])
    endcode = startcode + 1

    if segmentparams is None:
        segmentparams = np.array[70, -2]
        # epoch length, startlatency, endlatency
        # negative latency implies the splice is made before the start time

    mat_data, meta_data, matfilepath, metafilepath =loadalldata(filepath)
    fs = meta_data["fs"]
    startlatencyforstart = segmentparams[1]
    endlatencyforstart = segmentparams[0] + segmentparams[1]

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
            valid = np.where((diffs > 0) & (np.abs(diffs - duration) < fs))[0]
            if len(valid) > 0:
                idx = valid[0]
                pairs.append((int(s + startlatencyforstart * fs), int(stops[idx])))
                used_stops[idx] = True
            else:
                # missing stop → infer
                pairs.append((int(s + startlatencyforstart * fs), int(s + endlatencyforstart * fs)))

        # Second: handle stops pairs that were never used (missing starts)
        for i, stop_used in enumerate(used_stops):
            if not stop_used:
                e = stops[i]
                pairs.append((int(e - duration * fs), int(e)))

        pairs = cf.givemeUniqueTuples(pairs, tolerance=2 * fs)
        startendsample[(eventno)] = pairs
        eventreps[eventno] = len(pairs)

    for idx, pairs_i in enumerate(startendsample):
        if len(pairs_i) not in [2, 5]:
            print(f"Warning: Event {startevents[idx]} has {len(pairs_i)} reps")
            errorflag = 1

    # so far we have removed the obtained the time stamp, sample stamp and event code of all the stim that have been presented.
    # I have then removed the time and sample stamp of all those events that are not relevant to the experiment's analysis like 21/22/50/55
    # trimming the data points

    nch = mat_data.shape[1]
    ep_data = np.zeros((duration*fs , nch, sum(eventreps)))
    ## add functionality that makes this compatible with all expts, not just this one ie. self calculating the 41 in this case
    count = 0
    eventrepdata = {}
    startendsampid = np.zeros([sum(eventreps), 2])

    for eventno in range(len(startcode)):
        numrep = []
        for iterid in range(int(eventreps[eventno])):
            dsamp_start = startendsample[eventno][iterid][0]
            dsamp_end = startendsample[eventno][iterid][1]
            startendsampid[count, 0] = dsamp_start
            startendsampid[count, 1] = dsamp_end

            eventrep_dat = np.arange(dsamp_start, dsamp_end, dtype="int")
            # print("range:", dsamp_start, "-", dsamp_end, " length:", dsamp_end - dsamp_start)
            ep_data[:, :nch, count] = mat_data[eventrep_dat, :nch]
            numrep.append(count)
            count += 1
            # final data will have dimensions like so:(ndpts, nch, nevents)
        print("Event label being split: ", startevents[eventno], " number of reps: ", len(numrep))
        eventrepdata[str(startevents[eventno])] = np.array(numrep)

    print(startendsampid)
    if errorflag == 0:
        mat_data = ep_data.copy()
        meta_data["eventrepid"] = eventrepdata

    extnstr = stepno + "_ep "
    meta_data = cf.updatemetadataHistory(meta_data, extnstr)
    filepath = filepath.with_name( extnstr + filepath.stem)
    savealldata(mat_data, meta_data, filepath)
    return mat_data, meta_data, filepath

def interpolate(filepath,stepno, interp_chnames, bad_chnames = None):
    mat_data, meta_data, matfilepath, metafilepath = loadalldata(filepath)
    stepno = stepno.astype(str)
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
        mat_data[:, interp_chids[i], :] = np.mean(mat_data[:, srtd_idx, :], axis=1)
        new_entry = {interp_chnames[i]: srt_labels[srtd_idx]}
        interp_specs.update(new_entry)
        print(badch, "is interpolated using ", [labels[i] for i in srtd_idx])

    meta_data["interpolation"] = interp_specs
    extn_str = f"{len(interp_chnames)}_interp"

    filepath = matfilepath.with_name(extn_str + filepath.stem)
    savealldata(mat_data, meta_data, filepath)
    print("channels are interpolated and saved as ", filepath)
    print("\n")
    return mat_data, meta_data, filepath, badch

def globalreferencing(filepath,stepno, badchids= None):

    mat_data, meta_data,matfilepath,metafilepath = loadalldata(filepath)
    labels = meta_data["chanlocs"]["labels"]
    goodchids = np.where(~np.isin(labels, badchids))[0]
    glreference = np.mean(mat_data[:, goodchids, :], axis=1,keepdims=True)
    mat_data = mat_data - glreference
    meta_data["ref_chids"] = goodchids
    extn_str = f"{stepno}_ref"
    filepath = filepath.with_name(extn_str + filepath.stem)
    savealldata(mat_data, meta_data, filepath)
    return mat_data, meta_data, filepath

def spearateepochs(filepath, mergekeys = None, mergekeyflag = None):
    mat_data, meta_data, matfilepath,metafilepath = loadalldata(filepath)
    subjid = metafilepath.stem.split()[-1]
    if len(subjid) != 8:  # Prevents mistaking something else as the subject id as we know that subject id is usually 8 characters long
        subjid = metafilepath.stem.split()[-2]
    print(len(subjid))

    # default value is that some events need to be merged. if this is set to 0 then no event is merged
    if mergekeys == None:
        mergekeys = {
            "TOP": [210, 212],
            "BOTTOM": [214, 216],
            "RIGHT": [218, 220],
            "LEFT": [222, 224]
        }

    slices = meta_data["eventrepid"]
    print(slices.keys())
    if mergekeyflag == True:
        keys = mergekeys.keys()
        delkeys = []
        for i in keys:
            print(i)
            a = slices[str(mergekeys[i][0])]
            b = slices[str(mergekeys[i][1])]
            ab = np.union1d(a, b)
            slices[i] = ab
            delkeys.append(str(mergekeys[i][0]))
            delkeys.append(str(mergekeys[i][1]))

        list(map(slices.pop, delkeys))

    for label, (start, end) in slices.items():
        data = mat_data[:, :, start:end]
        # print(start:end)
        filepath = metafilepath.parent / f"{label} {subjid}.npy"
        np.save(filepath, data)
        # print(f"{label}: {start}:{end} for size {data.shape[2]}")
        print(f"event {label} Saved as: {filepath}: ")
        print("\n")

    return mat_data, meta_data, filepath

def mergeepochs(folderpath,event_label):
    print(f"The event you're running this script for is {event_label}")

    files = sorted([f for f in folderpath.glob(f"{event_label}*.npy")
                    if not f.stem.endswith("merged") and not f.name.startswith(f"{event_label}_")
                    ])
    # Excludes the merged files and preexisting files that have event_label_ (e.g. "10_")
    # in front of them

    files = files[:]
    print("\n")
    # print(files)
    data_list = []
    subj_merged = []
    for file in files:
        strings = file.stem
        print(strings)
        strings = strings.split(" ")
        subjid = strings[-1]
        if len(subjid) != 8:  # Prevents taking '.pkl' as the subject id as we know that subject id is usually 8 characters long
            continue
        data = np.load(file)  # loads array saved earlier
        print(data.shape)
        data_list.append(data)
        subj_merged.append(subjid)

    tempfilepath = file
    tempfilepath = tempfilepath.with_suffix(".pkl")
    tempmeta_data = cf.loadMetadata(tempfilepath)

    mat_data = np.concatenate(data_list, axis=2)
    print(mat_data.shape)
    meta_data = {
        "event_label": event_label,
        "subjids": subj_merged,
        "shape": mat_data.shape,
        "size": mat_data.size,
        "fs": 256,
        "chanlocs": tempmeta_data["chanlocs"],
        "history": {}
    }

    filepath = folderpath/ f"{event_label} MERGED.npy"
    savealldata(mat_data, meta_data,folderpath)
    return mat_data, meta_data, filepath

def frequencytransform(filepath,stepno,freqbin = None):
    if freqbin == None:
        freqbin = [0.05, 50]

    freq_min = freqbin[0]
    freq_max = freqbin[1]
    stepno = stepno.astype(str)
    extnstr = f"{stepno}_fft"
    mat_data, meta_data, matfilepath,metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    meta_data["freq_range"] = [freq_min, freq_max]

    freqs = np.fft.fftfreq(mat_data.shape[0], 1 / fs)
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]
    FFT_data = np.zeros([len(freq_idx), mat_data.shape[1], mat_data.shape[2]])

    for chid in range(mat_data.shape[1]):
        for epochid in range(mat_data.shape[2]):
            FFT_signal = np.abs(np.fft.fft((mat_data[:, chid, epochid])))
            FFT_data[:, chid, epochid] = FFT_signal[idx]

    meta_data["freq_range"] = [freq_min, freq_max]
    mat_data = FFT_data.copy()
    cf.updatemetadataHistory(meta_data,extnstr)
    filepath = filepath.with_name(extnstr + filepath.stem)
    del FFT_data
    savealldata(mat_data, meta_data, filepath)
    return mat_data, meta_data, filepath

def averagingacrosstrials(filepath,stepno):
    stepno = stepno.astype(str)
    mat_data, meta_data, matfilepath,metafilepath = loadalldata(filepath)
    mat_data = np.mean(mat_data, axis=2)
    extnstr = f"{stepno}_avg"
    cf.updatemetadataHistory(meta_data,extnstr)
    savealldata(mat_data, meta_data, filepath)

def chunking(filepath,stepno,basefreq,window_width = None):
    stepno = stepno.astype(str)
    extnstr = f"{stepno}_chunk"
    mat_data, meta_data, matfilepath,metafilepath = loadalldata(filepath)
    fs = meta_data["fs"]
    freq_min = meta_data["freq_range"][0]
    freq_max = meta_data["freq_range"][1]
    freq_res = (freq_max-freq_min)/fs
    freqs = np.arange(freq_min, freq_max, freq_res)

    chunkWidth = window_width + freq_res
    meta_data["chunk_width"] = chunkWidth
    harmonics = np.arange(basefreq, freq_max, basefreq)
    harmonics = harmonics[harmonics <= 50]
    chunk_data = np.zeros((int(np.floor(chunkWidth / freq_res)), mat_data.shape[1], len(harmonics)))
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]

    for chid in range(mat_data.shape[1]):
        for fhid in range(len(harmonics)):
            idx = np.where(
                (freq_idx > (harmonics[fhid] - chunkWidth / 2)) & (freq_idx <= (harmonics[fhid] + chunkWidth / 2)))[0]
            # print(freq_idx[idx])
            chunk_data[:, chid, fhid] = mat_data[idx, chid]

    mat_data = chunk_data.copy()
    filepath = filepath.with_name(extnstr + filepath.stem)
    cf.updatemetadataHistory(meta_data,extnstr)
    savealldata(mat_data, meta_data, filepath)

    return mat_data, meta_data, filepath

def selectingchunks(filepath, stepno,numharmonics = None):
    if numharmonics == None:
        numharmonics = 3

    #So far, this selects only the first three harmonics of the baseline and oddball responses
    stepno = stepno.astype(str)
    mat_data, meta_data, matfilepath,metafilepath = loadalldata(filepath)
    bl_chunks = np.arange(4, mat_data.shape[2], 5)
    bl_chunkmask = np.zeros(mat_data.shape[2], dtype=bool)
    bl_chunkmask[bl_chunks] = "True"
    odd_chunkmask = ~bl_chunkmask

    bl_data = mat_data[:, :, bl_chunkmask]
    odd_data = mat_data[:, :, odd_chunkmask]
    # Select the first 3 for baseline and 12 chunks for oddball
    
    bl_data = bl_data[:, :, 0:numharmonics]
    odd_data = odd_data[:, :, 0:numharmonics]

    meta_data_bl = meta_data.copy()
    meta_data_odd = meta_data.copy()

    meta_data_bl["nChunks"] = numharmonics
    meta_data_bl["eventType"] = "Baseline"

    meta_data_odd["nChunks"] = numharmonics
    meta_data_odd["eventType"] = "Oddball"

    meta_data_bl = cf.updatemetadataHistory(meta_data_bl,f"{stepno}_baseline")
    meta_data_odd = cf.updatemetadataHistory(meta_data_odd, f"{stepno}_oddball")

    filepath_br = filepath.with_name(f"{stepno}_baseline" + filepath.stem)
    filepath_odd = filepath.with_name(f"{stepno}_oddball" + filepath.stem)

    savealldata(bl_data, meta_data_bl, filepath_br)
    savealldata(odd_data, meta_data_odd, filepath_odd)

    return bl_data, meta_data_bl, odd_data, meta_data_odd, filepath_br, filepath_odd

def sumofharmonics(filepath,stepno):
    stepno = stepno.astype(str)
    mat_data, meta_data, matfilepath,metafilepath = loadalldata(filepath)
    mat_data = np.sum(mat_data, axis=2)
    filepath = filepath.with_name(f"{stepno}_sum " + filepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,f"{stepno}_sum")
    metafilepath = metafilepath.with_name(f"{stepno}_sum " + metafilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)
    np.save(matfilepath, mat_data)
    print("Harmonics of mat_data is summed and the new shape is:", mat_data.shape)
    print("\n")
    savealldata(mat_data, meta_data, filepath)
    return mat_data, meta_data, filepath

def baselinefiltering(filepath,stepno):
    ## The logic and instructions of this is unclear, ask cedric and gloria about this -BP

