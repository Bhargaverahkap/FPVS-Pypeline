# A package meant to be used
# here i want to create a set of functions that does all my preprocessing and post processing by simply inputting the mat and lw6 filepath and a configuration/constants file.
# This way I can have a GUI where i want to and parallely get only the finished data files.

def preprocessFPVSdata_phase1(matfilepath = None, metafilepath = None, configfilepath = None):
    """
    This function completes the preproccessing steps from loading,
    :param matfilepath:
    :param metafilepath:
    :param configfilepath:
    :return:
    """
    from pathlib import Path
    import h5py
    import numpy as np
    from scipy.signal import butter, filtfilt, iirnotch
    import importlib
    import cust_funcs as cf
    importlib.reload(cf)

    #Checks to make sure that matfile/metafile are really what they claim they are.
    if matfilepath.suffix != '.mat':
        matfilepath = metafilepath.with_suffix('.mat')

    if metafilepath.suffix != '.lw6':
        metafilepath = matfilepath.with_suffix('.lw6')

    with h5py.File(matfilepath, 'r') as f:
        # List all variables
        mat_data = f['data'][:]

    # This section cleans the data and meta data to make it more python friendly
    mat_data = np.squeeze(mat_data)
    meta_data = cf.rebrand_lw6data(metafilepath)

    #renaming the mat/lw6 filenames for saving the intermediary steps.
    metafilepath = metafilepath.with_suffix(".pkl")
    matfilepath = matfilepath.with_suffix(".npy")
    print(metafilepath)
    cf.saveMetadata(meta_data,metafilepath)

    np.save(matfilepath, mat_data)
    print("data is saved as: ", metafilepath)
    print("\n")

    #1. rename channels
    labels  = (meta_data["chanlocs"]["labels"])
    targets = ["EXG1", "EXG2", "EXG3", "EXG4"]
    replacements = ["I1", "I2", "PO9", "PO10"]
    mapping = dict(zip(targets, replacements))
    labels = [mapping.get(label, label) for label in labels]
    meta_data["chanlocs"]["labels"] = labels

    meta_data = cf.updatemetadataHistory(meta_data,"1_chanlabels")
    metafilepath = metafilepath.with_name("1_chanlabels " + metafilepath.stem + metafilepath.suffix)
    metafilepath = metafilepath.with_suffix(".pkl")

    matfilepath = matfilepath.with_name("1_chanlabels " + matfilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)
    print("channels are renamed and saved in and as: ", metafilepath)
    print("\n")

    #2. Electrode location change
    electrodePath = Path(
        "D:\Goffaux lab\letswave6_TR\\resources\electrodes\spherical_locations\\biosemi_locations_64_10-20_fixP9P10_add4.xyz")
    electrode_data = np.loadtxt(electrodePath, dtype=str, max_rows=68)

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
    meta_data = cf.updatemetadataHistory(meta_data,"2_chanlocs")
    metafilepath = metafilepath.with_name("2_chanlocs " + metafilepath.stem)

    matfilepath = matfilepath.with_name("2_chanlocs " + matfilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)
    print("elec. locations are changed! and saved as ", metafilepath)
    print("\n")

    #3. Downsampling the data
    dsfact = 8  # downsampling factor
    ndpts = mat_data.shape[0]  # number of data points
    nch = mat_data.shape[1]  # number of channels

    ds_data = np.zeros((ndpts // dsfact, nch))

    dsampind = range(0, ndpts, dsfact)
    ds_data[:, :nch] = mat_data[dsampind, :nch]

    # saving the data file
    mat_data = ds_data.copy()
    del ds_data

    matfilepath = matfilepath.with_name("3_ds " + matfilepath.stem)
    np.save(matfilepath, mat_data)

    # saving lw6_file
    meta_data["fs"] = int(meta_data["fs"]) // dsfact
    meta_data = cf.updatemetadataHistory(meta_data,"3_ds")
    metafilepath = metafilepath.with_name("3_ds " + metafilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    print("Data is downsampled! by a factor of ",dsfact ," and saved as ", metafilepath)
    print("\n")

    #4. delete exg and status channels:
    labels = meta_data["chanlocs"]["labels"]
    exgindices = np.where(np.char.find(np.char.lower(labels), 'ex') != -1)[0]
    statindice = np.where(np.char.find(np.char.lower(labels), 'status') != -1)[0]

    todeletechid = np.union1d(exgindices, statindice)
    todeletechid = todeletechid.astype(int)

    mask = np.ones(len(labels), dtype=bool)
    mask[todeletechid] = False

    # deleting data
    mat_data = mat_data[:, mask]

    for key in meta_data["chanlocs"].keys():
        arr = meta_data["chanlocs"][key]
        meta_data["chanlocs"][key] = np.delete(arr, todeletechid)


    meta_data["shape"] = mat_data.shape
    meta_data["size"] = mat_data.size
    meta_data["deleted_chnames"] = [labels[i] for i in todeletechid]
    meta_data = cf.updatemetadataHistory(meta_data,"4_chan-select")
    metafilepath = metafilepath.with_name("4_chan-select " + metafilepath.stem)
    matfilepath = matfilepath.with_name("4_chan-select " + matfilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)
    print("channels being deleted are: ", meta_data["deleted_chnames"])
    print("\n data saved in ",metafilepath)
    print("\n")


    #5. bandpass and 6. notch filtering
    # declaring variables
    fs = (meta_data["fs"])
    lowcut = 0.05
    highcut = 100
    order = 4
    nch = mat_data.shape[1]

    # Design Butterworth bandpass filter
    slope = 2
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    notch_param = [50,100]
    b, a = butter(order, [low, high], btype='band')
    bn, an = iirnotch(notch_param[0], slope, fs)
    bn2, an2 = iirnotch(notch_param[1], slope, fs)
    filt_data = np.zeros(mat_data.shape)

    # Apply filter
    for chid in range(0, nch, 1):
        signal = mat_data[:, chid]
        filtered_signal = filtfilt(b, a, signal)  # bandpass filtering
        filtered_signal2 = filtfilt(bn, an, filtered_signal)  # 50hz notch filtering
        filtered_signal3 = filtfilt(bn2, an2, filtered_signal2)  # 100hz notch filtering
        filt_data[:, chid] = filtered_signal3

    meta_data["bandpass filter param"] =  [lowcut, highcut, order]
    meta_data["notch filter param"] =  [notch_param, slope]
    meta_data["fields"].update({"bandpass filter param": "low cutoff, high cutoff, filter order" , "notch filter param": "frequencies filtered out, slope of the filter"})

    mat_data = filt_data.copy()
    del filt_data

    metafilepath = metafilepath.with_name("6_fft-notchfilter 5_but " + metafilepath.stem )
    matfilepath = matfilepath.with_name("6_fft-notchfilter 5_but " + matfilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)
    print("data has been band pass filtered", [lowcut, highcut], "notch filtered at 50,100hz")
    metafilepath = metafilepath.with_suffix(".npy")
    print("data saved as ", metafilepath)
    print("\n")
    return mat_data, meta_data, metafilepath

def preprocessFPVSdata_performICA(matfilepath, metafilepath, ch_name = None):
    # this is the second phase of the preprocess, where
    from pathlib import Path

    import numpy as np
    from scipy.signal import butter, filtfilt, iirnotch
    import mne
    from mne.preprocessing import ICA
    import importlib
    import cust_funcs as cf
    importlib.reload(cf)

    if ch_name == None:
        ch_name = 'Fp1'

    meta_data = np.load(metafilepath, allow_pickle= True)
    mat_data = np.load(matfilepath)

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

def overlayICAondata(matfilepath,metafilepath, ica_data):
    import numpy as np
    import cust_funcs as cf
    import importlib
    importlib.reload(cf)

    mat_data = np.load(matfilepath)
    meta_data = cf.loadMetadata(metafilepath)
    labels = np.array(meta_data["chanlocs"]["labels"], dtype=object)
    subjid = matfilepath.stem.split()[-1]
    cf.showmeICAoverlayedondata(mat_data, ica_data, labels=labels,subjid=subjid)

def preprocessFPVSdata_applyICA(matfilepath, metafilepath, ica, raw, rmidx):
    import numpy as np
    import importlib
    import cust_funcs as cf
    importlib.reload(cf)

    print("Removing ICA components:", rmidx)
    ica.exclude = rmidx
    raw_clean = ica.apply(raw.copy())
    mat_data = raw_clean.get_data()
    meta_data = cf.loadMetadata(metafilepath)



    meta_data["ICA"] = rmidx
    meta_data["shape"] = mat_data.shape
    meta_data["size"] = mat_data.size
    meta_data = cf.updatemetadataHistory(meta_data,"ica_filt")
    matfilepath = matfilepath.with_name("ica_filt" + matfilepath.stem)
    metafilepath = matfilepath.with_name("ica_filt" + metafilepath.stem)

    mat_data = np.moveaxis(mat_data, -1, 0)
    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)
    print("ICA performed, data saved as ", metafilepath)
    metafilepath = metafilepath.with_suffix(".npy")

    return mat_data, meta_data, metafilepath

def preprocesssFPVSdata_segmentation(matfilepath, metafilepath):

    import numpy as np
    import cust_funcs as cf
    import importlib
    importlib.reload(cf)

    meta_data = cf.loadMetadata(metafilepath)
    mat_data = np.load(matfilepath)
    errorflag = 0
    fs = (meta_data["fs"])  # Hz
    eventid = np.array(meta_data["events"]["code"])
    timestamps = np.array(meta_data["events"]["latency"], dtype=float)
    duration = 70 * fs

    unq_eventid = np.unique(eventid)
    mask = np.array([x.isnumeric() for x in unq_eventid])
    unq_eventid = unq_eventid[mask]
    unq_eventid = unq_eventid.astype(int)
    unq_eventid.sort()
    unq_eventid = unq_eventid.astype(str)

    startcode = np.array([10, 45, 90, 135, 200, 210, 212, 214, 216, 218, 220, 222, 224])
    startcode_ind = np.where(np.isin(unq_eventid.astype(int), startcode))[0]
    startevents = unq_eventid[startcode_ind]
    startevents.sort()

    endcode = np.array([10, 45, 90, 135, 200, 210, 212, 214, 216, 218, 220, 222, 224]) + 1
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
                pairs.append((int(s - 2 * fs), int(stops[idx])))
                used_stops[idx] = True
            else:
                # missing stop → infer
                pairs.append((int(s - 2 * fs), int(s + 68 * fs)))

        # Second: handle stops pairs that were never used (missing starts)
        for i, stop_used in enumerate(used_stops):
            if not stop_used:
                e = stops[i]
                pairs.append((int(e - 70 * fs), int(e)))

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
    ep_data = np.zeros((17920, nch, 41))
    ## add functionality that makes this compatible with all expts, not just this one ie. self calculating the 41 in this case
    count = 0
    eventrepdata = {}
    startendsampid = np.zeros([41, 2])
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
            # final data will have dimensions like so:(17920, 70, 41)
        print("Event label being split: ", startevents[eventno], " number of reps: ", len(numrep))
        eventrepdata[str(startevents[eventno])] = np.array(numrep)

    print(startendsampid)
    if errorflag == 0:
        mat_data = ep_data.copy()
        meta_data["eventrepid"] = eventrepdata

    meta_data = cf.updatemetadataHistory(meta_data, "7_ep")
    metafilepath = metafilepath.with_name("7_ep " + metafilepath.stem)
    matfilepath = matfilepath.with_name("7_ep " + matfilepath.stem)

    matfilepath = matfilepath.with_suffix(".npy")
    metafilepath = metafilepath.with_suffix(".pkl")
    np.save(matfilepath, mat_data)
    cf.saveMetadata(meta_data, metafilepath)

    return mat_data, meta_data, metafilepath

def preprocessFPVSdata_phase2(matfilepath, metafilepath,interp_chnames = None, bad_but_ignore = None, mergekeyflag = None , mergekeys = None):
    import numpy as np
    import importlib
    import cust_funcs as cf
    importlib.reload(cf)
    if mergekeyflag == None:
        mergekeyflag = True

    mat_data = np.load(matfilepath)
    meta_data = cf.loadMetadata(metafilepath)
    labels = (meta_data["chanlocs"]["labels"])
    interp_chids = np.where(np.isin(labels, interp_chnames))[0]
    
    # interpolation
    badch = np.union1d(interp_chnames, bad_but_ignore)
    interp_specs = {}
    for i in range(len(interp_chnames)):
        srt_idx, srt_labels = cf.givemeNNearestNeighbour(meta_data, interp_chids[i])
        srtd_idx = np.where(~np.isin(srt_labels, badch))[0]
        srtd_idx = srtd_idx[:3]
        badch = np.append(badch, srt_labels[srtd_idx])
        mat_data[:, interp_chids[i], :] = np.mean(mat_data[:, srtd_idx, :], axis=1)
        new_entry = {interp_chnames[i] : srt_labels[srtd_idx]}
        interp_specs.update(new_entry)
        print(badch, "is interpolated using ", [labels[i] for i in srtd_idx])

    meta_data["interpolation"] = interp_specs
    extn_str = f"{len(interp_chnames)}_interp "

    metafilepath = matfilepath.with_name(extn_str + metafilepath.stem)
    meta_data = cf.updatemetadataHistory(meta_data,extn_str)
    cf.saveMetadata(meta_data,metafilepath)

    matfilepath = matfilepath.with_name(extn_str + matfilepath.stem)
    np.save(matfilepath, mat_data)
    print("channels are interpolated and saved as ", metafilepath)
    print("\n")

    goodchids = np.where(~np.isin(labels, badch))[0]

    #referencing
    glreference = np.mean(mat_data[:, goodchids, :], axis=1,keepdims=True)
    mat_data = mat_data - glreference

    meta_data["ref_chids"] = goodchids
    matfilepath = matfilepath.with_name("8_ref" + matfilepath.stem)
    metafilepath = metafilepath.with_name("8_ref" + metafilepath)
    meta_data = cf.updatemetadataHistory(meta_data,"8_ref")

    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)
    print("data is referenced and saved as ", metafilepath)
    print("\n")

    ## Segmentation
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

    return mat_data, meta_data, metafilepath

def postprocessFPVSdata(event_label, folderpath):
    #A lot of the meta_data updating still remains and is pending on this step
    import numpy as np
    import importlib
    import cust_funcs as cf
    importlib.reload(cf)

    print(f" the event you're running this script for is {event_label}")

    files = sorted(folderpath.glob(f"{event_label}*.npy"))
    print(*files, sep='\n')
    files = files[:]
    print("\n")
    data_list = []
    subj_merged = []
    for file in files:
        strings = file.stem
        strings = strings.split(" ")
        subjid = strings[-1]
        if len(subjid) != 8:  # Prevents taking 'lw6' as the subject id as we know that subject id is usually 8 characters long
            continue
        data = np.load(file)  # loads array saved earlier
        data_list.append(data)
        subj_merged.append(subjid)

    mat_data = np.concatenate(data_list, axis=2)

    meta_data = {
        "event_label": event_label,
        "subjids": subj_merged,
        "shape": mat_data.shape,
        "size": mat_data.size,
        "fs" : 256,
        "history" :{}
    }

    matfilepath = folderpath / f"{event_label} MERGED.npy"
    metafilepath = folderpath / f"{event_label} MERGED .pkl"
    np.save(matfilepath, mat_data)
    cf.saveMetadata(meta_data,metafilepath)


    print("data is merged based on epochs, saved as: ",matfilepath)
    print("\n")

    fs  = meta_data["fs"]
    freq_min = 0
    freq_max = 50
    meta_data["freq_range"] = [freq_min, freq_max]

    freqs = np.fft.fftfreq(mat_data.shape[0], 1 / fs)
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_idx = freqs[idx]
    FFT_data = np.zeros([len(freq_idx), mat_data.shape[1], mat_data.shape[2]])

    for chid in range(mat_data.shape[1]):
        for epochid in range(mat_data.shape[2]):
            FFT_signal = np.abs(np.fft.fft((mat_data[:, chid, epochid])))
            FFT_data[:, chid, epochid] = FFT_signal[idx]

    mat_data = FFT_data.copy()
    del FFT_data
    matfilepath = matfilepath.with_name("10_FFT " + matfilepath.stem)

    meta_data = cf.updatemetadataHistory(meta_data,"10_FFT")
    metafilepath = matfilepath.with_name("10_FFT " + metafilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    np.save(matfilepath, mat_data)

    print("FFT data is saved as ", matfilepath)
    print("\n")

    #11. averaging within trials
    avg_data = np.mean(mat_data, axis=2)
    mat_data = avg_data.copy()

    matfilepath = matfilepath.with_name("11_avg " + matfilepath.stem)
    np.save(matfilepath, mat_data)

    meta_data = cf.updatemetadataHistory(meta_data,"11_avg")
    metafilepath = metafilepath.with_name(matfilepath)

    cf.saveMetadata(meta_data,metafilepath)


    print("averaged data is saved as ", matfilepath)
    print("\n")

    ## Chunking
    freq_res = (freq_max - freq_min) / mat_data.shape[0]
    window_width = .4
    chunkWidth = window_width + freq_res
    meta_data["chunk_width"] = chunkWidth
    harmonics = np.arange(1.2, 50, 1.2)
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
    matfilepath = matfilepath.with_name("12_chunk " + matfilepath.stem)
    np.save(matfilepath, mat_data)

    meta_data = cf.updatemetadataHistory(meta_data,"12_chunk")
    metafilepath = metafilepath.with_name("12_chunk " + metafilepath.stem)
    cf.saveMetadata(meta_data,metafilepath)


    print("data is chunked at frequencies of interest and saved as ", matfilepath)
    print("\n")

    #selecting chunks for the data
    bl_chunks = np.arange(4, mat_data.shape[2], 5)
    bl_chunkmask = np.zeros(mat_data.shape[2], dtype=bool)
    bl_chunkmask[bl_chunks] = "True"
    odd_chunkmask = ~bl_chunkmask

    bl_data = mat_data[:, :, bl_chunkmask]
    odd_data = mat_data[:, :, odd_chunkmask]

    # Select the first 3 for baseline and 12 chunks for oddball
    blchunkselect = 3
    oddchunkselect = 3
    bl_data = bl_data[:, :, 0:blchunkselect]
    odd_data = odd_data[:, :, 0:oddchunkselect]

    meta_data_bl = meta_data.copy()
    meta_data_odd = meta_data.copy()

    meta_data_bl["nChunks"] = blchunkselect
    meta_data_bl["eventType"] = "Baseline"

    meta_data_odd["nChunks"] = oddchunkselect
    meta_data_odd["eventType"] = "Oddball"

    meta_data_bl = cf.updatemetadataHistory(meta_data_bl,"13_baseline")
    meta_data_odd = cf.updatemetadataHistory(meta_data_odd, "13_oddball")
    blmatfilepath = matfilepath.with_name("13_baseline " + matfilepath.stem)
    np.save(blmatfilepath, bl_data)
    blmetadatafilepath = metafilepath.with_name("13_baseline " + metafilepath.stem)
    cf.saveMetadata(blmetadatafilepath, meta_data_bl)

    print("baseline data saved, size of the data in ",blmatfilepath)
    print("\n")

    oddmatfilepath = matfilepath.with_name("13_oddball " + matfilepath.stem)
    np.save(oddmatfilepath, odd_data)
    oddmetafilepath = metafilepath.with_name("13_oddball " + metafilepath.stem)
    cf.saveMetadata(oddmetafilepath, meta_data_odd)
    print("oddball data saved, size of the data in ",oddmatfilepath)
    print("\n")

    ## Sum of harmonics
    bl_data = np.sum(bl_data, axis=2)
    odd_data = np.sum(odd_data, axis=2)

    blmatfilepath = blmatfilepath.with_name("14_sum " + blmatfilepath.stem)
    meta_data_bl = cf.updatemetadataHistory(meta_data_bl,"14_sum")
    blmetadatafilepath = blmetadatafilepath.with_name("14_sum " + blmetadatafilepath.stem)
    np.save(blmetadatafilepath, meta_data_bl)
    np.save(blmatfilepath, bl_data)
    print("Harmonics of bl_data is added and the new shape is:", bl_data.shape)
    print("\n")

    oddmatfilepath = oddmatfilepath.with_name("14_sum " + oddmatfilepath.stem)
    oddmetafilepath = oddmetafilepath.with_name("14_sum " + oddmetafilepath.stem)
    np.save(oddmatfilepath, odd_data)
    meta_data_odd = cf.updatemetadataHistory(meta_data_odd, "14_sum")
    np.save(oddmetafilepath, meta_data_odd)
    print("Harmonics of odd_data is added and the new shape is:", odd_data.shape)
    print("\n")
    #Technically the pipeline is incomplete needs to be completed

    return odd_data, bl_data, oddmatfilepath, blmatfilepath