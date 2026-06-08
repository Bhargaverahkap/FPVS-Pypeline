FPVS Preprocessing — Letswave6 to Python

A Python port of the EEG preprocessing pipeline used in Letswave6 for a Fast Periodic Visual Stimulation (FPVS) face-categorization task.

Participants viewed a continuous stream of non-face images at a base rate of 6 Hz. Every 5th image was a face, so faces appeared periodically at 6/5 = 1.2 Hz. Because the stimulation is frequency-tagged, the face-selective response can be read directly in the frequency domain at 1.2 Hz and its harmonics, while the general visual response appears at 6 Hz and its harmonics.
This project reimplements the Letswave6 preprocessing and analysis chain in Python so that it is scriptable, reproducible, and version-controlled.
Pipeline

1. Rename channels
2. Set electrode locations (montage)
3. Remove unused channels
4. Butterworth band-pass filtering (0.05–100 Hz)
5. Notch filtering (50 and 100 Hz, width 0.05 Hz, slope 2 Hz)
6. Downsampling to 256 Hz
7. Segmentation of events (2 s before start + 2 s fade-in + 64 s block + 2 s fade-out = 70 s, starting at −2 s)
(Optional) ICA
(Optional) Channel interpolation
8. Re-reference
9. Segmentation to an exact number of bins (16213 bins, starting at 2 s, one file per condition)
10. Fourier transform
11. Average trials within each condition
12. Segmentation by chunking (onset 1.0037, duration 0.394671868, interval 1.201)
13. Selection of harmonics (significant harmonics only, split into baseline and oddball)
14. Sum of harmonics
Baseline correction


EEG data for two participants are available on OSF: https://doi.org/10.17605/OSF.IO/EH9CQ

