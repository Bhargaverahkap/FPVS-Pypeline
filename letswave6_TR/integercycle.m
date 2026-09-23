
function nbins = integercycle(StimFreq,SampRate,SegmentLength)

%usage:
%nbins = integercycle(StimFreq,SampRate,SegmentLength).
%
%Provides the number of time samples to use so that the duration of a
%time-series is an integer multiple of the sampling rate.
% 
% input:
%   -StimFreq = stimulation frequency
%   -SampRate = sampling rate
%   -SegmentLength = maximal duration of the time-series (in seconds).
% 
% example:
% nbins = integercycle(4,1024,40)
% 

% size_segment = input('What is the length of the desired segment (s)? ');
% freq = 6; %use Sinstim variable Stim_Freqs
freq_cycle = 1/StimFreq;
% sampling_rate = 250;

window = [0:freq_cycle:SegmentLength];
%
nbins = window(end)*SampRate;
