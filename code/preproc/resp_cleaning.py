import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks,argrelmax,argrelmin, hilbert, butter, filtfilt, freqs


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)

    # #plot filter profile
    # w, h = freqs(b, a)
    # plt.semilogx(w, 20 * np.log10(abs(h)))
    # plt.title('Butterworth filter frequency response')
    # plt.xlabel('Frequency [radians / second]')
    # plt.ylabel('Amplitude [dB]')
    # plt.margins(0, 0.1)
    # plt.grid(which='both', axis='both')
    # plt.axvline(cutoff, color='green') # cutoff frequency
    # plt.show()
    return y

def zScore(series):
    return (series - np.mean(series))/np.std(series)


def responsIndexes(onsets, rts, sFreq):
    responsI = []
    for onset, rt in zip(onsets, rts):
        if not np.isnan(rt):
            responsI.append(onset + int(round(rt * sFreq)))
        else:
            responsI.append(0)
    return responsI

def blockRanges(log, sf, add):
    addSamples = sf * add
    include = []
    for type in log.stimuli.unique():
        for b in log.block.unique():
            if not np.isnan(b):
                start = min(log.loc[(log.stimuli==type) & (log.block==b)].onset) - addSamples
                end = max(log.loc[(log.stimuli==type) & (log.block==b)].onset)  + addSamples
                include.append((start,end))
    return sorted(include, key=lambda tup: tup[0])

def blockRangeVector(log, physioLenght, sf, add):
    include = blockRanges(log, sf, add)
    vec = np.zeros(physioLenght)
    for i in include:
        vec[i[0]:i[1]] = 1
    return vec

def notBlockRanges(log, sf, add):
    block = blockRanges(log, sf, add)

    noBlock = [(0, block[0][0])]

    for i in range(len(block))[1:]:
        noBlock.append((block[i-1][1], block[i][0]))

    #noBlock.append((block[-1][1], physioLenght))

    return noBlock


def findPeaksAndTroughs(respTrace, prominence=0.2, width=2, dist=2, sFreq=1000):
    #respTrace= zScore(respTrace) # signal is zScored as part of prepross
    #respTrace = respTrace **3 # Cube of the signal with baseline as zero to enhance peaks before detection
    # Find the peak indexes
    peaks=list(find_peaks(respTrace,distance=dist*sFreq, width=width, prominence=prominence)[0])

    # Find trough indexes as index with min value between peaks
    troughs = []
    nPeak = 0
    while nPeak < len(peaks)-1:
        minValue = min(respTrace[peaks[nPeak]:peaks[nPeak+1]])
        trough = list(respTrace[peaks[nPeak]:peaks[nPeak+1]]).index(minValue) + peaks[nPeak]
        troughs.append(trough)
        nPeak += 1
    #invertedRespSignal = [-elm for elm in respTrace]            # invert the signal
    #troughs = list(find_peaks(invertedRespSignal,width=10)[0])   # find peaks

    return np.array(peaks).astype(int), np.array(troughs).astype(int)


def breathMeasures(trough, peaks, troughs, respTrace):

    peak = peaks[peaks > trough][0]
    nextTrough = troughs[troughs > peak][0]
    ptp = np.ptp(respTrace[trough : nextTrough])
    inspRange = np.ptp(respTrace[trough : peak])
    expRange = np.ptp(respTrace[peak : nextTrough])
    duration = nextTrough - trough
    inspDur = peak - trough
    expDur = nextTrough - peak
    halfInsp = np.ptp(respTrace[trough:int(trough+0.5*inspDur)]) / inspRange
    halfExp = np.ptp(respTrace[peak:int(peak+0.5*expDur)]) / expRange
    relativeRange = ptp/duration
    durRatio = inspDur / expDur

    return {'inspStart': trough, 'expStart': peak, 'expEnd': nextTrough,
            'range': ptp, 'inspRange': inspRange, 'expRange': expRange,
            'dur':duration, 'relativeRange':relativeRange, 'halfInsp':halfInsp,
            'halfExp':halfExp, 'durRatio':durRatio}


def breathDF(peaks, troughs, respTrace):
    breaths = pd.DataFrame()
    for trough in troughs[:-1]: # there will be no full breath after the last trough
        thisBreath = breathMeasures(trough, peaks, troughs, respTrace)
        breaths = breaths.append(thisBreath, ignore_index=True)
    return breaths


def keepBlockBreaths(breaths:pd.DataFrame, blockRange:list):
    keep = pd.DataFrame()

    for interval in blockRange:
        keep = keep.append(breaths.loc[(breaths.inspStart >= interval[0]) & (breaths.expEnd<=interval[1])])
        # keep = keep.append(breaths.loc[(interval[0] <= breaths['inspStart'] <= interval[1]) |
        #                                         (interval[0] <= breaths['expEnd'] <= interval[1])])

    return keep


def badManBreaths(breathsDur, breathsInspStart, breathsExpEnd, badEpochs):

    badList = []

    for breathDur, breathInspStart, breathExpEnd in zip(breathsDur, breathsInspStart, breathsExpEnd):

        appending = False
        for start, stop in zip(badEpochs['0'], badEpochs['1']):
            # print(f'breathDur + stop - start: {breathDur}+{stop}-{start}={breathDur + stop - start}')
            # print(f'max(breathExpEnd, stop) - min(breathInspStart, start: {max(breathExpEnd, stop)}-{min(breathInspStart, start)}={(max(breathExpEnd, stop)-min(breathInspStart, start))}')
            # print(f'max(breathExpEnd {breathExpEnd}, stop {stop}) - min(breathInspStart {breathInspStart}, start {start}) = {(max(breathExpEnd, stop)-min(breathInspStart, start))}')

            # see if length of the breath + length of the bad epoc is larger than the largest combination of breath and epoch start and stop
            # == if the breath overlap with the bad epoch
            if (breathDur+stop-start) > (max(breathExpEnd, stop)-min(breathInspStart, start)):
                appending = True
                badList.append(True)
                break
            # special case where just one point is overlapping
            elif (breathDur+stop-start) == (max(breathExpEnd, stop)-min(breathInspStart, start)):
                print ('Shit! they overlap by one point!')
                appending = True
                badList.append(False)
                break

        if appending:
            continue
        badList.append(False)
    return badList


def badManualTrial(trials, breaths):
    badBreaths = breaths.loc[breaths.manually_bad == True]
    badOnset = []
    badRespons = []
    for onset in trials.onset:
        bad = False
        for start, stop in zip(badBreaths.inspStart, badBreaths.expEnd):
            if start <= onset <= stop:
                bad = True
            continue
        badOnset.append(bad)
    for respons in trials.responsI:
        bad = False
        for start, stop in zip(badBreaths.inspStart, badBreaths.expEnd):
            if start <= respons <= stop:
                bad = True
            continue
        badRespons.append(bad)
    print(f"{sum(badOnset)} of {len(badOnset)}({(sum(badOnset)/len(badOnset))*100}%) had manually bad respiration signal during onset")
    print(f"{sum(badOnset)} of {len(badOnset)}({(sum(badOnset)/len(badOnset))*100}%) had manually bad respiration signal during respons")
    return badOnset, badRespons


def getRespPhaseSeries(respTrace, peaks, troughs):

    # Create list of len == len(respTrace) to loop over and add phases
    respPhaseList= [0]*len(respTrace)

    # Set the peaks as 'expiration' and troughs as 'inspiration'
    for i in range(len(respPhaseList)):
        if i in peaks:
            respPhaseList[i]="expiration"
        elif i in troughs:
            respPhaseList[i]="inspiration"

    # Fill in the phase names between peaks and troughs
    for i in range(len(respPhaseList))[1:]:
        if respPhaseList[i] == 0:
            respPhaseList[i] = respPhaseList[i-1]

    # set peaks and troughs back
    for i in range(len(respPhaseList)):
        if i in peaks:
            #print(i)
            respPhaseList[i]="peak"
        elif i in troughs:
            respPhaseList[i]="trough"
    return respPhaseList