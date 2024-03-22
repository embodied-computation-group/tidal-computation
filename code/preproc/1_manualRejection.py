from peakdet import operations,  Physio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
sys.path.append('..')
import paths
import resp_cleaning as rc
from importFunctions import *

dataPath = paths.raw
saveRelevantPhysioSegments = paths.respToInspect
print(str(saveRelevantPhysioSegments))
manualDataRej = paths.badSegments

# find subject ids based on the raw files
subList = findSubjectsIDs(dataPath)
print(subList)

# identify respiratory segments that are in the trial blocks
for subject in subList:

    trialsFileName, physioFileName = findSubjectFiles(directory = dataPath, id=subject)
    trials, physio = importSubjectData(dataPath+trialsFileName, dataPath+physioFileName)

    stimIndexes = findStimIndexes(physio.stim)

    # Check if number of stimuli in the physio file match number of trials in the log
    if len(stimIndexes) == len(trials):
        trials['onset'] = stimIndexes # add trial onset times to trials df
        trials['responsI'] = rc.responsIndexes(trials.onset, trials.rt, 10)
    else:
        warnings.warn('number of trials in log file do not match number of stimuli in physio file')
    type(trials.responsI[0])

    resp = physio.respSignal
    filtered = rc.butter_lowpass_filter(resp, 1, 10, 5) # no filtering due to the low sampeling freq
    normalized = rc.zScore(filtered)

    # set resp signal value = 0 in nonrelevant segments (to avoid labelling)
    trialPhysio = np.multiply(rc.blockRangeVector(trials, len(resp), 10, 5), normalized)
    trialData = pd.DataFrame(data={'resp':trialPhysio})

    # save modded resp signal
    trialData.to_csv(str(saveRelevantPhysioSegments / f'{subject}_trial_physio.csv'))

    print(f' segments to inspect for subject {subject} saved')


for file in os.listdir(saveRelevantPhysioSegments):
    # check if bad segments has already been labled for thi sub
    if not os.path.isfile(str(manualDataRej/f'{file[:4]}_manualBads.csv')):
        print(file)
        relevantResp = pd.read_csv(str(saveRelevantPhysioSegments / file))
        sampeling_rate = 10

        resp = np.array(relevantResp.resp)
        data = Physio(resp, fs=sampeling_rate)
        data = operations.peakfind_physio(data, thresh=0.1, dist=100)
        data = operations.edit_physio(data)

        plt.plot(data)
        for seg in data.badSeg:
            plt.axvspan(seg[0], seg[1], alpha=0.5, color='red')

        badSeq = pd.DataFrame(data.badSeg)

        badSeq.to_csv(str(manualDataRej / f'{file[:4]}_manualBads.csv'))
