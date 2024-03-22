import matplotlib.pyplot as plt
import numpy as np
import os
import sys
os.chdir(os.path.dirname(__file__))

sys.path.append('..')
import paths
import resp_cleaning as rc
from importFunctions import *

dataPath = paths.raw
savePath = paths.cleanSubData
subList = findSubjectsIDs(dataPath)
#subList = ['0019']#subList
manualDataRej = paths.badSegments
plot_qc = True


for sub in subList:
    if not os.path.isfile(str(savePath/f'{sub}_trials_w_physio.csv')):
        
        # import trial and physio files
        trialsFileName, physioFileName = findSubjectFiles(directory = dataPath, id=sub)

        trials, physio = importSubjectData(dataPath / trialsFileName, dataPath / physioFileName)
        stimIndexes = findStimIndexes(physio.stim)

        # Check if number of stimuli in the physio file match number of trials in the log
        if len(stimIndexes) == len(trials):
            trials['onset'] = stimIndexes # add trial onset times to trials df
            trials['responsI'] = rc.responsIndexes(trials.onset, trials.rt, 10)
            #trials['responsI'] = trials.loc[trials['responsI'].notnull(), 'responsI'] = trials.loc[trials['responsI'].notnull(), 'responsI'].apply(int)
        else:
            warnings.warn('number of trials in log file do not match number of stimuli in physio file')
        type(trials.responsI[0])
        
        
        # pre proc resp signal
        resp = physio.respSignal
        filtered = rc.butter_lowpass_filter(resp, 1, 10, 5) # no filtering due to the low sampeling freq
        normalized = rc.zScore(filtered)
        
        # find peaks and troughs
        peaks, troughs = rc.findPeaksAndTroughs(normalized, prominence=0.2,  dist=1, sFreq=10)
        
        # import manually labled bad segment indexes
        if os.path.isfile(str(manualDataRej / f'{sub}_manualBads.csv')):
            manBad = pd.read_csv(str(manualDataRej / f'{sub}_manualBads.csv'))
        else:
            manBad = None
            warnings.warn(f'no manual file for bad physio segments found for sub {sub}')
        
        # set trials with onset in a manually labled bad breath as manBad = True         
        breaths = rc.breathDF(peaks, troughs, resp)
        blockBreaths = rc.keepBlockBreaths(breaths, rc.blockRanges(trials, 10, 5)) # remove breaths outside of trial blocks

        if manBad is not None:
            blockBreaths['manually_bad'] = rc.badManBreaths(blockBreaths.dur, blockBreaths.inspStart, blockBreaths.expEnd, manBad)
            goodBreaths = blockBreaths.loc[blockBreaths.manually_bad == False]
            manBadBreaths = blockBreaths.loc[blockBreaths.manually_bad == True]
        print(len(breaths), len(blockBreaths), len(goodBreaths))
        trials['manBadOnset'], trials['manBadRespons'] = rc.badManualTrial(trials, blockBreaths)
        
        
        # inspiration-expiration info for every sample
        phaseSeries = rc.getRespPhaseSeries(normalized, peaks, troughs)
        # inspiration-expiration info for trial onsets
        phaseListOnset = [phaseSeries[int(start)] for start in trials.onset] 
        # inspiration-expiration info for trial responses
        phaseListRespons = [phaseSeries[int(start)] for start in trials.responsI] 

        # add inspiration-expiration info to trial df
        trials['onset_phase'] = phaseListOnset
        trials['respons_phase'] = phaseListRespons
        
        # save trial df with inspiration-expiration info
        trials.to_csv(str(savePath/f'{sub}_trials_w_physio.csv'))
           

        # plot qc image og peaks, troughs and bad segments
        noBlockAreas = rc.notBlockRanges(trials, sf=10, add=5) # times outside trial blocks
        bc = noBlockAreas[0][1]
        time= np.linspace(0, len(resp[bc:]), len(resp[bc:]))
        
        nPlots = int(len(time)/2000)+1
        
        plt.rcParams["figure.figsize"] = (25,35)
        if plot_qc:
            fig, axes = plt.subplots(nPlots,1,  figsize=(15, 25), sharey=True, sharex=False,
                    gridspec_kw={ 'wspace':0.1, 'hspace':0.1})
            
            for i in range(nPlots):
                print
                
                # shade no trial areas in red
                for area in noBlockAreas:
                    axes[i].axvspan(area[0]-bc, area[1]-bc, alpha=0.5, color='red')

                # shade manually bad areas in blue
                if manBad is not None:    
                    for start, stop in zip(manBad['0'], manBad['1']):
                        start, stop = start-bc, stop-bc
                        axes[i].axvspan(start, stop, alpha=0.5, color='blue')

                axes[i].plot(time, normalized[bc:])
                #plt.plot(time,dif*5)
                #plt.plot(time, signChange*2)
                for peak in peaks:
                    axes[i].axvline(peak-bc, color='black', alpha=1)
                for trough in troughs:
                    axes[i].axvline(np.array(trough-bc), color='blue', alpha=1)
            
                axes[i].set_xlim(2000*i,2000*(i+1))
                #plt.ylim(-1,2)
            fig.savefig(str(paths.peak_trough_fig / f'{sub}_peak_trough.png'))
            plt.close('all')
    else:
        print(str(savePath/f'{sub}_trials_w_physio.csv') + ' already exists')