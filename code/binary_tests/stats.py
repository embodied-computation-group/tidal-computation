from turtle import color
import weakref
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#from numpy.core.fromnumeric import size
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import sem
#import ptitprince as ptp
import pingouin as pt
import statsmodels.formula.api as smf
from tabulate import tabulate


def subjectLvlParam(trials, behavParam, locking='respons_phase', centralTendencyMethod='mean'):
    """Calculates a subject lvl behav param in two circular windows (default = inspirateion and expiration)

    Args:
        trials (DataFrame): trial log with phase column
        window1 (tuple): lenght = 2, contains the lower an upper bounds of the 1st window
        window2 (tuple): lenght = 2, contains the lower an upper bounds of the 2nd window
        phaseParam (string): a valid column name in 'trials' containing phase information
        behavParam (string):  a valid column name in 'trials' containing behav information
        centralTendencyMethod (string): 'mean' or 'median'

    Returns:
        panda Data Frame: three columns with 'subject', 'window1' value, 'window2' value
    """
    subjects,w1s,w2s, nTrials = [],[],[], []
    for subject in trials.subject.unique():
        sData = trials.loc[trials.subject == subject]
        if centralTendencyMethod == 'mean':
            w1 = np.mean(sData.loc[sData[locking] == 'inspiration'][behavParam])
            w2 = np.mean(sData.loc[sData[locking] == 'expiration'][behavParam])
        elif centralTendencyMethod == 'median':
            w1 = np.median(sData.loc[sData[locking] == 'inspiration'][behavParam])
            w2 = np.median(sData.loc[sData[locking] == 'expiration'][behavParam])
        w1n = len(sData.loc[sData[locking] == 'inspiration'][behavParam])
        w2n = len(sData.loc[sData[locking] == 'expiration'][behavParam])
        
        subjects.append(subject)
        w1s.append(w1)
        w2s.append(w2)
        nTrials.append(w1n+w2n)

        
    outDF = pd.DataFrame(data={'subject':subjects, 'w1':w1s, 'w2':w2s, 'nTrials':nTrials})
    outDF['diff'] = outDF.w1 - outDF.w2
    
    #print(outDF)

    return outDF




def ttest(testframe):
    test = pt.ttest(testframe.w1, testframe.w2, paired=True)
    return test

def ttestAll(data, dataOrder, lockings, behavParams, centralTendencyMethods,tl, plot=False, showDiff=True, show_tests=True, box=False, ylim=None, save=False, show_sig=False):
    diffs, tests = [], []

    if plot:
        plottingData = None


    for locking, behavParam, centralTendencyMethod, set in zip(lockings, behavParams, centralTendencyMethods, dataOrder):
        sData = subjectLvlParam(data[set], behavParam, locking , centralTendencyMethod)
        diffs.append(sData['diff'])
        tests.append(ttest(sData))

        if plot:
            sData['phase_behav'] = f'{locking}_{behavParam}_{set}'

            if plottingData is None:
                plottingData = sData
            else:
                plottingData = plottingData.append(sData, ignore_index=True)
        

    
    if plot:

        #plottingData['Phase at'] = ['Response' if 'Res' in p else 'Stim' for p in plottingData['phase_behav']]
        if showDiff:
            #ax=rain(plottingData, y='diff', x='phase_behav')
            sns.set(style="white")
            #sns.despine(offset=10, trim=True)
            my_pal = {phase_behav: "b" if int(phase_behav[-1])%2==0 else "r" for phase_behav in plottingData["phase_behav"].unique()}
            if not box:
                ax = sns.violinplot(x='phase_behav', y='diff',
                                    palette=my_pal, split=True,
                                    data=plottingData,
                                    scale="count", inner="box", ax=ax)
            else:
                sns.boxplot(x = 'phase_behav', 
                y = 'diff', 
                data = plottingData,
                notch=True,
                whis=10,
                saturation=1,
                palette=my_pal,
                showmeans=True
                )
                sns.stripplot(x = "phase_behav",
                y = "diff", 
                color = 'black',
                edgecolor= 'black',
                linewidth = 1,
                size = 10,
                alpha = .5,
                jitter=0.2,
                data = plottingData,
                )
                sns.despine()
                
            if len(tl)>5:
                plt.text(len(tl)/4-.5, 0.2, 'Stimulus onset based', ha='center', va='center', size='large')
                #ax.hlines(y=-0.2, xmin=4.75, xmax=9.25, color='black')
                plt.text((len(tl)*3)/4-.5, 0.2, 'Response based', ha='center', va='center', size='large')
                plt.axvline(len(tl)/2-.5, ls='-', linewidth=5, color='black', alpha=0.1)
            plt.xlabel('phase_behav')
            if ylim is None:
                plt.ylim(-0.2,0.3)
            else:
                plt.ylim(ylim[0],ylim[1])
                
            if len(behavParams)==2:
                label_size = 40
                plt.ylabel('Delta RT (ms.)', size=label_size/1.5)
                #plt.title('Inspiration - Expiration\nmedian rt.', fontsize=label_size)
                plt.yticks(ticks=[0.025, 0, -0.025, -0.05, -0.075, -0.1, -0.125],
                           labels=['25', '0','-25' ,'-50', '-75', '-100', '-125'],
                           size=label_size/2)
                plt.xticks([0,1],labels=['RDM','FAD'], size=label_size/2)
                plt.xlabel('')



            else: 
                plt.ylabel('Insp.-Exp. difference (proportion or sec.)')
                #plt.title('Inspiration - Expiration behav.', fontsize=20)
                plt.xlabel('Behavior parameter')
                # Creating legend with color box
                res_patch = mpatches.Patch(color='b', label='RDM')
                stim_patch = mpatches.Patch(color='r', label='FAD')
                plt.legend(handles=[stim_patch, res_patch], loc='lower center')
                plt.xticks(range(len(tl)), tl)
                #ax.hlines(y=-0.2, xmin=-0.25, xmax=4.25, color='black')

            
            
        else:
            plottingData = plottingData.drop('diff', 1)
            plottingData.rename(columns = {'w1':'insp', 'w2':'exp'}, inplace = True)
            plottingData = pd.melt(plottingData, id_vars=['subject','phase_behav'], value_vars=['insp', 'exp', ])
            ax = sns.violinplot(x='phase_behav', y='value',hue="variable",
                                palette="Set2", split=True,
                                data=plottingData,
                                scale="count", inner="box")
        
        plt.axhline(0, ls='--', linewidth=5, color='black', alpha=0.5)

        if show_tests:
            loc = 0
            for t, d, mod in zip(tests, diffs, ['RDM','FAD']):
                pval = pd.to_numeric(t['p-val'])[0]
                ci1 = t['CI95%'][0][0]
                ci2 = t['CI95%'][0][1]
                print(t['CI95%'][0][0], t['CI95%'][0][1])
                w = 'heavy' if pval < 0.05 else 'normal'
                plt.text(2.5, loc, f'{mod}\nmean={round(np.mean(d),3)}\n95%-CI {ci1} to {ci2}\np={round(pval,4)}', ha='center', va='center', size='xx-large', weight=w)
                loc -= 0.05
        if show_sig:
            for t, loc in zip(tests, range(len(tl))):
                pval = pd.to_numeric(t['p-val'])[0]
                if pval < 0.05:
                    ns = '*'
                    if pval < 0.01:
                        ns = '**'
                    if pval < 0.001:
                        ns='***'
                    plt.text(loc, ylim[1], ns, ha='center', va='center', size='xx-large')
            
    
        if save is not None:
            print('trying to save')
            plt.savefig(save, bbox_inches='tight')


    return diffs, tests, sData.nTrials




def ttable(testList, dataSetNames, taskList, nameList, diffs, lockings, ms=False):
    scale = 'ms' if ms else ''
    data = []
    for test, task, name, d, locking in zip(testList, taskList, nameList, diffs, lockings):
        ci1 = test['CI95%'][0][0]
        ci2 = test['CI95%'][0][1]
        pval = pd.to_numeric(test['p-val'])[0]
        tval = pd.to_numeric(test['T'])[0]
        BF10 = pd.to_numeric(test['BF10'])[0]
        BF01 = 1/BF10

        if ms:
            sc= f'({scale})'
            ci = f'[{1000*ci1}-{1000*ci2}]'
            data.append([dataSetNames[task],locking[:-6], name , 1000*round(np.mean(d),4), ci, round(tval,4), round(pval,4) , round(BF10,1), round(BF01,2)])
        else:
            sc = ''
            ci = f'[{ci1}-{ci2}]'
            data.append([dataSetNames[task],locking[:-6], name , round(np.mean(d),4), ci, round(tval,4), round(pval,4), round(BF10,1), round(BF01,2)])
            
    print (tabulate(data, headers=['Task','Locking', 'Behav. Param.', f'Diff.'+ sc, '95%-CI', 'T', 'p-val', 'BF10', 'BF01']))