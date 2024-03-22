import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm


def PSI_psychometric( x, alpha, beta, delta, min):
        _normCdf = norm.cdf(x, alpha, beta)
        if min==0:
            return .5 * delta + (1 - delta) * _normCdf
        elif min ==0.5:
            return .5 * delta + (1 - delta) * (.5 + .5 * _normCdf)

def thresholds_combined(data, tasks, show_mean=True, individual_thresholds= True, color_curves=True, fontsize=20, nSubjects=None, save=None):
    fig, axes = plt.subplots(2,1, figsize=(15, 15), sharey=False, sharex=False, gridspec_kw={ })
    fig.tight_layout(pad=10.0, rect=[0.05, 0.05, 0.95, 1])
    fig.suptitle('Psychometric functions', fontsize=fontsize*1.5)
    
    sns.set(style="white")

    for task in range(len(tasks)):

        if tasks[task] == 'emotion':
            stimRange = (0,200) 
            delta = 0.02
            min_response = 0
            thresholds = [0.35, 0.45, 0.55, 0.65]
            p_correction = 2
        elif tasks[task] == 'cRDM':
            stimRange = (0,.8)
            delta = 0.05
            min_response = 0.5
            thresholds = [0.75]
            p_correction = 1



        meanThresEst = None
        meanThres = None
        meanSlope = None
        n = len(data.subject.unique()) if nSubjects is None else nSubjects

        color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
        for sub in data.subject.unique()[:n]:
            currentStaircase = data.loc[(data["trialType"]=="psi_staircase") & (data.subject==sub) & (data.stimuli==tasks[task])]              
            currentStaircase['trialNumber']=range(currentStaircase.shape[0])
            c = next(color)

            stim_lvl =np.linspace(stimRange[0],stimRange[1],1000)
            alpha = list(currentStaircase.estThres)[-1]
            beta = list(currentStaircase.estSlope)[-1]

            psychometric = PSI_psychometric(x=stim_lvl, alpha=alpha, beta=beta, delta=delta, min=min_response)
            curveColor = c if color_curves else 'k'
            axes[task].plot(stim_lvl, psychometric,  c='k', alpha=.2, linewidth=2.5, zorder=0)

            # if individual_thresholds:
            #     for thres in thresholds:
            #         thres_stim_lvl = stim_lvl[(np.abs(psychometric - thres)).argmin()]
            #         axes[task].scatter(thres_stim_lvl, thres,  alpha=0.5, c=c, s=500)
            

            if meanThresEst is None:
                meanThresEst = np.array(currentStaircase.estThres[:49])
                meanThres = [list(currentStaircase.estThres)[-1]]
                meanSlope = [list(currentStaircase.estSlope)[-1]]
            else:
                meanThresEst = np.vstack([meanThresEst, np.array(currentStaircase.estThres[:49])])
                meanThres.append(list(currentStaircase.estThres)[-1])
                meanSlope.append(list(currentStaircase.estSlope)[-1])


        sdThresEst = round(np.std(meanThres), 2)
        minThresEst = round(np.min(meanThres), 2)
        maxThresEst = round(np.max(meanThres), 2)
        meanThres= np.mean(meanThres)
        
        print(f'{task}: mean {round(meanThres/p_correction, 2)}, SD {sdThresEst/p_correction}, range {minThresEst/p_correction}-{maxThresEst/p_correction}')
        
        meanThresEst = np.mean(meanThresEst, axis=0)
        meanSlope = np.mean(meanSlope)
        


        
        
        if show_mean:
            mean_psychometric = PSI_psychometric(stim_lvl, meanThres, meanSlope, delta, min_response)
            print(task)
            if tasks[task]== 'emotion':
                tc = ['blueviolet', 'plum', 'palegreen', 'darkgreen']
            if tasks[task] == 'cRDM':
                tc = ['blueviolet', 'darkgreen']
            for thres, this_c in zip(thresholds,tc):
                    thres_stim_lvl = stim_lvl[(np.abs(mean_psychometric - thres)).argmin()]
                    if tasks[task]== 'emotion':
                        axes[task].scatter(thres_stim_lvl, thres,  alpha=1, s=500, c=this_c, zorder=1000)
                    elif tasks[task]== 'cRDM':
                        axes[task].scatter(thres_stim_lvl, thres,  alpha=1, s=500, zorder=1000, marker=11, c=tc[0])
                        axes[task].scatter(thres_stim_lvl, thres,  alpha=1, s=500, zorder=1000, marker = 10, c=tc[1])
                    
                                
            axes[task].plot(stim_lvl, mean_psychometric,
                                c='lightcoral', alpha=1, linewidth=5,  linestyle='dashed', zorder=50)
            
        if tasks[task] == 'emotion':
            axes[task].set_ylabel('P(response = happy)', fontsize=fontsize)
            axes[task].set_xlabel('Face Happiness (%)', fontsize=fontsize)
            axes[task].set_title('FAD', fontsize=fontsize, y=0.975)
            axes[task].set_xticks([0,50,100,150,200])
            axes[task].set_xticklabels(['0', '25', '50','75', '100'])
            axes[task].set_yticks([0.00, 0.50, 1.00])
            axes[task].set_yticklabels(['0.00', '0.50', '1.00'])




        if tasks[task] == 'cRDM':
            axes[task].set_ylabel('P(response = correct)', fontsize=fontsize)
            axes[task].set_xlabel('Proportion coherent motion', fontsize=fontsize)
            axes[task].set_title('RDM', fontsize=fontsize, y=0.975)
            axes[task].set_yticks([0.5, 0.75, 1.0])


            
        #axes[task].yaxis.set_major_formatter('{x:0<4.1f}')
        axes[task].tick_params(axis='both', which='major', labelsize=fontsize)




        # axes[0].set_ylabel('Stimulus level', fontsize=20)
        # axes[0].set_xlabel('Trial number', fontsize=20)

        # axes[0].set_ylim([stimRange[0],stimRange[1]])
        # axes[0].spines['top'].set_visible(False)
        # axes[0].spines['right'].set_visible(False)
        axes[task].spines['top'].set_visible(False)
        axes[task].spines['right'].set_visible(False)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')


def findStimIndexes(stimSeries):
    stimIndexList =[]
    previous = 0
    for i in range(len(stimSeries)):
        if stimSeries[i] !=0 and previous == 0:
            stimIndexList.append(i)
        previous = stimSeries[i]
    return stimIndexList