import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm
import seaborn as sns

########## data fram wrangling  ###########
def getSubjectDFs(trialsDataframe):
    """Seperate trial list with trials from all subjects and return them in a list

    Args:
        trialsDataframe ([pd.DataFrame]): [data frame with all triaxs from all subjects]

    Returns:
        [list]: [list of individual subject dataframes]
    """
    subjetsDFs = []    
    for subject in trialsDataframe.subject.unique():
        subjetsDFs.append(trialsDataframe.loc[trialsDataframe.subject == subject])
    
    return subjetsDFs

def getExperimentTrials(trialsDataframe):
    return trialsDataframe.loc[trialsDataframe.trialType == 'experiment']

def getModalityTrials(trialsDataframe, modal):
    return trialsDataframe.loc[trialsDataframe.stimuli == modal]


#######  extract subject lvl info ########
def criterion(TP, FA):
    return -(norm.ppf(TP) + norm.ppf(FA))/2

def dPrime(TP, FA):
    return norm.ppf(TP) - norm.ppf(FA)

def getSubjectLvlData(trialsDataFrame, fastRT=10):
    df = pd.DataFrame(columns=[
        'subject',
        'missedEmo',
        'missedRDM',
        'RDMcriterion',
        'emoLowRTs',
        'rdmLowRTs',
        'rdmDPrime',
        'rdmAccuracy',
  
        'up_prop',
      
        ]
    )
    for subjectDF in getSubjectDFs(trialsDataFrame):
        sID = subjectDF.subject.unique()[0]
        subjectDF = getExperimentTrials(subjectDF)
        emotion = getModalityTrials(subjectDF, 'emotion')
        rdm = getModalityTrials(subjectDF, 'cRDM')

        emoMissed = len(emotion) - emotion.response.describe()[0]
        rdmMissed = len(rdm) - rdm.response.describe()[0]

        hitRate = len(rdm.loc[(rdm.response == 'up') & (rdm.rand == 'up')]) / len(rdm.loc[rdm.rand == 'up'])
        falseAlarmRate = len(rdm.loc[(rdm.response == 'up') & (rdm.rand == 'down')]) / len(rdm.loc[rdm.rand == 'down'])
        RDMcriterion = criterion(hitRate, falseAlarmRate)
        emoLowRTs = len(emotion.loc[emotion.rt < fastRT])
        rdmLowRTs = len(rdm.loc[rdm.rt < fastRT])
        rdmDPrime = dPrime(len(rdm.loc[(rdm.rand == 'up')&(rdm.response == 'up')])/len(rdm.loc[(rdm.rand == 'up')]),
                            len(rdm.loc[(rdm.rand == 'down')&(rdm.response == 'up')])/len(rdm.loc[(rdm.rand == 'down')]))
        rdmAccuracy = len(rdm.loc[rdm.response == rdm.rand]) / len(rdm)
        up_prop = len(emotion.loc[emotion.response == 'up']) / len(emotion)
        
        subjectRow = pd.Series([sID, 
                                emoMissed, 
                                rdmMissed, 
                                RDMcriterion, 
                                emoLowRTs, 
                                rdmLowRTs, 
                                rdmDPrime, 
                                rdmAccuracy,  
                              
                                up_prop,
                  
                                ], 
                                index=df.columns)

        df = df.append(subjectRow, ignore_index=True)
    
    df['absC'] = np.abs(df.RDMcriterion)
    return df

# setting exclusions in original dataframe

def getBadSubjects(metaDF, item, threshold, relative):

    if relative == '<':
        return list(metaDF.loc[metaDF[item] < threshold].subject)
    elif relative == '>':
        return list(metaDF.loc[metaDF[item] > threshold].subject)
    
def setBadTrials(trialsDF, item, threshold, relative, exclusionColName):
    if relative == '<':
        trialsDF[exclusionColName] = trialsDF[item] < threshold
    elif relative == '>':
        trialsDF[exclusionColName] = trialsDF[item] > threshold
    return trialsDF

def setExclusion(trialsDF, badSubjects, exclusionColName):
    trialsDF[exclusionColName] = np.where(trialsDF['subject'].isin(badSubjects), True, False)
    return trialsDF

######## plotting class #############
class plot():

    def missedResponses(self, subjectMetaInfo):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        
        fig.suptitle('Missed responses per condition per subject')

        axs[0].set_title('Emotion')
        axs[0].hist(subjectMetaInfo.missedEmo )
        axs[0].set_xlabel('missed responses')
        axs[0].set_ylabel('subject #')
        
        axs[1].set_title('RDM')
        axs[1].hist(subjectMetaInfo.missedRDM )
        axs[1].set_xlabel('missed responses')

    def criterion(self, subjectMetaInfo):
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        
        fig.suptitle('Criterion')

        axs.set_title('RDM')
        axs.hist(subjectMetaInfo.absC)
        axs.set_xlabel('abs criterion')
        axs.set_ylabel('subject #')

    def lowRTs(self, subjectMetaInfo):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        
        fig.suptitle('Fast responses per subject')

        axs[0].set_title('Emotion')
        axs[0].scatter(range(len(subjectMetaInfo.emoLowRTs)), subjectMetaInfo.emoLowRTs )
        axs[0].set_ylabel('fast responses')
        axs[0].set_xlabel('subject #')
        
        axs[1].set_title('RDM')
        axs[1].scatter(range(len(subjectMetaInfo.rdmLowRTs)), subjectMetaInfo.rdmLowRTs )
        axs[1].set_xlabel('subject #')
    
    def dprime(self, subjectMetaInfo):
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        
        fig.suptitle('dPrime')

        axs.set_title('RDM')
        axs.scatter(range(len(subjectMetaInfo.rdmDPrime)), subjectMetaInfo.rdmDPrime)
        axs.set_ylabel('dPrime')
        axs.set_xlabel('subject #')
    
    def dPrimeAccuracy(self, subjectMetaInfo):
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        
        fig.suptitle('Accuracy and dPrime')

        axs.set_title('RDM')
        axs.scatter(subjectMetaInfo.rdmAccuracy, subjectMetaInfo.rdmDPrime)
        axs.set_ylabel('dPrime')
        axs.set_xlabel('subject #')

    def binSDE(self, nPlus,n):
        return sqrt(((nPlus/n)*(1-(nPlus/n)))/n)
    
    def staircases(self, subjectTrials, subject):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.tight_layout(h_pad=2)
        fig.suptitle('Subject #'+str(int(subject)))
        fig.subplots_adjust(top=0.85)
        
        sTrials = subjectTrials.loc[subjectTrials["trialType"]=="psi_staircase" ]
        eTrials = subjectTrials.loc[subjectTrials["trialType"]=="experiment" ]
        
        for stimType in subjectTrials.stimuli.unique():
            staircaseTrials = sTrials.loc[sTrials['stimuli']==stimType]
            expTrials = eTrials.loc[eTrials['stimuli']==stimType]
            lineCol=["b","y","c","g"]
            s=0
            for stairType in staircaseTrials.trialType.unique():
                currentStaircase = staircaseTrials.loc[staircaseTrials["trialType"]==stairType]
                currentStaircase['trialNumber']=range(currentStaircase.shape[0])
                
                if stimType=="emotion":
                    #ax1.ylim(0,200)
                    ax3.plot(currentStaircase.trialNumber,currentStaircase.estThres,color=lineCol[s],zorder=1,alpha=0.7)
                    ax3.plot(currentStaircase.trialNumber,currentStaircase.estSlope,color=lineCol[s+1],zorder=1,alpha=0.7)
                    ax3.plot(currentStaircase.trialNumber,currentStaircase.estIntens,color=lineCol[s+2],zorder=1,alpha=0.7)

                    happy=currentStaircase[currentStaircase['response']=='up']
                    angry=currentStaircase[currentStaircase['response']=='down']
                    ax3.scatter(happy.trialNumber,happy.stimLvl,zorder=2,alpha=0.7,marker="^",c='g')
                    ax3.scatter(angry.trialNumber,angry.stimLvl,zorder=2,alpha=0.7,marker="v",c='r')
                    ax3.set_title("Emotion Psi Staircases")
                    #ax1.ylabel("emotion")
                    
                    #eTrials['happy']=[response == 'up' for response in eTrials['response']]
                    lvls=[]
                    happyRate=[]
                    error =[]
                    for lvl in expTrials.stimLvl.unique():
                        lvlDat=eTrials[eTrials['stimLvl']==lvl]
                        lvls.append(lvl)
                        error.append(self.binSDE(sum(lvlDat.happy),len(lvlDat.happy)))
                        happyRate.append(sum(lvlDat.happy)/len(lvlDat.response))
                    #ax4.plot(lvls,happyRate)    
                    ax4.scatter(lvls,happyRate)
                    ax4.errorbar(lvls,happyRate,yerr=error, linestyle="None",color='r')
                    ax4.set_title('Happy rate per emotion lvl')
                        
                        
                else:
                    #ax3.ylim(0.0,1)
                    ax1.plot(currentStaircase.trialNumber,currentStaircase.estThres,color=lineCol[s],zorder=1,alpha=0.7)
                    ax1.plot(currentStaircase.trialNumber,currentStaircase.estSlope,color=lineCol[s+1],zorder=1,alpha=0.7)
                    ax1.plot(currentStaircase.trialNumber,currentStaircase.estIntens,color=lineCol[s+2],zorder=1,alpha=0.7)
                    hits=currentStaircase[currentStaircase["hit"]]
                    misses=currentStaircase[[not elm for elm in currentStaircase["hit"]]]  
                    ax1.scatter(misses.trialNumber,misses.stimLvl,color='r',zorder=2,alpha=0.7,marker=".")
                    ax1.scatter(hits.trialNumber,hits.stimLvl,color='g',zorder=2,alpha=0.7,marker=".")
                    ax1.set_title("Dot Motion Psi Staircases")
                    #ax3.ylabel("coherence")
                    
                    blocks=[]
                    correctRate=[]
                    error =[]
                    for block in expTrials.block.unique():
                        blockDat=expTrials[expTrials['block']==block]
                        blocks.append(block)
                        error.append(self.binSDE(sum(blockDat.hit),len(blockDat.hit)))
                        correctRate.append(sum(blockDat.hit)/len(blockDat.hit))
                    #ax4.plot(lvls,happyRate)    
                    ax2.bar(blocks,correctRate)
                    ax2.axhline(y=sum(expTrials.hit)/len(expTrials.hit),linestyle='dashed')
                    ax2.errorbar(blocks,correctRate,yerr=error, linestyle="None",color='r')
                    ax2.set_title('Correct rate RDM per block')
                

                #plt.xlabel("trial N")
                s+=1
        ax1.set_ylim([0,0.5])
        ax2.set_ylim([0.5,1])
        ax3.set_ylim([0,200])
        ax4.set_ylim([0,1])       
        plt.show()  

    def PSI_psychometric(self, x, alpha, beta, delta, min):
        _normCdf = norm.cdf(x, alpha, beta)
        if min==0:
            return .5 * delta + (1 - delta) * _normCdf
        elif min ==0.5:
            return .5 * delta + (1 - delta) * (.5 + .5 * _normCdf)
    
    def staircase_combined(self, data, task, save=None):
        fig, axes = plt.subplots(1,2, figsize=(20, 10), sharey=True, sharex=False, gridspec_kw={ 'width_ratios':[1.5,1]})
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        sns.set(style="white")

        if task == 'emotion':
            stimRange = (0,200) 
            delta = 0.02
            min_response = 0
        elif task == 'cRDM':
            stimRange = (0,1)
            delta = 0.05
            min_response = 0.5

        #fig.tight_layout(h_pad=2)
        #fig.suptitle('Subject #'+str(int(subject)))
        #fig.subplots_adjust(top=0.85)
        meanThresEst = None
        meanThres = None
        meanSlope = None
        for sub in data.subject.unique():
            currentStaircase = data.loc[(data["trialType"]=="psi_staircase") & (data.subject==sub) & (data.stimuli==task)]              
            currentStaircase['trialNumber']=range(currentStaircase.shape[0])
            

            axes[0].plot(currentStaircase.trialNumber[:49]+1, currentStaircase.estThres[:49], c='k', zorder=1, alpha=0.1)
            x =np.linspace(stimRange[0],stimRange[1],100)
            alpha = list(currentStaircase.estThres)[-1]
            beta = list(currentStaircase.estSlope)[-1]

            psychometric = self.PSI_psychometric(x=x, alpha=alpha, beta=beta, delta=delta, min=min_response)

            axes[1].plot(psychometric, x, c='k', alpha=0.1)
            

            if meanThresEst is None:
                meanThresEst = np.array(currentStaircase.estThres[:49])
                meanThres = [list(currentStaircase.estThres)[-1]]
                meanSlope = [list(currentStaircase.estSlope)[-1]]
            else:
                meanThresEst = np.vstack([meanThresEst, np.array(currentStaircase.estThres[:49])])
                meanThres.append(list(currentStaircase.estThres)[-1])
                meanSlope.append(list(currentStaircase.estSlope)[-1])


        meanThresEst = np.mean(meanThresEst, axis=0)
        meanThres= np.mean(meanThres)
        meanSlope = np.mean(meanSlope)
        axes[0].plot(currentStaircase.trialNumber[:49]+1, meanThresEst, c='lightcoral', zorder=10, alpha=1, linewidth=5)
        axes[1].plot(self.PSI_psychometric(x, meanThres, meanSlope, delta, min_response),
                        np.linspace(stimRange[0],stimRange[1],100), c='lightcoral', alpha=1, linewidth=5)
        
        if task == 'emotion':
            axes[1].set_xlabel('P(response = happy)', fontsize=20)
        if task == 'cRDM':
            axes[1].set_xlabel('P(response = correct)', fontsize=20)

        axes[0].set_ylabel('Stimulus level', fontsize=20)
        axes[0].set_xlabel('Trial number', fontsize=20)

        axes[0].set_ylim([stimRange[0],stimRange[1]])
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        if save is not None:
            fig.savefig(save)

    def thresholds_combined(self, data, tasks, show_mean=True, individual_thresholds= True, color_curves=True, fontsize=20, nSubjects=None, save=None):
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
            elif tasks[task] == 'cRDM':
                stimRange = (0,.8)
                delta = 0.05
                min_response = 0.5
                thresholds = [0.75]



            meanThresEst = None
            meanThres = None
            meanSlope = None
            n = len(data.subject.unique()) if nSubjects is None else nSubjects

            color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
            for sub in data.subject.unique()[:n]:
                currentStaircase = data.loc[(data["trialType"]=="psi_staircase") & (data.subject==sub) & (data.stimuli==tasks[task])]              
                currentStaircase['trialNumber']=range(currentStaircase.shape[0])
                c = next(color)

                #axes[0].plot(currentStaircase.trialNumber[:49]+1, currentStaircase.estThres[:49], c='k', zorder=1, alpha=0.1)
                stim_lvl =np.linspace(stimRange[0],stimRange[1],1000)
                alpha = list(currentStaircase.estThres)[-1]
                beta = list(currentStaircase.estSlope)[-1]

                psychometric = self.PSI_psychometric(x=stim_lvl, alpha=alpha, beta=beta, delta=delta, min=min_response)
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


            meanThresEst = np.mean(meanThresEst, axis=0)
            meanThres= np.mean(meanThres)
            meanSlope = np.mean(meanSlope)
            
            if show_mean:
                mean_psychometric = self.PSI_psychometric(stim_lvl, meanThres, meanSlope, delta, min_response)
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
                axes[task].set_xlabel('Face Happiness', fontsize=fontsize)
                axes[task].set_title('FAD', fontsize=fontsize)

            if tasks[task] == 'cRDM':
                axes[task].set_ylabel('P(response = correct)', fontsize=fontsize)
                axes[task].set_xlabel('Proportion coherent motion', fontsize=fontsize)
                axes[task].set_title('RDM', fontsize=fontsize)
                axes[task].set_yticks([0.5, 0.75, 1.0])
                
            axes[task].yaxis.set_major_formatter('{x:0<4.1f}')
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



