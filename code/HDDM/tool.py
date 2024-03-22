import hddm
import pandas as pd
import os
import kabuki
from kabuki.analyze import gelman_rubin as gr_fun
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
import numpy as np
import seaborn as sns


def loadModels(model_dict, model_loc, in_file = None, in_file2 = None):
    wd = os.getcwd()
    os.chdir(model_loc)
    if model_dict is None:
        model_dict = {}


    for file in os.listdir(model_loc):
        if not file.endswith('.db'):
            if not file.startswith('.'):
                if in_file is not None:
                    if in_file in file:
                        if in_file2 is not None:
                            if in_file2 in file:
                                model_dict[file] = hddm.load(model_loc+file)        
                        else:
                            model_dict[file] = hddm.load(model_loc+file)
                else:
                    model_dict[file] = hddm.load(model_loc+file)
    if len(model_dict)==0:
        print('No models in model_dict')
    os.chdir(wd)
    return model_dict


def bestModel(model_dict, within=False):
    df = pd.DataFrame()
    if within:
        for model in model_dict:
            state_dependend = []
            for knode in model_dict[model].knodes:
                if 'state' in str(knode):
                    if '_C' in str(knode):
                        if str(knode)[:str(knode).index('_C')] not in state_dependend:
                            state_dependend.append(str(knode)[:str(knode).index('_C')])
                    else:
                        state_dependend.append(str(knode)[:2])
            df=df.append({
                'file': model,
                'model': model_dict[model].model,
                'DIC': round(model_dict[model].dic,0),
                'state_dependend': state_dependend,
                'samples': round(model_dict[model].mc._iter, 0),
                'psamples': max([knode['n'] for knode in list(model_dict[model]._stats.values())]),
                'burn': int(model_dict[model].mc._burn),
                'thin': int(model_dict[model].mc._thin),
                'name': model
                }, 
                    ignore_index=True)
        print(df[['file','model','state_dependend', 'samples','psamples','burn','thin', 'DIC']].sort_values('DIC'))

    else:
        for model in model_dict:
            print(model)
            df=df.append({
                        'file' : model,
                        #'model': model_dict[model].model,
                        'DIC': round(model_dict[model].dic,0),
                        'params': model_dict[model].include,
                        'samples': round(model_dict[model].mc._iter, 0),
                        'post samples': list(model_dict[model]._stats.values())[1]['n'], 
                        'burn': int(model_dict[model].mc._burn),
                        'thin': int(model_dict[model].mc._thin),
                        'name': model
                        }, 
                        ignore_index=True)

        print(df[['file','model','params', 'samples','post samples','burn','thin', 'DIC']].sort_values('DIC'))
    
    if 'state_dependend' in df.columns:
        return model_dict[df.loc[df.DIC == min(df.DIC)].name.iloc[0]], df.loc[df.DIC == min(df.DIC)].state_dependend.iloc[0]
    else:
        return model_dict[df.loc[df.DIC == min(df.DIC)].name.iloc[0]]

def sort_models(models, within_specifications):
    sortedModels = {}
    for model in models:
        for p in within_specifications:
            if p in model:
                if f'{models[model].model}_{p}' in sortedModels:
                    sortedModels[f'{models[model].model}_{p}'].append(models[model])
                else:
                    sortedModels[f'{models[model].model}_{p}'] = [models[model]]
    return sortedModels


def sort_models2(models, within_specifications):
    sortedModels = {}
    for model in models:
        for p in within_specifications:
            if model.startswith(p):
                if f'{models[model].model}_{p}' in sortedModels:
                    sortedModels[f'{models[model].model}_{p}'].append(models[model])
                else:
                    sortedModels[f'{models[model].model}_{p}'] = [models[model]]
    return sortedModels

def gelman_rubin(sortedModels):
    grDict = {}
    for model in sortedModels:
        models = sortedModels[model]
        gr = gr_fun(models).values()
        max_gr = max(gr)
        min_gr = min(gr)
        grDict[model] = (min_gr, max_gr)
    for model in grDict:
        print('min max GR:',  grDict[model])

def combine_models(models):
    combined = None
    for model in models:
        combined= kabuki.utils.concat_models(models[model])
    return combined


def plot_model(modelParams={'v':1,'a':1,'t':0.3,'z':0.5,'sv':1},
               xRange=(0,2.5),
               yRange=None,
               rtDat=None,
               ppc_rts=None,
               showParams=False,
               vcol = None,
               save=None,
               response_names=None,
               nSamps = 5,
               alphas = None,
               kws=1,
               bins=50,
               sim_alpha =None,
               legends=None
               ):
    
    # if (ppc_rts is None) & (rtDat is None):
    #     HR =[1,3,1]
    #     x_bound=0.5
    #     y_bound = 1.9
    #     hight=10
    # else:
    HR=[1,1,1]
    x_bound=0.75
    y_bound = 1.5
    hight=15
    fig, ax = plt.subplots(3,1,figsize=(15, hight),
            gridspec_kw={ 'height_ratios':HR, 'hspace':0}, sharex=True)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_yticks([])
    ax[0].set_xticks([])

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].xaxis.set_ticks_position('none') 

    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].set_yticks([])
    ax[2].set_xticks([])

    ax[2].patch.set_alpha(0)
    
    ax[0].zorder = 0
    ax[1].zorder = 1
    ax[2].zorder = 0
    
    #ax[0].sharey(ax[2])
    linewidth=3
    linewidth_kde= 1#0.001
    zorder_model = 100
    zorder_params=0
    histAlpha = 0.5
    #ax.spines['bottom'].set_visible(False)
    ax[2].set_xticks([0,0.5,1,1.5,2])
    ax[2].tick_params(labelsize=30)
    ax[2].set_xlabel('Time (sec.)', size=30)
    
    mean_params = {}
    for param in modelParams.keys():
        if isinstance(modelParams[param], np.ndarray):
            mean_params[param] = np.mean(modelParams[param])
        elif type(modelParams[param]) == list:
            mean_params[param] = []
            for elm in modelParams[param]:
                if isinstance(elm, np.ndarray):
                    mean_params[param].append(np.mean(elm))
                else:
                    mean_params[param].append(elm)
        else:
            mean_params[param] = modelParams[param]
                    


    ax[1].set_xlim(xRange[0], xRange[1])
    ax[1].set_xlim(0, xRange[1])


    # if rtDat is None:
    #     ax[1].set_ylim(-mean_params['a']*0.55, mean_params['a']*0.55)
    if yRange is not None:
        ax[1].set_ylim(yRange[0], yRange[1])
    else:
        ax[1].set_ylim(-mean_params['a']/2, mean_params['a']/2)
    
    c='k'
    # draw model components
    # a
    if isinstance(modelParams['a'], np.ndarray):
        for this_a in np.random.choice(modelParams['a'], nSamps, replace=True):
            ax[1].axhline(-this_a/2, c=c, linewidth=linewidth, zorder=zorder_model, alpha=alphas['a'])
            ax[1].axhline(this_a/2, c=c, linewidth=linewidth, zorder=zorder_model, alpha=alphas['a'])
    else:
        ax[1].axhline(-modelParams['a']/2, c=c, linewidth=linewidth, zorder=zorder_model)
        ax[1].axhline(modelParams['a']/2, c=c, linewidth=linewidth, zorder=zorder_model)
    # t
    if isinstance(modelParams['t'], np.ndarray):
        for this_t in np.random.choice(modelParams['t'], nSamps, replace=True):
                ax[1].plot([this_t, this_t], [-mean_params['a']/2,mean_params['a']/2 ], c=c, linewidth=linewidth, zorder=zorder_model, alpha=alphas['t'])
    else:
        ax[1].plot([modelParams['t'],modelParams['t']], [-mean_params['a']/2,mean_params['a']/2 ], c=c, linewidth=linewidth, zorder=zorder_model)
    
    # v and z
    vs = modelParams['v'] if type(modelParams['v']) == list else [modelParams['v']]
    # vs = [v * 2 for v in vs] # ajust v slope ??
    vcol = vcol if vcol is not None else ['k']*len(vs)
    if len(vs)>1 and showParams:
        showParams=False
        warnings.WarningMessage('can only show partms if len(v) == 1')
        
    startingPoint = -mean_params['a']*0.5 + mean_params['z']*mean_params['a']
    for v, c in zip(vs, vcol):
        if isinstance(v, np.ndarray):
            for this_v in np.random.choice(v, nSamps, replace=True):
                bound = mean_params['a']/2 if this_v > 0 else -mean_params['a']/2 
                boundCross = (bound-startingPoint)/this_v/2
                ax[1].plot([mean_params['t'],boundCross+mean_params['t']], [startingPoint, bound], c=c, linewidth=linewidth, zorder=zorder_model-1, alpha=alphas['v'])
        else:
            bound = mean_params['a']/2 if v > 0 else -mean_params['a']/2 
            boundCross = (bound-startingPoint)/v/2
            ax[1].plot([mean_params['t'],boundCross+mean_params['t']], [startingPoint, bound], c=c, linewidth=linewidth, zorder=zorder_model-1)
    
    p_size = 100
            
    if response_names is not None:
        #ax[0].text(xRange[0]+(xRange[1]-xRange[0])*0.75, mean_params['a'], response_names[0], horizontalalignment='center', va='bottom', size=p_size/2)
        #ax[2].text(xRange[0]+(xRange[1]-xRange[0])*0.75, mean_params['a'], response_names[1], horizontalalignment='center',va='top', size=p_size/2)
        
        
        ax[1].text(xRange[0]+(xRange[1]-xRange[0])*x_bound, mean_params['a']/y_bound, response_names[0], horizontalalignment='center', va='bottom', size=p_size/2, zorder=10)
        ax[1].text(xRange[0]+(xRange[1]-xRange[0])*x_bound, -mean_params['a']/y_bound, response_names[1], horizontalalignment='center',va='top', size=p_size/2, zorder=10)
        print(mean_params['a'])
    if ppc_rts is not None:    
        for ppc in ppc_rts.keys():
            for stim, c in zip(ppc_rts[ppc].keys(), vcol):                

                # sns.histplot( ppc_rts[ppc][stim], 
                #     bins=bins, ax=ax[0],  kde=False, kde_kws={'bw_adjust':kws, 'alpha':0.1}, alpha=0.01, 
                #     linewidth=linewidth_kde, color= c, binrange=(-xRange[1], xRange[1]), element='step', fill=False,
                #     legend = False)
                # sns.histplot( -ppc_rts[ppc][stim], 
                #     bins=bins, ax=ax[2],  kde=False, kde_kws={'bw_adjust':kws, 'alpha':0.1}, alpha=1, 
                #     linewidth=linewidth_kde, color= c, binrange=(-xRange[1], xRange[1]), element='step', fill=False,
                #     legend = False)
                ax[0].hist( ppc_rts[ppc][stim], 
                    bins=bins,   alpha=sim_alpha, 
                    linewidth=linewidth_kde, color= c, range=(-xRange[1], xRange[1]), histtype='step', fill=False,
                    zorder=0
                    )
                ax[2].hist( -ppc_rts[ppc][stim], 
                bins=bins,   alpha=sim_alpha, 
                linewidth=linewidth_kde, color= c, range=(-xRange[1], xRange[1]), histtype='step', fill=False,
                zorder=0
                )
    if rtDat is not None:
        for rts, c in zip(rtDat, vcol):
            sns.histplot( [rts], 
                bins=bins, ax=ax[0],  kde=False, kde_kws={'bw_adjust':kws}, alpha=histAlpha, 
                linewidth=linewidth, palette= [c], binrange=(-xRange[1], xRange[1]), element='step', fill=False,
                legend = False, zorder=1)
            sns.histplot( [-rts], 
                bins=bins, ax=ax[2],  kde=False, kde_kws={'bw_adjust':kws}, alpha=histAlpha, 
                linewidth=linewidth, palette= [c], binrange=(-xRange[1], xRange[1]), element='step', fill=False,
                legend = False, zorder=1)


            
        ymin_0, ymax_0 = ax[0].get_ylim()
        ymin_2, ymax_2 = ax[2].get_ylim()
        print(ymax_2, ymax_0, ymin_2, ymin_0)
        ymax = ymax_0 if ymax_0 > ymax_2 else ymax_2
        
        ax[0].set_ylim(0, ymax)
        ax[2].set_ylim(0, ymax)
        ax[2].invert_yaxis()
        

    # show params
    if showParams:
        linestyle='dotted'
        ajust = 0.045
        p_c = 'r'
        p_size = 100
        v_size = 0.5
        p_alpha = 0.7
        
        # a
        ax[1].plot([2,2], [-mean_params['a']/2,mean_params['a']/2 ], c =p_c, linestyle=linestyle, linewidth=linewidth, alpha=p_alpha)
        ax[1].text(2+ajust,0,'a', size = p_size, c=p_c, alpha=p_alpha, va='center')
        
        # t
        ax[1].plot([0,mean_params['t']], [startingPoint,startingPoint], c =p_c, linestyle=linestyle, linewidth=linewidth, alpha=p_alpha)
        ax[1].text(mean_params['t']/2-2*ajust,startingPoint+ajust,'t', size = p_size, c=p_c, alpha=p_alpha, va='baseline')
        
        # z
        ax[1].plot([mean_params['t']-ajust,mean_params['t']-ajust], [-mean_params['a'],startingPoint ], c=p_c, linestyle=linestyle, linewidth=linewidth, alpha=p_alpha)
        ax[1].text(mean_params['t']+ajust*1, -mean_params['a']/4,'z', size = p_size, c=p_c, alpha=p_alpha, va='center')
        
        # v
        axx = mean_params['t']*1.5
        ay = startingPoint + vs[0]*mean_params['t']
        bx = axx + v_size
        by = ay
        cx = bx
        cy = ay + vs[0]*v_size*2
        


        ax[1].plot([axx,bx,cx], [ay,by,cy], c=p_c, linestyle=linestyle, linewidth=linewidth, alpha=p_alpha)
        ax[1].text(bx+ajust, by+0.5*cy-ajust*4,'v', size = p_size, c=p_c,alpha= p_alpha)
    
    # Creating legend with color box
    
    if legends is not None:  
        
        legend_1 = ax[1].legend()
        legend_1.remove()
        leg = []
        for l, c in zip(legends, vcol):
            leg.append(mpatches.Patch(color=c, label=l))
        ax[1].legend(handles=leg, loc='upper right', title='Stimuli',fontsize='xx-large', title_fontsize=40,
                     #handleheight=1, handlelength=1,
                     prop={'size': 30}, facecolor='white'
                     )  
        
        #ax[1].legend(legends, loc="upper left")

        # legend = ax[1].get_legend()
        # handles = legend.legendHandles
        # legend.remove()
        # print(legends)
        # ax[1].legend(handles, legends, title='Stiuli')
    
   

        
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        


def test_phase_values(data, locking):  
    if ('peak' in data[locking].unique()):
        raise ValueError(f'"peak" in {locking}')
    elif ('trough' in data[locking].unique()):
        raise ValueError(f'"trough" in {locking}')
    else:
        print(f'only {data[locking].unique()} in {locking}')
    
def test_manual_bads(data, locking):
    # set bad lable to test based on locking:
    if 'onset' in locking:
            manBad = 'manBadOnset'
    elif 'respons' in locking:
            manBad = 'manBadRespons'
    # test that manual bad trials are excluded
    if sum(data[manBad] != 0):
            raise ValueError('manullaly labled bad trials in data')
    else:
        print('no manually labled bad trials in data')
        
def simdata(model, stimCol, nSets):
    emp_data = model.data
    sim_data = hddm.utils.post_pred_gen(model, 'subj_idx', nSets, append_data=False)
    set_data = {}
    for set in range(nSets):
        set_data[str(set)] = {}
        
        for stim in np.sort(emp_data[stimCol].unique()): 
            set_data[str(set)][str(stim)] = []
            
            for sub in emp_data.subj_idx.unique():    
                emp_sub_data = emp_data.loc[emp_data.subj_idx==sub]
                stim_indecies = np.array(emp_sub_data[stimCol] == stim, dtype=bool)
                if len(stim_indecies) != len(np.array(sim_data.loc[f'wfpt.{sub}', set].rt)):
                    print('woops')
                    print(len(stim_indecies) , len(np.array(sim_data.loc[f'wfpt.{sub}', set].rt)))
                set_data[str(set)][str(stim)].append(np.array(sim_data.loc[f'wfpt.{sub}', set].rt[stim_indecies]))
            set_data[str(set)][str(stim)] = np.concatenate(set_data[str(set)][str(stim)])
    return set_data