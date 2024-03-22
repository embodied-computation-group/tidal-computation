from pathlib import Path
import platform
p = platform.system()

# Get repo path
top = Path(__file__).resolve().parent.parent

# Data paths
data = top/'data'
raw = data/'raw' if p == 'Linux' else None
cleanSubData = data / 'clean_subs'
data_master = data/'master'/'allSubs_bothTasks.csv'
rdm_data = data/'master'/'rdm.csv'
fad_data = data/'master'/'fad.csv'
rdm_onset_data = data/'master'/'rdm_onset.csv'
fad_onset_data = data/'master'/'fad_onset.csv'
rdm_respons_data = data/'master'/'rdm_respons.csv'
fad_respons_data = data/'master'/'fad_respons.csv'
demographics = data/'master'/'demographics.csv'


# respiration
respirationData = data/'respiration'
badSegments = respirationData/'badSegments'
respToInspect = respirationData/'toInspect'

# HDDM path
HDDM = top / 'code' / 'HDDM'

# models
model_path = data / 'models'
rdm_models = model_path / 'RDM'
fad_models = model_path / 'FAD'

# figures
figures = top / 'figures'
paper_figs = figures / 'paper_figs'
supplementary_figs = figures / 'supplementary'
qc_figs = figures / 'qc'
peak_trough_fig = qc_figs / 'peak_trough'
fad_stimuli = figures / 'fad_stimuli'
respons_image = figures /'pngs/response.jpg'
post_combined_fig = figures / 'posteriors.png'
ex_model_fig = paper_figs / 'ex_model.png'
rdm_model_fig = paper_figs / 'rdm_model.png'
fad_model_fig = paper_figs / 'fad_model.png'
rdm_model_fig_ppc = paper_figs / 'rdm_model_ppc.png'
fad_model_fig_ppc = paper_figs / 'fad_model_ppc.png'
model_fig = paper_figs / 'combined_modelling.png'
experiment_setup_fig = paper_figs / 'setup.png'
ppc = supplementary_figs / 'ppc.png'

print(top)

