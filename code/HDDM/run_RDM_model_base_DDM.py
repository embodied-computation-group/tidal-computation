
import hddm
import numpy as np
import sys
import os 
from pathlib import Path
import sys
print(os.getcwd())
sys.path.append('..')
import paths
import tool
from  itertools import combinations

print(str(sys.argv))
# Data fromm call
nmcmc = int(sys.argv[1])
burn = int(sys.argv[2])
thin = int(sys.argv[3])
chains = int(sys.argv[4])
chain = int(sys.argv[5])

# specify where the data is located
dataPath = str(paths.rdm_respons_data) 

# specify where to save models
model_loc = paths.rdm_models
os.chdir(model_loc) # ch dir to make sure model .db file are located correctly -> models can be loaded even if files are moved

# Load data 
data = hddm.load_csv(dataPath)
print(data.subject.unique())

# specify the trial locking
locking = 'respons_phase' # using respons locking
# test content in data for this locking
tool.test_phase_values(data, locking) # test that there is only 'expiration' and 'inspiration' trials
tool.test_manual_bads(data, locking) # test for manually labled bad trials

# Create binary state variabel from the respiratory phase variabel
data['state'] = (data[locking] == 'inspiration').astype(int) #Inspiration = 1 

#Drop columns not mentioned here 
data = data[['subject', 'state', 'rt', 'response', 'dir']] 

#Rename columns to fit expected names
data = data.rename({'subject': 'subj_idx', 'dir':'stim'}, axis=1)

# # Flip rt's of errors
data = hddm.utils.flip_errors(data)

# # set group
group = True if len(data.subj_idx.unique()) > 1 else False

# params to add to base ddm
params_to_add = ['z']

# Specify full model regression
params = ['v','a','t','z']
specifications = []
# add treatment string for each parameter
for p in params:
        if p == 'v':
                specifications.append(f'v ~ 0 + C(stim, Treatment(0)) + C(stim, Treatment(0)):C(state, Treatment(0))')
        else:
                specifications.append(f'{p} ~ C(state, Treatment(0))')
print(specifications)



model_name = f'15_RDM_base_{nmcmc}_{burn}_{thin}_{chains}_{chain}'
print(model_name)
if not os.path.exists(model_loc/model_name):

    hddm_model = hddm.HDDMRegressor(data, 
                        specifications,
                        informative = False,
                        include = params_to_add,
                        p_outlier = 0.05,
                        bias=True,
                        is_group_model = group
                        )
    #hddm_model.find_starting_values()                
    hddm_model.sample(nmcmc, burn=burn, thin=thin, dbname=f'{model_name}.db', db='pickle')
    hddm_model.save(model_name)
    print('')
    
else:
    print(f'{model_name} is already in {model_loc}')
    
print(model_name, 'finished')
#print('model rts, range:', np.min(hddm_model.data.rt), np.max(hddm_model.data.rt))


