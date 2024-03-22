import matplotlib.pyplot as plt
import numpy as np
import os
import sys
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
sys.path.append('..')
import paths
import resp_cleaning as rc
from importFunctions import *

dataPath = paths.raw
cleanSubs = paths.cleanSubData
subList = findSubjectsIDs(dataPath)
manualDataRej = paths.badSegments
masterPath = paths.data_master

# load sub data
individualDFs = []
for file in os.listdir(cleanSubs):
    if not file.startswith('.'):
        subjectData = pd.read_csv(str(cleanSubs / file))
        individualDFs.append(subjectData) # add sub df to list
        print(file)
    
allTrials = pd.concat(individualDFs) # concat the list of dfs into one master df

# define hits and happy responses
allTrials["hit"]= allTrials["response"]==allTrials["rand"]
allTrials['happy']=[resp == 'up' for resp in allTrials['response']]

# save master df
allTrials.to_csv(str(masterPath))