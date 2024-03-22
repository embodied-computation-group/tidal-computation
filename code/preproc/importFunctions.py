import os
import warnings
import pandas as pd


def biggestFile(path,contains):
    """
    Find the biggest file containing 'contains' in the file name in a directory

    Parameters
    ----------
    path : str
        path to the directory.
    contains : str
        part of file name to search for.

    Returns
    -------
    biggest : str
        file name of biggest file in dir containing 'contains'.

    """
    biggest = None
    
    for file in os.listdir(path):
        if contains in file:
            if biggest is None or os.path.getsize(path+'/'+file) > os.path.getsize(path+'/'+biggest):
                biggest = file
           
    return biggest

def findSubjectsIDs(dir):
    subjects = []
    for trials in os.listdir(dir):
            if trials.startswith("trialLogs"):
                subject = trials[25:29]
                if subject not in subjects:
                    subjects.append(subject)
    return subjects

def findSubjectFiles(directory, subDirs=False, id=None):
    fileNames = []
    if not subDirs:
        for trials in os.listdir(directory):
            if trials.startswith("trialLogs"):
                if trials[25:29] != id:
                    continue
                subjectID=trials[25:29] if id is None else id
                for ts in os.listdir(directory):
                    if ts.startswith("timeSeries"):
                        if ts[27:31]==subjectID:
                            if trials[-26:] == ts[-26:]:
                                fileNames.append((trials,ts))
                                if id is not None: 
                                    return fileNames[0][0], fileNames[0][1]
                            else:
                                warnings.warn(f'timestamps of files {trials} and {ts} do not match')
    elif subDirs and id is None:
        subjectFolders =[f for f in os.listdir(directory) if not f.startswith('.')]
        for subject in subjectFolders:
            trials = biggestFile(directory+'/'+subject, "trialLogs")
            if trials is not None:
                ts = biggestFile(directory+'/'+subject, 'Breath-Brain_plux')
                fileNames.append((subject+'/'+trials,subject+'/'+ts))
    else:
        Warning
        
    return(fileNames)


def importSubjectData(trialsPath,respPath):
    trials = pd.read_csv(trialsPath)
    resp = pd.read_csv(respPath)
    return trials, resp


def findStimIndexes(stimSeries):
    stimIndexList =[]
    previous = 0
    for i in range(len(stimSeries)):
        if stimSeries[i] !=0 and previous == 0:
            stimIndexList.append(i)
        previous = stimSeries[i]
    return stimIndexList
