import glob
import numpy as np

maxevents = 400000
path = "/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_llr_2018_forTraining/"
folders = glob.glob(path + "*/")
for folder in folders:
    folder = folder.split("/")[-2]
    print(folder)
    arrays = []
    nevents = 0
    index = 0
    for filename in glob.glob(path+folder+"/*.npz"):
        with np.load(filename) as f:
            arrays.append(f["events"])
            nevents = nevents + len(f["events"])
            if nevents > maxevents:
                array = np.concatenate(arrays, axis = 0)
                np.savez(path + folder + "/" + f"output{index}.npz", events=array)
                arrays = []
                index += 1
                nevents = 0
                
    if(len(arrays) != 0):
        array = np.concatenate(arrays, axis = 0)
        np.savez(path + folder + "/" + f"output{index}.npz", events=array)