import pyedflib as pdf
import numpy as np
import os
import glob

path = "/home/sneha/UMASS/UBICOMP/project/data/S"
target_path = "/home/sneha/UMASS/UBICOMP/project/data/raw_data/"


def read_file(f, task, subject):
    D = f.signals_in_file
    N = f.getNSamples()[0]
    ann =  f.readAnnotations()	
    annotations = [float(a.split("T")[1]) for a in ann[2]]
    annotations = np.array(annotations)
    time_window = float(f.getNSamples()[0] / len(ann[2]))
    #sigbufs = np.zeros((N, D))
    all_annotations =  np.repeat(annotations, time_window)
    N_ann = all_annotations.shape
    
    sigbufs = np.zeros((N_ann[0], D))
    n = f.signals_in_file
    for i in np.arange(n):
    	signal = f.readSignal(i)
    	sigbufs[:,i] = signal[:N_ann[0]]
    print sigbufs.shape, all_annotations.shape
    final_data = np.append(sigbufs,all_annotations[:,None],axis=-1)
    tasks = [task] * N_ann[0]
    subjects = [subject] * N_ann[0]
    final_data_ = np.append(final_data, np.array(tasks)[:,None], axis=1)
    final_data_ = np.append(final_data_, np.array(subjects)[:,None], axis=1)
    print final_data_.shape
    return final_data_


# for i in range(1,110):
# 	folder = path + str(i) +"/"
# 	for j in range(1,15):
# 		g = '{0:02}'.format(j)
# 		filename = folder + "S"+ g + ".edf"
# 		print filename
# 		f = pdf.EdfReader(filename)
# 		final_data = read_file(f, j, i)
# 		np.save(target_path + str(i) + "_" + str(j) +".npy", final_data)


dataArray = []
fpath ="/home/sneha/UMASS/UBICOMP/project/data/raw_data/"
npyfilespath = "/home/sneha/UMASS/UBICOMP/project/data/raw_data/*.npy"   
#os.chdir(npyfilespath)
#files = glob.glob(npyfilespath)
for i, f in enumerate(os.listdir(fpath)):
	
	print np.load(fpath+f).shape, i 
	#np.save
print np.array(dataArray).shape
	
# dataArray = []
# with open(fpath, 'wb') as f_handle:
#     for npfile in glob.glob("*.npy"):

#         # Find the path of the file
#         filepath = os.path.join(path, npfile)
#         print filepath
#         # Load file
#         dataArray.append(np.load(filepath))
#         # print dataArray
#         # np.save(f_handle,dataArray)
# # dataArray= np.load(fpath)
# print dataArray.shape


     