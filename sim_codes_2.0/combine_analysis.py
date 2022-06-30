import numpy as np
import h5py 

data=h5py.File('replica_1/traj_chr10_0_analyze.h5', 'r')
count=1
rdp={}
msd={}
print(data['MSD_A1'].shape, data.keys())
#for key in data.keys():
    
for replica in range(3,4):
    try:
        data_new = h5py.File('replica_{}/traj_chr10_0_analyze.h5'.format(replica),'r')
        #print(data_new.keys())
        #for key in data_new.keys():
        #    print(key,type(data[key]),data[key].shape)
            #data[key] += data_new[key]
        #    count += 1

    except (OSError,): pass
