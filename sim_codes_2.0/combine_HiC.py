import numpy as np

hic=np.load('replica_1/traj_chr10_0_HiC_rc1p2.npy')
count=1
for rep in range(2,121):
    try:
        hic+=np.load('replica_{}/traj_chr10_0_HiC_rc1p2.npy'.format(rep,rep))
        count+=1
    except (FileNotFoundError,):
        pass

hic=hic/count
print(count)
np.save('chr10_all_HiC_rc1p2.npy', hic)
#np.savetxt('chr10_all_HiC.txt.gz', hic)
    
