from OpenMiChroM.CndbTools import cndbTools
import sys
import numpy as np


cndbT = cndbTools()
print('Loading trajectory ...')
traj = cndbT.load(sys.argv[1])

traj_xyz=cndbT.xyz(frames=[1,traj.Nframes,1000], beadSelection='all', XYZ=[0,1,2])

print('Trajectory size: ',traj_xyz.shape)

hic=cndbT.traj2HiC(traj_xyz,rc=2.0)

np.save(sys.argv[1].replace('.cndb','')+sys.argv[2]+'_HiC.npy', hic)
np.savetxt(sys.argv[1].replace('.cndb','')+sys.argv[2]+'_HiC.txt.gz', hic)

