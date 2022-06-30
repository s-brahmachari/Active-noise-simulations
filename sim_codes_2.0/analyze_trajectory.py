from OpenMiChroM.CndbTools import cndbTools
import numpy as np
import os
#from scipy.spatial import distance, Voronoi, ConvexHull
import h5py
import sys

cndbT=cndbTools()

traj_file=sys.argv[1]
out_file=traj_file.replace('.cndb','_analyze.h5')

print('Loading trajectory file')
traj=cndbT.load(traj_file)
xyz=cndbT.xyz(frames=[1,None,1])

print(traj.dictChromSeq.keys())

with h5py.File(out_file, 'w') as hf:
    print('Calculating Radius of gyration')
    rad_gyr=cndbT.compute_RG(xyz)
    hf.create_dataset('Rad Gyr',data=rad_gyr)

    print('Calculating Gyration Tensor')
    gyr_tensor=cndbT.compute_GyrTensorEigs(xyz)
    hf.create_dataset('Gyr Tensor Eigs',data=gyr_tensor)
    
    print('Calculating MSD')
    msd=cndbT.compute_MSD(xyz)
    hf.create_dataset('MSD_all',data=msd)    

    print('Calculating Radial Density profile')
    rdp=cndbT.compute_RadNumDens(xyz)
    hf.create_dataset('RadDens_all',data=rdp)

    for types in traj.dictChromSeq.keys():
        xyz_type=cndbT.xyz(frames=[1,None,1], beadSelection=traj.dictChromSeq[types])
        types=types.decode("utf-8")
        print('Calculating MSD for ', types)
        msd=cndbT.compute_MSD(xyz_type)
        hf.create_dataset('MSD_{}'.format(types), data=msd)  

        print('Calculating Radial Density profile for ', types)
        rdp=cndbT.compute_RadNumDens(xyz_type)
        hf.create_dataset('RadDens_{}'.format(types),data=rdp)

