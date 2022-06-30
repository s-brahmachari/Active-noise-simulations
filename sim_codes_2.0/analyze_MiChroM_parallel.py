import sys
import multiprocessing
import numpy as np
from scipy.spatial import distance
from OpenMiChroM.CndbTools import cndbTools
import time
cndbT=cndbTools()

#takes two cmd line arguments: traj file to compute hic from, output and num of processes
in_traj,num_proc = sys.argv[1], int(sys.argv[2])

def compute_HiC(self, mu=3.22, rc=1.2, avg_all=True):
    print('Computing probability of contact versus contour distance')
    if avg_all:
        def calc_prob(data, mu, rc):
            return 0.5 * (1.0 + np.tanh(mu * (rc - distance.cdist(data, data, 'euclidean'))))
        size=self.shape[1]
        Prob = np.zeros((size, size))

        for i in range(self.shape[0]):
            Prob += calc_prob(self[i], mu, rc)
            if i % 5000 == 0:
                print("Reading frame {:} of {:}".format(i, len(self)))

        Prob=Prob/(self.shape[0])

    return Prob

print('Loading trajectory ...')
traj = cndbT.load(in_traj)
xyz=cndbT.xyz(frames=[1,traj.Nframes,1],beadSelection='all', XYZ=[0,1,2])

print('Trajectory shape:', xyz.shape)


sub_frames=xyz.shape[0]//num_proc
inputs=[xyz[ii*sub_frames:(ii+1)*sub_frames,:,:] for ii in range(num_proc)]
#print([val.shape for val in inputs])
print("Dividing into {} processes".format(num_proc))
pool = multiprocessing.Pool()
pool = multiprocessing.Pool(processes=num_proc)

probs=pool.map(compute_HiC, inputs)

probs=np.array(probs).reshape(num_proc,xyz.shape[1],xyz.shape[1])

hic=np.mean(probs,axis=0)

print(probs.shape,hic.shape)

np.save(in_traj.replace('.cndb','')+'_HiC_rc1p2.npy', hic)
#np.savetxt(in_traj.replace('.cndb','')+'_HiC.txt.gz', hic)


