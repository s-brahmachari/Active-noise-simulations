import numpy as np
import matplotlib.pyplot as plt
import os
fpath='../../data/ROUSE_chain_confined_PD/'
for fname in os.listdir(fpath):
    if 'GyrEigs' in fname:
        eigs=np.load(fpath+fname)
        print(eigs.shape)

        flat_eigs=eigs.reshape(eigs.shape[0]*eigs.shape[1],eigs.shape[2])
        print(flat_eigs.shape, flat_eigs[0])
        rg=np.sqrt(np.sum(flat_eigs, axis=1))
        # rg=flat_eigs[2]-0.5*(flat_eigs[0]+flat_eigs[1])
        print(rg.shape)

        hist,bin_edges=np.histogram(rg, bins=np.arange(0,rg.max()+10,0.1), density=True)

        plt.plot(bin_edges[:-1], hist,'.-')
        plt.show()

        break
