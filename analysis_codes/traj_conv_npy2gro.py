import numpy as np
import sys

xyz=np.load(sys.argv[1])
print(xyz.shape)

if len(sys.argv)>3:
    seq=np.loadtxt(sys.argv[3], dtype=str)
else:
    seq=['A' for _ in range(int(sys.argv[2]))]

res=0
with open(sys.argv[1].replace('.npy','.gro'),'w') as f:
    for t,val in enumerate(xyz):
        f.write('Chromatin\t t={}\n{}\n'.format(t, sys.argv[2]))
        for bead,sq in zip(range(val.shape[0]),seq):
            if 'A' in sq[1]: res=str(bead+1)+'ASP'
            else: res=str(bead+1)+'ASP'
            f.write('{0:>8s}{1:>6s}{2: 5d}{3: 8.3f}{4: 8.3f}{5: 8.3f}{6: 8.3f}{7: 8.3f}{8: 8.3f}\n'.format(res,'CA',bead+1, val[bead][0],val[bead][1],val[bead][2],0,0,0))
        f.write('{0: 8.4f}{1: 8.4f}{2: 8.4f}\n'.format(500.0, 500., 500.))

