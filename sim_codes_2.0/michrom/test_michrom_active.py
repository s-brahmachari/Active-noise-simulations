import ActiveMonomerModule as AMM
from OpenMiChroM.ChromDynamics import MiChroM
import sys
import h5py
import numpy as np
from OpenMiChroM.CndbTools import cndbTools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

cndbT = cndbTools()
out='output/'
block = 100
n_blocks = 1000

"""
#Collapse structure in Michrom
sim = MiChroM(name='collapse',temperature=1.0, time_step=0.01)
sim.setup(platform="opencl")
sim.saveFolder(out)

Chrom10 = sim.create_springSpiral(ChromSeq='chr10_beads.txt', isRing=False)
sim.loadStructure(Chrom10, center=True)
sim.saveStructure(mode = 'auto')

sim.addFENEBonds(kfb=30.0)
sim.addAngles(ka=2.0)
sim.addRepulsiveSoftCore(Ecut=4.0)

sim.addTypetoType(mu=3.22, rc = 1.78)
sim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)

sim.addFlatBottomHarmonic( kr=5e-3, n_rad=8.0)

for _ in range(100):
    sim.runSimBlock(1000, increment=False)

#save collapsed structure
sim.saveStructure(filename='chr10_collapse',mode='ndb')
"""
"""
#MiChroM Run

msim = MiChroM(name='run',temperature=1.0, time_step=0.01)
msim.setup(platform="opencl")
msim.saveFolder(out)

collapse=msim.loadNDB(NDBfiles=[out+'collapse_0_block0.ndb'])

msim.loadStructure(collapse, center=True)
#1
AMM.addHarmonicBonds(msim,kfb=50)
#asim.addFENEBonds(kfb=30.0)

#2
msim.addAngles(ka=2.0)

#3
#asim.addRepulsiveSoftCore(Ecut=4.0)
AMM.addSelfAvoidance(msim,Ecut=4.0)

#4
msim.addTypetoType(mu=3.22, rc = 1.78)
#asim.addCustomTypes(mu=3.22, rc = 1.78, TypesTable='inputs/types_table_null.csv')
#5
msim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)
msim.addFlatBottomHarmonic( kr=5e-3, n_rad=8.0)



msim.initStorage(filename="traj_michrom_test")
# positions=[]
for _ in range(n_blocks):
    msim.runSimBlock(block, increment=True)
    msim.saveStructure()
    # asim.state = sim.context.getState(getPositions=True,getEnergy=True, getForces=True, getVelocities=True)
    # positions.append(asim.state.getPositions(asNumpy=True))
msim.storage[0].close()
"""
trajm = cndbT.load(out+'traj_michrom_test_0.cndb')
trajm_xyz=cndbT.xyz(frames=[1,trajm.Nframes,1], beadSelection='all', XYZ=[0,1,2])

print('Trajectory size: ',trajm_xyz.shape)

mhic=cndbT.traj2HiC(trajm_xyz)

np.save(out+'michrom_hic.npy', mhic)

fig,ax=plt.subplots(1,2, figsize=(13,6))
ax[0].imshow(mhic, norm=LogNorm(vmin=1e-4,vmax=1), cmap='Reds')
ax[0].set_title('michrom')

#ActiveMIchrom run

asim=AMM.ActiveMonomer(time_step=1e-3, collision_rate=1.0, temperature=120.0,
        name="test", active_corr_time=1.0, act_seq=np.zeros(2712),
        outpath=out, platform="opencl")

#asim = MiChroM(name='run',temperature=1.0, time_step=0.01)
#asim.setup(platform="hip")
#asim.saveFolder(sys.argv[1])

collapse=asim.loadNDB(NDBfiles=[out+'collapse_0_block0.ndb'])

asim.loadStructure(collapse, center=True)
#1
AMM.addHarmonicBonds(asim,kfb=50)
#asim.addFENEBonds(kfb=30.0)

#2
asim.addAngles(ka=2.0)

#3
#asim.addRepulsiveSoftCore(Ecut=4.0)
AMM.addSelfAvoidance(asim,Ecut=4.0)

#4
asim.addTypetoType(mu=3.22, rc = 1.78)
#asim.addCustomTypes(mu=3.22, rc = 1.78, TypesTable='inputs/types_table_null.csv')
#5
asim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)
asim.addFlatBottomHarmonic( kr=5e-3, n_rad=8.0)

asim.initStorage(filename="traj_active_test")
# positions=[]
for _ in range(n_blocks):
    asim.runSimBlock(block, increment=True)
    asim.saveStructure()
    # asim.state = sim.context.getState(getPositions=True,getEnergy=True, getForces=True, getVelocities=True)
    # positions.append(asim.state.getPositions(asNumpy=True))
asim.storage[0].close()

traja = cndbT.load(out+'traj_active_test_0.cndb')
traja_xyz=cndbT.xyz(frames=[1,traja.Nframes,1], beadSelection='all', XYZ=[0,1,2])

print('Trajectory size: ',traja_xyz.shape)

ahic=cndbT.traj2HiC(traja_xyz)

ax[1].imshow(ahic, norm=LogNorm(vmin=1e-4,vmax=1), cmap='Reds')
ax[1].set_title('Active')

fig.savefig(out+'compare_michrom_active.png', dpi=300, bbox_inches='tight')

np.save(out+'active_hic.npy', ahic)