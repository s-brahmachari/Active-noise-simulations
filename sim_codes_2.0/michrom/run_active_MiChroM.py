import ActiveMonomerModule as AMM
from OpenMiChroM.ChromDynamics import MiChroM
import sys
import h5py
import numpy as np

#Collapse structure in Michrom
sim = MiChroM(name='collapse',temperature=1.0, time_step=0.01)
sim.setup(platform="hip")
sim.saveFolder(sys.argv[1])

Chrom10 = sim.create_springSpiral(ChromSeq='inputs/chr10_beads.txt', isRing=False)
sim.loadStructure(Chrom10, center=True)
sim.saveStructure(mode = 'auto')

sim.addFENEBonds(kfb=30.0)
sim.addAngles(ka=2.0)
sim.addRepulsiveSoftCore(Ecut=4.0)

sim.addTypetoType(mu=3.22, rc = 1.78)
sim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)

sim.addFlatBottomHarmonic( kr=5e-3, n_rad=8.0)

for _ in range(1000):
    sim.runSimBlock(1000, increment=False)

#save collapsed structure
sim.saveStructure(filename='chr10_collapse',mode='ndb')


#Initialize the active simulations
#0
asim=AMM.ActiveMonomer(time_step=1e-3, collision_rate=0.1, temperature=120.0,
        name="test", active_corr_time=1.0, act_seq=np.zeros(2712),
        outpath=sys.argv[1], platform="hip")

#asim = MiChroM(name='run',temperature=1.0, time_step=0.01)
#asim.setup(platform="hip")
#asim.saveFolder(sys.argv[1])

collapse=asim.loadNDB(NDBfiles=[sys.argv[1]+'/collapse_0_block0.ndb'])

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

block = 40
n_blocks = 100000

asim.initStorage(filename="traj_chr10")
# positions=[]
for _ in range(n_blocks):
    asim.runSimBlock(block, increment=True)
    asim.saveStructure()
    # asim.state = sim.context.getState(getPositions=True,getEnergy=True, getForces=True, getVelocities=True)
    # positions.append(asim.state.getPositions(asNumpy=True))
asim.storage[0].close()
#with h5py.File(sys.argv[1]+'/chr10_positions.h5', 'w') as hf:
#    hf.create_dataset("positions",  data=positions)

