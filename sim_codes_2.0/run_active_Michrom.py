import ActiveMonomerModule as AMM
from OpenMiChroM.ChromDynamics import MiChroM
import sys
import h5py

#Collapse structure in Michrom
sim = MiChroM(name='collapse',temperature=1.0, time_step=0.01)
sim.setup(platform="opencl")
sim.saveFolder(sys.argv[1])

Chrom10 = sim.create_springSpiral(ChromSeq='inputs/chr10_beads.txt', isRing=False)
sim.loadStructure(Chrom10, center=True)
sim.saveStructure(mode = 'auto')

sim.addFENEBonds(kfb=30.0)
sim.addAngles(ka=2.0)
sim.addRepulsiveSoftCore(Ecut=4.0)

sim.addTypetoType(mu=3.22, rc = 1.78)
sim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)

sim.addFlatBottomHarmonic( kr=5e-3, n_rad=10.0)

for _ in range(1000):
    sim.runSimBlock(1000, increment=False)

#save collapsed structure
sim.saveStructure(filename='chr10_collapse',mode='ndb')

#Initialize the active simulations
asim=AMM.ActiveMonomer(time_step=1e-5, collision_rate=0.1, temperature=120.0,
        name="test", active_corr_time=1.0, act_seq='inputs/chr10_rnaseq.txt',
        outpath=sys.argv[1], platform="opencl")

collapse=sim.loadPDB(PDBfiles=[sys.argv[1]+'/collapse_0_block0.pdb'])

asim.loadStructure(collapse, center=True)

asim.addFENEBonds(kfb=30.0)
asim.addAngles(ka=2.0)
asim.addRepulsiveSoftCore(Ecut=4.0)

asim.addTypetoType(mu=3.22, rc = 1.78)
asim.addIdealChromosome(mu=3.22, rc = 1.78, dinit=3, dend=500)

block = 500
n_blocks = 50000

positions=[]
for _ in range(n_blocks):
    asim.runSimBlock(block, increment=True)
    # asim.state = sim.context.getState(getPositions=True,getEnergy=True, getForces=True, getVelocities=True)
    positions.append(sim.state.getPositions(asNumpy=True))

with h5py.File(sys.argv[1]+'/chr10_positions.h5', 'w') as hf:
    hf.create_dataset("positions",  data=positions)

