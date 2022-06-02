from OpenMiChroM.ChromDynamics import MiChroM
import numpy as np
# import simtk.openmm as mm
import simtk.unit as unit
from openmmtools.constants import kB
# from openmmtools import respa, utils
# from openmmtools.integrators import PrettyPrintableIntegrator as PrettyPrintableIntegrator
from openmmtools.integrators import ThermostatedIntegrator

import custom_integrator as ci

class PersistentBrownianIntegrator(ThermostatedIntegrator):
    def __init__(self,
                 timestep=0.001, 
                temperature=120.0,
                collision_rate=0.1,
                persistent_time=10.0,
                constraint_tolerance=1e-8,
                 ):

        # Create a new CustomIntegrator
        super(PersistentBrownianIntegrator, self).__init__(temperature, timestep)
        #parameters
        kbT = kB * temperature
        
        #add globall variables
        self.addGlobalVariable("kbT", kbT)
        self.addGlobalVariable("g", collision_rate)
        self.addGlobalVariable("Ta", persistent_time)
        self.setConstraintTolerance(constraint_tolerance)

        self.addPerDofVariable("x1", 0) # for constraints

        self.addUpdateContextState()
        #update velocities. note velocities are active and not derived from positions.
        self.addComputePerDof("v", "(exp(- dt / Ta ) * v) + ((sqrt(1 - exp( - 2 * dt / Ta)) * f0 / g) * gaussian)")
        self.addConstrainVelocities()

        self.addComputePerDof("x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kbT / g) * dt) * gaussian)")
        
        #remove the contribution from force group 0: the persistent force, which is already taken into account in the v*dt term
        self.addComputePerDof("x", "x - (dt  * f0 / g)")

        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained



def ActiveMonomer(
        time_step=0.001, collision_rate=0.1, temperature=120.0,
        name="ActiveMonomer", active_corr_time=10.0, act_seq=None,
        outpath='output/',
        platform="opencl"):

    #Initialize Michrom class
    self=MiChroM(name=name, velocity_reinitialize=False,
                temperature=temperature,collision_rate=collision_rate,)
    
    self.timestep=time_step
    self.name=name
    self.collisionRate=collision_rate
    self.temperature=temperature
    self.activeCorrTime=active_corr_time

    #set up integrator
    integrator=PersistentBrownianIntegrator(
                timestep=self.timestep, 
                collision_rate=self.collisionRate,
                temperature=self.temperature,
                persistent_time=self.activeCorrTime,
                )
    
    # integrator=ci.CustomBrownianIntegrator(temperature=self.temperature*unit.kelvin,
    #                                         noise_corr=self.activeCorrTime*unit.picoseconds)

    self.setup(platform=platform,integrator=integrator,)  
    self.saveFolder(outpath)

    #define active force group
    act_force=self.mm.CustomExternalForce(" - f_act * (x + y + z)")
    act_force.addPerParticleParameter('f_act')
    self.forceDict["ActiveForce"]=act_force

    try:
        if type(act_seq) is not np.ndarray:
            act_seq=np.loadtxt(act_seq)
                
        for bead_id, Fval in enumerate(act_seq):
            self.forceDict["ActiveForce"].addParticle(int(bead_id),[Fval])

        print('\n\
        ==================================\n\
        ActiveMonomer now set up.\n\
        Active correlation time: {}\n\
        Total number of active particles: {}\n\
        ==================================\n'.format(active_corr_time, self.forceDict["ActiveForce"].getNumParticles()))
    
    except (ValueError,):
        print('Critical Error! Active force not added.')

    return self


def addNLBonds(self, kfb=30.0):
    
    R"""
    Adds FENE (Finite Extensible Nonlinear Elastic) bonds between neighbor loci :math:`i` and :math:`i+1` according to "Halverson, J.D., Lee, W.B., Grest, G.S., Grosberg, A.Y. and Kremer, K., 2011. Molecular dynamics simulation study of nonconcatenated ring polymers in a melt. I. Statics. The Journal of chemical physics, 134(20), p.204904".

    Args:

        kfb (float, required):
            Bond coefficient. (Default value = 30.0).
        """

    for start, end, isRing in self.chains:
        for j in range(start, end):
            addBond(self,j, j + 1, kfb=kfb)
            self.bondsForException.append((j, j + 1))

        if isRing:
            addBond(self, start, end, distance=1, kfb=kfb)
            self.bondsForException.append((start, end ))

    self.metadata["NLBond"] = repr({"kfb": kfb})
    
def _initNLBond(self, kfb=30):
    R"""
    Internal function that inits FENE bond force.
    """
    if "NLBond" not in list(self.forceDict.keys()):
        force = ("0.5 * kfb * (r-r0)*(r-r0)")
        bondforceGr = self.mm.CustomBondForce(force)
        bondforceGr.addGlobalParameter("kfb", kfb)
        bondforceGr.addGlobalParameter("r0", 1.) 
            
        self.forceDict["NLBond"] = bondforceGr
    
def addBond(self, i, j, distance=None, kfb=30):
    
    R"""
    Adds bonds between loci :math:`i` and :math:`j` 

    Args:

        kfb (float, required):
            Bond coefficient. (Default value = 30.0).
        i (int, required):
            Locus index **i**.
        j (int, required):
            Locus index **j**
        """

    if (i >= self.N) or (j >= self.N):
        raise ValueError("\n Cannot add a bond between beads  %d,%d that are beyond the chromosome length %d" % (i, j, self.N))
    if distance is None:
        distance = self.length_scale
    else:
        distance = self.length_scale * distance
    distance = float(distance)

    _initNLBond(self, kfb=kfb)
    self.forceDict["NLBond"].addBond(int(i), int(j), [])

