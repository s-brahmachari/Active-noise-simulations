import numpy as np

import simtk.unit as unit
import simtk.openmm as mm
from sys import stdout, argv
import time

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

def PersistentBrownianIntegratorfunc(
                timestep=0.001, 
                temperature=298.0 * unit.kelvin,
                collision_rate=0.1,
                persistent_time=10.0,
                constraint_tolerance=1e-8,):
    
    #initialize integrator
    integrator = mm.CustomIntegrator(timestep)

    #parameters
    kT = kB * temperature
    
    #add globall variables
    integrator.addGlobalVariable("kT", kT)
    integrator.addGlobalVariable("g", collision_rate)
    integrator.addGlobalVariable("Ta", persistent_time)
    integrator.setConstraintTolerance(constraint_tolerance)

    # integrator.addGlobalVariable("a", np.exp(- timestep / Ta))    
    # integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * timestep / Ta)))

    integrator.addPerDofVariable("x1", 0) # for constraints

    integrator.addUpdateContextState()
    #update velocities. note velocities are active and not derived from positions.
    integrator.addComputePerDof("v", "(exp(- dt / Ta ) * v) + ((sqrt(1 - exp( - 2 * dt / Ta)) * f0 / g) * gaussian)")
    integrator.addConstrainVelocities()

    integrator.addComputePerDof("x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kT / g) * dt) * gaussian)")
    
    #remove the contribution from force group 0: the persistent force, which is already taken into account in the v*dt term
    integrator.addComputePerDof("x", "x - (dt  * f0 / g)")

    integrator.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
    integrator.addConstrainPositions()  # x is now constrained
    
    return integrator

class PersistentBrownianIntegrator():
    def __init__(self,
                timestep=0.001, 
                temperature=298.0 * unit.kelvin,
                collision_rate=0.1,
                persistent_time=10.0,
                constraint_tolerance=1e-8,):
    
        #initialize integrator
        integrator = mm.CustomIntegrator(timestep)

        #parameters
        kT = kB * temperature
        
        #add globall variables
        integrator.addGlobalVariable("kT", kT)
        integrator.addGlobalVariable("g", collision_rate)
        integrator.addGlobalVariable("Ta", persistent_time)
        integrator.setConstraintTolerance(constraint_tolerance)

        # integrator.addGlobalVariable("a", np.exp(- timestep / Ta))    
        # integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * timestep / Ta)))

        integrator.addPerDofVariable("x1", 0) # for constraints

        integrator.addUpdateContextState()
        #update velocities. note velocities are active and not derived from positions.
        integrator.addComputePerDof("v", "(exp(- dt / Ta ) * v) + ((sqrt(1 - exp( - 2 * dt / Ta)) * f0 / g) * gaussian)")
        integrator.addConstrainVelocities()

        integrator.addComputePerDof("x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kT / g) * dt) * gaussian)")
        
        #remove the contribution from force group 0: the persistent force, which is already taken into account in the v*dt term
        integrator.addComputePerDof("x", "x - (dt  * f0 / g)")

        integrator.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        integrator.addConstrainPositions()  # x is now constrained
        
    


def AddActivity(self, act_seq=None):

    #Initialize active force
    act_force=self.mm.CustomExternalForce(" - f_act * (x + y + z)")
    act_force.addPerParticleParameter('f_act')

    self.forceDict["ActiveForce"]=act_force

    #Add particles to active force group
    try:
        if type(act_seq)==np.ndarray:

            if self.N!=act_seq.shape[0]: 
                print('Number of particles in active sequence file and seq file do not match!')
                raise ValueError

            for bead_id, Fval in enumerate(act_seq):
                self.forceDict["ActiveForce"].addParticle(int(bead_id),[Fval])

        else:
            seq=np.loadtxt(act_seq)
            if self.N!=seq.shape[0]: 
                print('Number of particles in active sequence file and seq file do not match!')
                raise ValueError
                
            for bead_id, Fval in enumerate(act_seq):
                self.forceDict["ActiveForce"].addParticle(int(bead_id),[Fval])

        print('Active Force added to {} particles'.format(self.forceDict['ActiveForce'].getNumParticles()))
    except (ValueError,):
        print('Critical Error! Active force not added.')

def runSimBlock(self, steps=None, increment=True, num=None, check_energy=False):
        R"""
        Performs a block of simulation steps.
        
        Args:

            steps (int, required):
                 Number of steps to perform in the block.
            increment (bool, optional):
                 Whether to increment the steps counter. Typically it is set :code:`False` during the collapse or equilibration simulations. (Default value: :code:`True`).
            num (int or None, required):
                 The number of subblocks to split the steps of the primary block. (Default value: :code:`None`).                
        """
        if increment == True:
            self.step += 1
        if steps is None:
            steps = self.steps_per_block
        if (increment == True) and ((self.step % 100) == 0):
            self.printStats()
        
        if num is None:
            num = steps // 5 + 1

        for _ in range(steps // num):
            
            self.integrator.step(num)  # integrate!
            # stdout.flush()
        if (steps % num) > 0:
            self.integrator.step(steps % num)


def runSims(self, nblocks=100, blocksize=1000, check_every=100, save_format='.npy'):
    
    if self.forcesApplied == False:
        if self.verbose:
            print("applying forces")
            stdout.flush()
        self._applyForces()
        self.forcesApplied = True

    positions=[]
    check=False
    pos_check_point=self.data

    for ii, _ in enumerate(range(nblocks)):
        num = blocksize // 5 + 1
        for attempt in range(6):
            num+=attempt
            a = time.time()
            runSimBlock(self, steps=blocksize, increment=True, num=num)
            b = time.time()
            
            if ii%check_every==0 or check==True: #check energy component
                self.state = self.context.getState(getPositions=True,
                                                getEnergy=True)

                coords = self.state.getPositions(asNumpy=True)
                newcoords = coords / self.nm

                eK = (self.state.getKineticEnergy() / self.N / unit.kilojoule_per_mole)
                eP = self.state.getPotentialEnergy() / self.N / unit.kilojoule_per_mole
                
                print("bl=%d" % (self.step), end=' ')
                print("pos[1]=[%.1lf %.1lf %.1lf]" % tuple(newcoords[0]), end=' ')
                stdout.flush()

                if ((np.isnan(newcoords).any()) or (eK > self.eKcritical) or
                    (np.isnan(eK)) or (np.isnan(eP))):

                    self.context.setPositions(pos_check_point)
                    print("eK={0}, eP={1}, trying one more time at step {2} ".format(eK, eP, self.step))
                    check=True
                else:
                    dif = np.sqrt(np.mean(np.sum((newcoords -
                        self.getPositions()) ** 2, axis=1)))
                    print("dr=%.2lf" % (dif,), end=' ')
                    #update self.data only when energy looks fine
                    self.data = coords
                    # print("t=%2.1lfps" % (self.state.getTime() / unit.second * 1e-12), end=' ')
                    print("kin=%.2lf pot=%.2lf" % (eK,
                        eP), "Rg=%.3lf" % self.chromRG(), end=' ')
                    print("SPS=%.0lf" % (blocksize/ (float(b - a))))
                    #update check_point only when energy components look fine
                    pos_check_point=self.data
                    check=False
                    break

            else:    
                self.state = self.context.getState(getPositions=True,
                                   getEnergy=False)

                coords = self.state.getPositions(asNumpy=True)
                newcoords = coords / self.nm

                # print("pos[1]=[%.1lf %.1lf %.1lf]" % tuple(newcoords[0]), end=' ')

                if (np.isnan(newcoords).any()):
                    self.context.setPositions(pos_check_point)
                    print("trying one more time at step {0} with reduced time steps per block {1} ".format(self.step, num))
                
                else:
                    dif = np.sqrt(np.mean(np.sum((newcoords -
                        self.getPositions()) ** 2, axis=1)))
                    # print("dr=%.2lf" % (dif,), end=' ')
                    self.data = coords
                    # print("t=%2.1lfps" % (self.state.getTime() / unit.second * 1e-12), end=' ')
                    # print("SPS=%.0lf" % (blocksize / (float(b - a))))
                    break
    
        positions.append(newcoords)
    
        # if ii%int(nblocks*0.1)==0:
            # self.printForces()
            # np.save(self.folder+'traj_'+self.name+'_positions.npy',np.array(positions))
            # np.save(self.folder+self.name+'_lastFrame.npy',np.array(positions)[-1,:,:])
            # print('saved')

    # np.save(self.folder+'traj_'+self.name+'_positions.npy',np.array(positions))
    # np.save(self.folder+self.name+'_lastFrame.npy',np.array(positions)[-1,:,:])


#=======================================#
#   Brownian integrator                 #
#=======================================#

import simtk.openmm as mm

from openmmtools.constants import kB
from openmmtools import respa, utils
from openmmtools.integrators import PrettyPrintableIntegrator as PrettyPrintableIntegrator
from openmmtools.integrators import ThermostatedIntegrator as ThermostatedIntegrator


class CustomBrownianIntegrator(ThermostatedIntegrator):
    def __init__(self,
                 timestep=0.001, 
                temperature=298.0 * unit.kelvin,
                collision_rate=0.1,
                persistent_time=10.0,
                constraint_tolerance=1e-8,
                 ):

        # Create a new CustomIntegrator
        super(CustomBrownianIntegrator, self).__init__(temperature, timestep)
        #parameters
        kbT = kB * temperature
        
        #add globall variables
        self.addGlobalVariable("kbT", kbT)
        self.addGlobalVariable("g", collision_rate)
        self.addGlobalVariable("Ta", persistent_time)
        self.setConstraintTolerance(constraint_tolerance)

        # integrator.addGlobalVariable("a", np.exp(- timestep / Ta))    
        # integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * timestep / Ta)))

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
        