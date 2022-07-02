from OpenMiChroM.ChromDynamics import MiChroM
import numpy as np
# import simtk.openmm as mm
import simtk.unit as unit
from openmmtools.constants import kB
# from openmmtools import respa, utils
# from openmmtools.integrators import PrettyPrintableIntegrator as PrettyPrintableIntegrator
from openmmtools.integrators import ThermostatedIntegrator
from ActivePolymer import CustomBrownianIntegrator
#import custom_integrator as ci

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
    """
    integrator=CustomBrownianIntegrator(temperature=self.temperature * unit.kelvin,
                                        timestep=self.timestep * unit.picoseconds,
                                        collision_rate=self.collisionRate / unit.picoseconds,
                                           noise_corr=self.activeCorrTime * unit.picoseconds)
    """
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

def addHarmonicBonds(self, kfb=30.0):
    
    R"""
    Adds FENE (Finite Extensible Nonlinear Elastic) bonds between neighbor loci :math:`i` and :math:`i+1` according to "Halverson, J.D., Lee, W.B., Grest, G.S., Grosberg, A.Y. and Kremer, K., 2011. Molecular dynamics simulation study of nonconcatenated ring polymers in a melt. I. Statics. The Journal of chemical physics, 134(20), p.204904".

    Args:

        kfb (float, required):
            Bond coefficient. (Default value = 30.0).
        """

    for start, end, isRing in self.chains:
        for j in range(start, end):
            addHarmonicBond_ij(self,j, j + 1, kfb=kfb)
            self.bondsForException.append((j, j + 1))

        if isRing:
            addHarmonicBond_ij(self, start, end, distance=1, kfb=kfb)
            self.bondsForException.append((start, end ))

    self.metadata["HarmonicBond"] = repr({"kfb": kfb})
    
def _initHarmonicBond(self, kfb=30):
    R"""
    Internal function that inits FENE bond force.
    """
    if "HarmonicBond" not in list(self.forceDict.keys()):
        force = ("0.5 * kfb * (r-r0)*(r-r0)")
        bondforceGr = self.mm.CustomBondForce(force)
        bondforceGr.addGlobalParameter("kfb", kfb)
        bondforceGr.addGlobalParameter("r0", 1.) 
            
        self.forceDict["HarmonicBond"] = bondforceGr
    
def addHarmonicBond_ij(self, i, j, distance=None, kfb=30):
    
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

    _initHarmonicBond(self, kfb=kfb)
    self.forceDict["HarmonicBond"].addBond(int(i), int(j), [])

def addSelfAvoidance(self, Ecut=4.0):
        
    R"""
        Adds a soft-core repulsive interaction that allows chain crossing, which represents the activity of topoisomerase II. Details can be found in the following publications: 
        
            - Oliveira Jr., A.B., Contessoto, V.G., Mello, M.F. and Onuchic, J.N., 2021. A scalable computational approach for simulating complexes of multiple chromosomes. Journal of Molecular Biology, 433(6), p.166700.
            - Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173.
            - Naumova, N., Imakaev, M., Fudenberg, G., Zhan, Y., Lajoie, B.R., Mirny, L.A. and Dekker, J., 2013. Organization of the mitotic chromosome. Science, 342(6161), pp.948-953.

        Args:

            Ecut (float, required):
                Energy cost for the chain passing in units of :math:`k_{b}T`. (Default value = 4.0).
          """
    
    Ecut = Ecut*self.Epsilon
    
    repul_energy = ("0.5 * Ecut * (1.0 + tanh(1.0 - (20.0 * (r - r0))))")
    
    self.forceDict["SelfAvoidance"] = self.mm.CustomNonbondedForce(repul_energy)
    repulforceGr = self.forceDict["SelfAvoidance"]
    repulforceGr.addGlobalParameter('Ecut', Ecut)
    repulforceGr.addGlobalParameter('r0', 0.9)
    repulforceGr.setCutoffDistance(3.0)

    for _ in range(self.N):
        repulforceGr.addParticle(())

def addSphericalConfinement(self, R0=None, vol_frac=None, method='FlatBottomHarmonic', kr=5.0):
    
    try:
        if R0 is None and vol_frac is None:
            print("Error! specify either R (radius of confinement) or vol_frac (confinement volume fraction)")
            raise ValueError
    
        elif R0 is None and vol_frac is not None:
            R0=(self.N / (8 * vol_frac))**(1 / 3)
            vol_frac = float(vol_frac)
    
        elif R0 is not None and vol_frac is None:
            R0=float(R0)
            vol_frac= self.N / (2 * R0)**3
        
        if method=='LennardJones':
            print("-------\nImplementing Lennard-Jones confinement:\n\Radius: {0:.2f} \nVolume fraction: {1:.3f}\n".format(R0,vol_frac))

            LJ_energy="(4 * GROSe * (GROSs/r)^12 + GROSe) * step(GROScut - r); r= R - sqrt(x^2 + y^2 + z^2) "
            LJ_conf_fg = self.mm.CustomExternalForce(LJ_energy)
            LJ_conf_fg.addGlobalParameter('R', R0)
            LJ_conf_fg.addGlobalParameter('GROSe', 1.0)
            LJ_conf_fg.addGlobalParameter('GROSs', 1.0)
            LJ_conf_fg.addGlobalParameter("GROScut", 2.**(1./6.))
        
            self.forceDict["RadialConfinement"] = LJ_conf_fg
        
        elif method=='FlatBottomHarmonic':
            print("-------\nImplementing Flat-Bottom Harmonic confinement:\nRadius: {0:.2f} \nVolume fraction: {1:.3f} \nStiffness: {2:.2f}\n".format(R0,vol_frac,kr))

            FlatBottomHarm_energy="step(r-r_res) * 0.5 * kr * (r-r_res)^2; r=sqrt(x*x+y*y+z*z)"
            FBH_conf_fg = self.mm.CustomExternalForce(FlatBottomHarm_energy)
            FBH_conf_fg.addGlobalParameter('r_res', R0)
            FBH_conf_fg.addGlobalParameter('kr', kr)
            
            self.forceDict["RadialConfinement"] = FBH_conf_fg

        else:
            print("ERROR!!!\n\
                method='{}' is not a valid input. \nChoose either 'FlatBottomHarmonic' or 'LennardJones'.".format(method))
            raise ValueError
            
        for i in range(self.N):
            self.forceDict["RadialConfinement"].addParticle(i, [])

        self.ConfinementRadius = R0
    
    except (ValueError):
        print("ERROR!!!\nNO confinement potential added!")
        pass

def addCylindricalConfinement(self, r_conf=5.0, z_conf=10.0, kr=30.0):
    cyl_conf_energy="step(r_xy-r_res) * 0.5 * kr * (r_xy-r_res)^2 + step(z^2-zconf^2) * 0.5 * kr * (z-zconf)^2; r_xy=sqrt(x*x+y*y)"
    
    cyl_conf_fg = self.mm.CustomExternalForce(cyl_conf_energy)
    cyl_conf_fg.addGlobalParameter('r_res', r_conf)
    cyl_conf_fg.addGlobalParameter('kr', kr)
    cyl_conf_fg.addGlobalParameter('zconf', z_conf)
    
    self.forceDict["CylindricalConfinement"]=cyl_conf_fg

    for i in range(self.N):
        self.forceDict["CylindricalConfinement"].addParticle(i, [])
