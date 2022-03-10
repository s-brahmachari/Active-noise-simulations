
from OpenMiChroM.ChromDynamics import MiChroM
import numpy as np
import logging
import re
from sys import stdout, argv
import time 

import simtk.unit as unit
import simtk.openmm as mm

from openmmtools.constants import kB
from openmmtools import respa, utils
from openmmtools.integrators import PrettyPrintableIntegrator as PrettyPrintableIntegrator
from openmmtools.integrators import ThermostatedIntegrator as ThermostatedIntegrator

logger = logging.getLogger(__name__)

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole

#=====================================================

def ActivePolymer(
        time_step=0.001, collision_rate=0.1, temperature=120.0,
        name="ActivePolymer", active_corr_time=10.0, activity_amplitude=0.0,
        outpath='output/', init_struct=None, seq_file=None, active_particles=[],
        platform="opencl"):

    self=MiChroM(name=name, velocity_reinitialize=False, verbose=False,
                temperature=temperature,collision_rate=collision_rate)
    
    self.timestep=time_step
    self.name=name
    self.collisionRate=collision_rate
    self.temperature=temperature
    self.activeAmplitude=activity_amplitude
    self.activeCorrTime=active_corr_time

    integrator=CustomBrownianIntegrator(
                timestep=self.timestep * unit.picoseconds, 
                collision_rate=self.collisionRate / unit.picoseconds,
                temperature=self.temperature * unit.kelvin,measure_heat=False,
                noise_corr=self.activeCorrTime * unit.picoseconds,
                )

    self.setup(platform=platform,integrator=integrator,)  
    self.saveFolder(outpath)

    try:
        seq=np.loadtxt(seq_file, dtype=str)
        if init_struct is None:
            chrm=self.create_springSpiral(ChromSeq=seq_file)
            init_struct='default spiral spring'
        else:
            if '.npy' in init_struct:
                chrm=np.load(init_struct)
            else:
                chrm=np.loadtxt(init_struct) 
        if seq.shape[0]!=chrm.shape[0]:
            print('ERROR!!!\n\
                    Mismatched number of monomers in sequence file and initial structure\n')
            raise TypeError

        self.loadStructure(chrm,center=True)
        self._translate_type(seq_file)
    
    except (TypeError,):
        print('Please check:\n\
            1. sequence file is in .txt or .csv format\n\
            2. the number of monomers in the sequence file and initial structure file (if provided) are the same')
        pass

    act_force=self.mm.CustomExternalForce(" - f_act * (x + y + z)")
    act_force.addGlobalParameter('f_act',activity_amplitude)
    self.forceDict["ActiveForce"]=act_force
    for ii in active_particles:
        self.forceDict["ActiveForce"].addParticle(ii,[])

    print('\n\
        ==================================\n\
        ActivePolymer Simulation now set up.\n\
        Loaded sequence file: {}\n\
        Loaded initial structure: {}\n\
        Active amplitude: {}\n\
        Active correlation time: {}\n\
        Total number of active particles: {}\n\
        ==================================\n'.format(
        seq_file, init_struct,activity_amplitude, active_corr_time, len(active_particles)))
    
    return self


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
    
        if ii%int(nblocks*0.1)==0:
            # self.printForces()
            np.save(self.folder+'traj_'+self.name+'_positions.npy',np.array(positions))
            # print('saved')

    np.save(self.folder+'traj_'+self.name+'_positions.npy',np.array(positions))

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

        
    
def _addHarmonicBond_ij(self, i, j, kb=5.0,d=1.):

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

    if "HarmonicBond" not in list(self.forceDict.keys()):
        harmonic_energy = ("0.5 * k_har * (r - r0_har) * (r - r0_har)")
        harmonic_bond_fg = self.mm.CustomBondForce(harmonic_energy)
        harmonic_bond_fg.addGlobalParameter("k_har", kb)
        harmonic_bond_fg.addGlobalParameter("r0_har", d)

        self.forceDict["HarmonicBond"] = harmonic_bond_fg

    self.forceDict["HarmonicBond"].addBond(int(i), int(j), [])
    self.bondsForException.append((int(i), int(j)))


def addHarmonicBonds(self,top_file=None,kb=5.0,d=1.0):
    try:
        if top_file is None: 
            print('No topology file specified!')
            raise TypeError
        
        chrm_top=np.loadtxt(top_file, delimiter=' ', dtype=int)
        print("-------\nAdding harmonic bonds according to topology defined in {}\nNumber of polymer segments: {}\n".format(top_file, chrm_top.shape[0]))
        for row in chrm_top:
            for ii in range(row[0],row[1]):
                _addHarmonicBond_ij(self,ii,ii+1, kb=kb, d=d)
                if row[2]==1:
                    _addHarmonicBond_ij(self,row[0],row[1], kb=kb,d=d)

    except (TypeError,):
        print("ERROR!!! \nTopology file either missing or not in .txt or .csv format.")
        pass


def addRadialConfinement(self, R0=None, vol_frac=None, method='FlatBottomHarmonic', kr=5.0):
    
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
    

#=======================================#
#   Brownian integrator                 #
#=======================================#
class CustomBrownianIntegrator(ThermostatedIntegrator):
    """Integrates Brownian dynamics with a thermal and a an active noise.

    Brownian system is divided into three parts which can each be solved "exactly:"
        - R: stochastic update of *positions*, using current velocities and current temperature
            x <- x + v dt

        - V: Null step (nothing happens in this step)
            v <- v 

        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities associated with active correlated force (F)
            v <- av + (b/gamma) R
                where
                a = e^(-gamma dt)
                b = F * sqrt(1 - exp(- 2 * h / Ta))
                R is i.i.d. standard normal

    We then construct integrators by solving each part for a certain timestep in sequence.

    Attributes
    ----------
    _kinetic_energy : str
        This is 0.5*m*v*v by default, and is the expression used for the kinetic energy
    shadow_work : unit.Quantity with units of energy
       Shadow work (if integrator was constructed with measure_shadow_work=True)
    heat : unit.Quantity with units of energy
       Heat (if integrator was constructed with measure_heat=True)

    References
    ----------
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
    """

    _kinetic_energy = "0.5 * m * v * v"

    def __init__(self,
                 temperature=298.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=0.001 * unit.picoseconds,
                 noise_corr=10.0 * unit.picoseconds,
                 splitting="V O R",
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=False,
                 ):
        """Create a Langevin integrator with the prescribed operator splitting.

        Parameters
        ----------
        splitting : string, default: "V O R"


        temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
           Fictitious "bath" temperature

        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
           Collision rate

        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           Integration timestep
           
        noise_amp : amplitude of active noise
        
        noise_corr : correlation time of active force

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: False
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        """

        # Compute constants
        gamma = collision_rate
        self._gamma = gamma
        
        Ta=noise_corr
        self._Ta = Ta

        # Check if integrator is metropolized by checking for M step:
        if splitting.find("{") > -1:
            self._metropolized_integrator = True
            # We need to measure shadow work if Metropolization is used
            measure_shadow_work = True
        else:
            self._metropolized_integrator = False

        # Record whether we are measuring heat and shadow work
        self._measure_heat = measure_heat
        self._measure_shadow_work = measure_shadow_work

        ORV_counts, mts, force_group_nV = self._parse_splitting_string(splitting)

        # Record splitting.
        self._splitting = splitting
        self._ORV_counts = ORV_counts
        self._mts = mts
        self._force_group_nV = force_group_nV

        # Create a new CustomIntegrator
        super(CustomBrownianIntegrator, self).__init__(temperature, timestep)

        # Initialize
        self.addPerDofVariable("sigma", 0)

        # Velocity mixing parameter: current velocity component
        h = timestep / max(1, ORV_counts['O'])
        self.addGlobalVariable("a", np.exp(- h / Ta))
        
        self.addGlobalVariable("g", gamma)

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * h / Ta)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add global variables
        self._add_global_variables()

        # Add integrator steps
        self._add_integrator_steps()


    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = {
            'O': (self._add_O_step, False),
            'R': (self._add_R_step, False),
            '{': (self._add_metropolize_start, False),
            '}': (self._add_metropolize_finish, False),
            'V': (self._add_V_step, True)
        }
        return dispatch_table

    def _add_global_variables(self):
        """Add global bookkeeping variables."""
        if self._measure_heat:
            self.addGlobalVariable("heat", 0)

        if self._measure_shadow_work or self._measure_heat:
            self.addGlobalVariable("old_ke", 0)
            self.addGlobalVariable("new_ke", 0)

        if self._measure_shadow_work:
            self.addGlobalVariable("old_pe", 0)
            self.addGlobalVariable("new_pe", 0)
            self.addGlobalVariable("shadow_work", 0)

        # If we metropolize, we have to keep track of the before and after (x, v)
        if self._metropolized_integrator:
            self.addGlobalVariable("accept", 0)
            self.addGlobalVariable("ntrials", 0)
            self.addGlobalVariable("nreject", 0)
            self.addGlobalVariable("naccept", 0)
            self.addPerDofVariable("vold", 0)
            self.addPerDofVariable("xold", 0)



    def _get_energy_with_units(self, variable_name, dimensionless=False):
        """Retrive an energy/work quantity and return as unit-bearing or dimensionless quantity.

        Parameters
        ----------
        variable_name : str
           Name of the global context variable to retrieve
        dimensionless : bool, optional, default=False
           If specified, the energy/work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the work in kT (float).
           Otherwise, the unit-bearing work in units of energy.
        """
        work = self.getGlobalVariableByName(variable_name) * _OPENMM_ENERGY_UNIT
        if dimensionless:
            return work / self.kT
        else:
            return work


    def get_heat(self, dimensionless=False):
        """Get the current accumulated heat.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the heat in kT (float).
           Otherwise, the unit-bearing heat in units of energy.
        """
        if not self._measure_heat:
            raise Exception("This integrator must be constructed with 'measure_heat=True' in order to measure heat.")
        return self._get_energy_with_units("heat", dimensionless=dimensionless)


    @property
    def heat(self):
        return self.get_heat()

    def get_acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators.

        Returns
        -------
        acceptance_rate : float
           Acceptance rate.
           An Exception is thrown if the integrator is not Metropolized.
        """
        if not self._metropolized_integrator:
            raise Exception("This integrator must be Metropolized to return an acceptance rate.")
        return self.getGlobalVariableByName("naccept") / self.getGlobalVariableByName("ntrials")


    @property
    def acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators."""
        return self.get_acceptance_rate()

    @property
    def is_metropolized(self):
        """Return True if this integrator is Metropolized, False otherwise."""
        return self._metropolized_integrator

    def _add_integrator_steps(self):
        """Add the steps to the integrator--this can be overridden to place steps around the integration.
        """
        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/g)"})

        for i, step in enumerate(self._splitting.split()):
            self._substep_function(step)

    def _sanity_check(self, splitting):
        """Perform a basic sanity check on the splitting string to ensure that it makes sense.

        Parameters
        ----------
        splitting : str
            The string specifying the integrator splitting
        mts : bool
            Whether the integrator is a multiple timestep integrator
        allowed_characters : str, optional
            The characters allowed to be present in the splitting string.
            Default RVO and the digits 0-9.
        """

        # Space is just a delimiter--remove it
        splitting_no_space = splitting.replace(" ", "")

        allowed_characters = "0123456789"
        for key in self._step_dispatch_table:
            allowed_characters += key

        # sanity check to make sure only allowed combinations are present in string:
        for step in splitting.split():
            if step[0]=="V":
                if len(step) > 1:
                    try:
                        force_group_number = int(step[1:])
                        if force_group_number > 31:
                            raise ValueError("OpenMM only allows up to 32 force groups")
                    except ValueError:
                        raise ValueError("You must use an integer force group")
            elif step == "{":
                    if "}" not in splitting:
                        raise ValueError("Use of { must be followed by }")
                    if not self._verify_metropolization(splitting):
                        raise ValueError("Shadow work generating steps found outside the Metropolization block")
            elif step in allowed_characters:
                continue
            else:
                raise ValueError("Invalid step name '%s' used; valid step names are %s" % (step, allowed_characters))

        # Make sure we contain at least one of R, V, O steps
        assert ("R" in splitting_no_space)
        assert ("V" in splitting_no_space)
        assert ("O" in splitting_no_space)

    def _verify_metropolization(self, splitting):
        """Verify that the shadow-work generating steps are all inside the metropolis block

        Returns False if they are not.

        Parameters
        ----------
        splitting : str
            The langevin splitting string

        Returns
        -------
        valid_metropolis : bool
            Whether all shadow-work generating steps are in the {} block
        """
        # check that there is exactly one metropolized region
        #this pattern matches the { literally, then any number of any character other than }, followed by another {
        #If there's a match, then we have an attempt at a nested metropolization, which is unsupported
        regex_nested_metropolis = "{[^}]*{"
        pattern = re.compile(regex_nested_metropolis)
        if pattern.match(splitting.replace(" ", "")):
            raise ValueError("There can only be one Metropolized region.")

        # find the metropolization steps:
        M_start_index = splitting.find("{")
        M_end_index = splitting.find("}")

        # accept/reject happens before the beginning of metropolis step
        if M_start_index > M_end_index:
            return False

        #pattern to find whether any shadow work producing steps lie outside the metropolization region
        RV_outside_metropolis = "[RV](?![^{]*})"
        outside_metropolis_check = re.compile(RV_outside_metropolis)
        if outside_metropolis_check.match(splitting.replace(" ","")):
            return False
        else:
            return True

    def _add_R_step(self):
        """Add an R step (position update) given the velocities.
        """
        if self._measure_shadow_work:
            self.addComputeGlobal("old_pe", "energy")
            self.addComputeSum("old_ke", self._kinetic_energy)

        n_R = self._ORV_counts['R']

        # update positions (and velocities, if there are constraints)
        self.addComputePerDof("x", "x + (v * dt) + (sqrt(2 * dt) * sigma * gaussian)")
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeGlobal("new_pe", "energy")
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")

    def _add_V_step(self, force_group="0"):
        """Deterministic velocity update, using only forces from force-group fg.

        Parameters
        ----------
        force_group : str, optional, default="0"
           Force group to use for this step
        """
        if self._measure_shadow_work:
            self.addComputeGlobal("old_pe", "energy")
            self.addComputeSum("old_ke", self._kinetic_energy)

        n_R = self._ORV_counts['R']
        
        # # update velocities
        # if self._mts:
        #     self.addComputePerDof("x", "x + ((dt / {}) * f{} / g)".format(self._force_group_nV[force_group], force_group))
        # else:
        #     self.addComputePerDof("x", "x + ((dt / {}) * f / g)".format(self._force_group_nV["0"]))

        self.addComputePerDof("x", "x + (dt  * f / g)")
        # update positions (and velocities, if there are constraints)
    
        # self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        # self.addConstrainPositions()  # x is now constrained
        # self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        # self.addConstrainVelocities()

        self.addComputePerDof("x", "x - (dt  * f0 / g)")
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()


        if self._measure_shadow_work:
            self.addComputeGlobal("new_pe", "energy")
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")
        
        
        
    def _add_O_step(self):
        """Add an O step (stochastic velocity update)
        """
        if self._measure_heat:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        self.addComputePerDof("v", "(a * v) + ((b * f0 / g) * gaussian)")
        self.addConstrainVelocities()

        if self._measure_heat:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

    def _substep_function(self, step_string):
        """Take step string, and add the appropriate R, V, O step with appropriate parameters.

        The step string input here is a single character (or character + number, for MTS)
        """
        function, can_accept_force_groups = self._step_dispatch_table[step_string[0]]
        if can_accept_force_groups:
            force_group = step_string[1:]
            function(force_group)
        else:
            function()

    def _parse_splitting_string(self, splitting_string):
        """Parse the splitting string to check for simple errors and extract necessary information

        Parameters
        ----------
        splitting_string : str
            The string that specifies how to do the integrator splitting

        Returns
        -------
        ORV_counts : dict
            Number of O, R, and V steps
        mts : bool
            Whether the splitting specifies an MTS integrator
        force_group_n_V : dict
            Specifies the number of V steps per force group. {"0": nV} if not MTS
        """
        # convert the string to all caps
        splitting_string = splitting_string.upper()

        # sanity check the splitting string
        self._sanity_check(splitting_string)

        ORV_counts = dict()

        # count number of R, V, O steps:
        for step_symbol in self._step_dispatch_table:
            ORV_counts[step_symbol] = splitting_string.count(step_symbol)

        # split by delimiter (space)
        step_list = splitting_string.split(" ")

        # populate a list with all the force groups in the system
        force_group_list = []
        for step in step_list:
            # if the length of the step is greater than one, it has a digit after it
            if step[0] == "V" and len(step) > 1:
                force_group_list.append(step[1:])

        # Make a set to count distinct force groups
        force_group_set = set(force_group_list)

        # check if force group list cast to set is longer than one
        # If it is, then multiple force groups are specified
        if len(force_group_set) > 1:
            mts = True
        else:
            mts = False

        # If the integrator is MTS, count how many times the V steps appear for each
        if mts:
            force_group_n_V = {force_group: 0 for force_group in force_group_set}
            for step in step_list:
                if step[0] == "V":
                    # ensure that there are no V-all steps if it's MTS
                    assert len(step) > 1
                    # extract the index of the force group from the step
                    force_group_idx = step[1:]
                    # increment the number of V calls for that force group
                    force_group_n_V[force_group_idx] += 1
        else:
            force_group_n_V = {"0": ORV_counts["V"]}

        return ORV_counts, mts, force_group_n_V

    def _add_metropolize_start(self):
        """Save the current x and v for a metropolization step later"""
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def _add_metropolize_finish(self):
        """Add a Metropolization (based on shadow work) step to the integrator.

        When Metropolization occurs, shadow work is reset.
        """
        self.addComputeGlobal("accept", "step(exp(-(shadow_work)/kT) - uniform)")
        self.addComputeGlobal("ntrials", "ntrials + 1")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.addComputeGlobal("nreject", "nreject + 1")
        self.endBlock()
        self.addComputeGlobal("naccept", "ntrials - nreject")
        self.addComputeGlobal("shadow_work", "0")

