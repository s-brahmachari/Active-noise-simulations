# ============================================================================================
# MODULE DOCSTRING
# ============================================================================================

"""
Custom integrators for molecular simulation.

DESCRIPTION

This module provides various custom integrators for OpenMM.

EXAMPLES

COPYRIGHT

@author John D. Chodera <john.chodera@choderalab.org>

All code in this repository is released under the MIT License.

This program is free software: you can redistribute it and/or modify it under
the terms of the MIT License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the MIT License for more details.

You should have received a copy of the MIT License along with this program.

"""

# ============================================================================================
# GLOBAL IMPORTS
# ============================================================================================

import logging
import re

import numpy as np
import simtk.unit as unit
import simtk.openmm as mm

from openmmtools.constants import kB
from openmmtools import respa, utils
from openmmtools.integrators import PrettyPrintableIntegrator as PrettyPrintableIntegrator
from openmmtools.integrators import ThermostatedIntegrator as ThermostatedIntegrator
logger = logging.getLogger(__name__)

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole

# ============================================================================================
# BASE CLASSES
# ============================================================================================

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

