from OpenMiChroM.ChromDynamics import MiChroM
import numpy as np
import pandas as pd

def addHarmonicBond(self, i, j, kb=0.1,d=1.):

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
    d=d * self.Sigma * self.length_scale
    
    
    if "HarmonicBond" not in list(self.forceDict.keys()):
        force = ("0.5 * k_har * (r - r0_har) * (r - r0_har)")
        bondforceGr = self.mm.CustomBondForce(force)
        bondforceGr.addGlobalParameter("k_har", kb)
        bondforceGr.addGlobalParameter("r0_har", d)

        self.forceDict["HarmonicBond"] = bondforceGr

    self.forceDict["HarmonicBond"].addBond(int(i), int(j), [])
    self.bondsForException.append((int(i), int(j)))

    return True


def addCustomFENEBond(self, i, j, eh=0.01, kfb=0.1):
    eh=eh * self.Epsilon
    
    if "CustomFENEBond" not in list(self.forceDict.keys()):
        force = ("- 0.5 * kfb * r0 * r0 * log(1-(r/r0)*(r/r0)) + (4 * e * ((s/r)^12 - (s/r)^6) + e) * step(cut - r)")
        bondforceGr = self.mm.CustomBondForce(force)
        bondforceGr.addGlobalParameter("kfb", kfb)
        bondforceGr.addGlobalParameter("r0", 1.5) 
        bondforceGr.addGlobalParameter('e', eh)   
        bondforceGr.addGlobalParameter('s', 1.0)
        bondforceGr.addGlobalParameter("cut", 2.**(1./6.))

        self.forceDict["CustomFENEBond"] = bondforceGr

    self.forceDict["CustomFENEBond"].addBond(int(i), int(j), [])
    
    return True


def addTanhHarmonicBond(self, i, j, eh=5, d=1, kb=1, mu=3, rc=1.5):
    
    d = d * self.Sigma * self.length_scale
    rc = rc * self.Sigma * self.length_scale
    eh = eh * self.Epsilon
    
    if "TanhHarmonicBond" not in list(self.forceDict.keys()):
        force = ("0.5 * kb * (r - r0) * (r - r0) * step(r - r0) + 0.5 * Ehard * (1 + tanh(mu * (rc - r)))")
        
        bondforceGr = self.mm.CustomBondForce(force)
        bondforceGr.addGlobalParameter("kb", kb)
        bondforceGr.addGlobalParameter("r0", d)
        bondforceGr.addGlobalParameter('Ehard', eh)
        bondforceGr.addGlobalParameter('mu', mu)
        bondforceGr.addGlobalParameter("rc", rc)

        self.forceDict["TanhHarmonicBond"] = bondforceGr

    self.forceDict["TanhHarmonicBond"].addBond(int(i), int(j), [])
    self.bondsForException.append((int(i), int(j))) 
    return True


def addTanhRepulsion(self, es=1, mu=3, rc=1.5):
    
    rc = rc * self.Sigma * self.length_scale
    es = es * self.Epsilon
    
    if "TanhRepulsion" not in list(self.forceDict.keys()):
        repel_energy = ("0.5 * Esoft * (1 + tanh(mu * (rc - r)))")
        
        repelforceGr = self.mm.CustomNonbondedForce(repel_energy)
        repelforceGr.addGlobalParameter('Esoft', es)
        repelforceGr.addGlobalParameter('mu', mu)
        repelforceGr.addGlobalParameter("rc", rc)
        repelforceGr.addGlobalParameter("lim", 1.0)
        repelforceGr.setCutoffDistance(3.0)

        self.forceDict["TanhRepulsion"] = repelforceGr

    for _ in range(self.N):
        self.forceDict["TanhRepulsion"].addParticle(())
    
    return True



def addRadialConfinement(self, R0=None, vol_frac=None, LJ=False, FlatBottomHarmonic=False, kr=5.0):
        
        sigma=self.Sigma * self.length_scale
        
        try:
        
            if R0 is None and vol_frac is None:
                print("Error! specify either R (radius of confinement) or vol_frac (confinement volume fraction)")
                raise ValueError
        
            elif R0 is None and vol_frac is not None:
                R0=sigma*(self.N / (8 * vol_frac))**(1 / 3)
                vol_frac = float(vol_frac)
        
            elif R0 is not None and vol_frac is None:
                R0=float(R0)
                vol_frac= self.N * (sigma / (2 * R0))**3
            
            if (LJ is True and FlatBottomHarmonic is True) or (LJ is False and FlatBottomHarmonic is False):
                print("Error! specify type of confinement: Lennard-Jones or Flat-Bottom Harmonic")
                raise ValueError
                
            elif LJ is True and FlatBottomHarmonic is False:
                print("Implementing Lennard-Jones confinement:\n Radius: {0:.2f} \n Volume fraction: {1:.3f}".format(R0,vol_frac))

                LJ_conf = self.mm.CustomExternalForce("(4 * GROSe * (GROSs/r)^12 + GROSe) * step(GROScut - r);"
                                                 "r= R - sqrt(x^2 + y^2 + z^2) ")
                LJ_conf.addGlobalParameter('R', R0)
                LJ_conf.addGlobalParameter('GROSe', 1.0)
                LJ_conf.addGlobalParameter('GROSs', 1.0)
                LJ_conf.addGlobalParameter("GROScut", 2.**(1./6.))
            
                self.forceDict["RadialConfinement"] = LJ_conf
            
            elif LJ is False and FlatBottomHarmonic is True:
                print("Implementing Flat-Bottom Harmonic confinement:\n Radius: {0:.2f} \n Volume fraction: {1:.3f} \n Stiffness: {2:.2f}".format(R0,vol_frac,kr))

                FBH_conf = self.mm.CustomExternalForce("step(r-r_res) * 0.5 * kr * (r-r_res)^2; r=sqrt(x*x+y*y+z*z)")
                FBH_conf.addGlobalParameter('r_res', R0)
                FBH_conf.addGlobalParameter('kr', kr)
                
                self.forceDict["RadialConfinement"] = FBH_conf
                
            for i in range(self.N):
                self.forceDict["RadialConfinement"].addParticle(i, [])

            self.ConfinementRadius = R0
            ret=True
        
        except (ValueError):
            print("No confinement added!")
            ret=False
            pass
        
        return ret
        
def set_activity(self,F_act=1,particle_list=[]):
    act_force=self.mm.CustomExternalForce(" - f_act * (x + y + z)")
    act_force.addGlobalParameter('f_act',F_act)

    self.forceDict["ActiveForce"]=act_force
    # print(self.forceDict["ActiveForce"].getForceGroup())
    for ii in particle_list:
        self.forceDict["ActiveForce"].addParticle(ii,[])

    return True



def addCustomTypes(self, name="CustomTypes", mu=3, rc = 1.5, TypesTable=None,):
    R"""
    Adds the type-to-type potential using custom values for interactions between the chromatin types. The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
    
    The function receives a txt/TSV/CSV file containing the upper triangular matrix of the type-to-type interactions. A file example can be found `here <https://www.ndb.rice.edu>`__.
    
    +---+------+-------+-------+
    |   |   A  |   B   |   C   |
    +---+------+-------+-------+
    | A | -0.2 | -0.25 | -0.15 |
    +---+------+-------+-------+
    | B |      |  -0.3 | -0.15 |
    +---+------+-------+-------+
    | C |      |       | -0.35 |
    +---+------+-------+-------+
    
    Args:

        name (string, required):
            Name to customType Potential. (Default value = "CustomTypes") 
        mu (float, required):
            Parameter in the probability of crosslink function. (Default value = 3.22).
        rc (float, required):
            Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
        TypesTable (file, required):
            A txt/TSV/CSV file containing the upper triangular matrix of the type-to-type interactions. (Default value: :code:`None`).


    """
    def _types_Letter2number(self, header_types):
        R"""
        Internal function for indexing unique chromatin types.
        """
        type2number = {}
        for i,t in enumerate(header_types):
            type2number[t] = i
        # print(type2number)
        # print(self.type_list_letter)
        for bead in self.type_list_letter:
            self.type_list.append(type2number[bead])
        # print(self.type_list)


    self.metadata["CrossLink"] = repr({"mu": mu})
    if not hasattr(self, "type_list"):
            self.type_list = self.random_ChromSeq(self.N)

    energy = "mapType(t1,t2)*0.5*(1. + tanh(mu*(rc - r)))*step(r-lim)"
    
    crossLP = self.mm.CustomNonbondedForce(energy)

    crossLP.addGlobalParameter('mu', mu)
    crossLP.addGlobalParameter('rc', rc)
    crossLP.addGlobalParameter('lim', 1.0)
    crossLP.setCutoffDistance(3.0)

    tab = pd.read_csv(TypesTable, sep=None, engine='python')

    header_types = list(tab.columns.values)

    if not set(self.diff_types).issubset(set(header_types)):
        errorlist = []
        for i in self.diff_types:
            if not (i in set(header_types)):
                errorlist.append(i)
        raise ValueError("Types: {} are not present in TypesTables: {}\n".format(errorlist, header_types))

    diff_types_size = len(header_types)
    lambdas = np.triu(tab.values) + np.triu(tab.values, k=1).T
    lambdas = list(np.ravel(lambdas))
        
    fTypes = self.mm.Discrete2DFunction(diff_types_size,diff_types_size,lambdas)
    crossLP.addTabulatedFunction('mapType', fTypes) 
        
    _types_Letter2number(self,header_types)
    crossLP.addPerParticleParameter("t")

    for i in range(self.N):
            value = [float(self.type_list[i])]
            crossLP.addParticle(value)
            # print(jj)
            
            
    self.forceDict[name] = crossLP
    
    
        