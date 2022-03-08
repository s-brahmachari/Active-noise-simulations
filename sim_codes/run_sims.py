#==================#
#Import libraries  #
#==================#
import argparse as arg
from OpenMiChroM.ChromDynamics import MiChroM
from openmmtools import integrators
from simtk import unit
from Custom_integrators import CustomBrownianIntegrator
import Custom_Bonds
import numpy as np
import time
start=time.time()
#==============================#
#Parse command line arguments  #
#==============================#
parser=arg.ArgumentParser()
parser.add_argument('-ftop',required=True,dest='ftop',type=str)
parser.add_argument('-fseq',required=True,dest='fseq',type=str)
parser.add_argument('-finit',required=True,dest='finit',type=str)

parser.add_argument('-name',required=True,dest='name',type=str)
parser.add_argument('-Ta',required=True,dest='Ta',type=str)
parser.add_argument('-Na',required=True,dest='Na',type=str)
parser.add_argument('-temp',required=True,dest='temp',type=str)
parser.add_argument('-F',required=True,dest='F',type=str)
parser.add_argument('-Esoft',required=True,dest='Esoft',type=str)
parser.add_argument('-vol_frac',default=None,dest='vol_frac',type=str)
parser.add_argument('-R0',default=None,dest='R0',type=str)

parser.add_argument('-rep',default=1,dest='replica',type=int)
parser.add_argument('-gamma',default=0.1,type=float,dest='gamma')
parser.add_argument('-kb',default=10.0,dest='kb',type=float)
parser.add_argument('-dt',default=0.001,dest='dt',type=float)
parser.add_argument('-nblocks',default=1000,dest='nblocks',type=int)
parser.add_argument('-blocksize',default=500,dest='blocksize',type=int)
parser.add_argument('-outpath',default='./',dest='opath',type=str)
parser.add_argument('-kr',default='20',dest='kr',type=float)

args=parser.parse_args()

#Define name
savename=args.name+"_T{0:}_F{1:}_Ta{2:}_Esoft{3:}_R0{4:}_Na{5:}_blocksize{6:}_kb{7:}_dt{8:}_kr{9:}".format(
                                args.temp, args.F, args.Ta,
                                args.Esoft, args.R0, args.Na,
                                args.blocksize,args.kb, args.dt, args.kr)

#=======================#
#Simulation Parameters  #
#=======================#
try:
    T=float(args.temp)
    t_corr= float(args.Ta)
    F=float(args.F)
    Na=int(args.Na)
    Esoft=float(args.Esoft)
except(ValueError) as msg:
    print(msg)
    print('Critical ERROR in simulation parameters!! Exiting!')
    pass

#==========================#
#Initialize Michrom Class  #
#==========================#
sim=MiChroM(name=savename, velocity_reinitialize=False)

#Specify integrator
integrator=CustomBrownianIntegrator(
                    timestep=args.dt * unit.picoseconds, 
                    collision_rate=args.gamma / unit.picoseconds,
                    temperature=T * unit.kelvin,measure_heat=False,
                    noise_corr=t_corr * unit.picoseconds,
                    )

sim.setup(platform="hip",integrator=integrator,)
sim.saveFolder(args.opath)

#Initial structure (spiral by default)
if '.npy' not in args.finit:
    chrm=sim.create_springSpiral(ChromSeq=args.fseq)
else:
    chrm=np.load(args.finit)
sim.loadStructure(chrm,center=True)
sim._translate_type(args.fseq)

#Initialize save structure
#sim.saveStructure(filename=savename+'_initpos',mode='gro')
#sim.initStorage('traj_'+savename, mode='w')

#===================#
# Add activity      #
#===================#
Custom_Bonds.set_activity(sim,F_act=F, particle_list=range(sim.N))


#==================#
#Add interactions  #
#==================#

##---------#
##polymers #
##---------#

#chromosome topology
chrm_top=np.loadtxt(args.ftop, delimiter=' ', dtype=int)

##add bonds between nearest neighbors
for ii in range(sim.N-1):
    #skip bonds between different chromosomes
    if ii in chrm_top[:,1]: continue
    Custom_Bonds.addFENEBond(sim, ii, ii+1, kfb=args.kb)
    #Custom_Bonds.addHarmonicBond(sim, ii, ii+1, kb=args.kb, d=0) 

## circularize if chromosome is circular
#for xx in chrm_top:
#    if xx[2]==1: 
#        #addTanhHarmonicBond(sim, xx[0], xx[1], eh=20, kb=args.kb)
#        addHarmonicBond(sim, xx[0], xx[1], kb=args.kb,d=1)


##----------#
##Dumbbells #
##----------#
#for ii in range(sim.N-1):
#    if ii%2==0:
#        addHarmonicBond(sim, ii, ii+1, kb=args.kb,d=1)


#=====================#
# Radial confinement  #
#=====================#

Custom_Bonds.addRadialConfinement(sim, vol_frac=args.vol_frac, R0=args.R0, FlatBottomHarmonic=True, kr=args.kr)

#=====================#
# Soft-core repulsion #
#=====================#
# Custom_Bonds.addTanhRepulsion(sim, es=Esoft)

#=========================#
# Inter-monomer adhesion  #
#=========================#
# sim.addCustomTypes(mu=3, rc=1, TypesTable='type_table.csv')

#===============#
#Run simulation #
#===============#
positions=[]
for ii, _ in enumerate(range(args.nblocks)):
    sim.runSimBlock(args.blocksize)
    #sim.saveStructure()
    if ii<20000: continue
    state = sim.context.getState(getPositions=True,
            getVelocities=False, getEnergy=False)
    #vel = state.getVelocities(asNumpy=True)#/unit.sqrt(unit.kilojoule_per_mole / mass)
    positions.append(state.getPositions(asNumpy=True))
    
    if ii%int(args.nblocks*0.1)==0 and ii>0==0:
        np.save(args.opath+'traj_'+savename+'_positions.npy',np.array(positions))
#save cndb file
#sim.storage[0].close()

#np.save(args.opath+savename+'_velocities.npy',Velocities)
np.save(args.opath+'traj_'+savename+'_positions.npy',np.array(positions))

print('\n\m/Finished!\m/\nSaved trajectory at: {}'.format(args.opath))
print('******\nTotal run time: {:.2f} secs\n******'.format(time.time()-start))

