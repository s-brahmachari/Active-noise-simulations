#==================#
#Import libraries  #
#==================#
import argparse as arg
import ActivePolymer
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

sim=ActivePolymer.ActivePolymer(
    name=savename,
    platform='hip', 
    time_step=args.dt, 
    collision_rate=args.gamma, 
    temperature=T, 
    active_corr_time=t_corr, 
    activity_amplitude=F,
    outpath=args.opath, 
    init_struct=None, 
    seq_file=args.fseq, 
    active_particles=range(Na),
    )

ActivePolymer.addHarmonicBonds(sim, top_file=args.ftop, kb=args.kb, d=1.0)

ActivePolymer.addRadialConfinement(sim, R0=args.R0, method='FlatBottomHarmonic', kr=args.kr)

ActivePolymer.runSims(sim, nblocks=args.nblocks, blocksize=args.blocksize, )

print('\n\m/Finished!\m/\nSaved trajectory at: {}'.format(args.opath))
print('******\nTotal run time: {:.2f} secs\n******'.format(time.time()-start))

