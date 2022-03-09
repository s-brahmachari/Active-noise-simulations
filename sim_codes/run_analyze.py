import AnalyzeTrajectory
import numpy as np
import argparse as arg

#==============================#
#Parse command line arguments  #
#==============================#
parser=arg.ArgumentParser()
parser.add_argument('-top',default=None,dest='top',type=str)
parser.add_argument('-f',default=None,dest='datafile',type=str)
parser.add_argument('-s',default='./',dest='savedest',type=str)
parser.add_argument('-gyr',action='store_true')
parser.add_argument('-RDP',action='store_true')
parser.add_argument('-MSD',action='store_true')
parser.add_argument('-HiC',action='store_true')
parser.add_argument('-SXp',action='store_true')
parser.add_argument('-bondlen',action='store_true')

args=parser.parse_args()

traj=AnalyzeTrajectory.AnalyzeTrajectory(datafile=args.datafile, top_file=args.top, discard_init_steps=20000)
savename=args.savedest+traj.savename

if args.gyr:
    gyr_eigs=traj.compute_GyrationTensorEigs()
    np.save(savename+'_GyrEigs.npy', gyr_eigs)

if args.RDP:
    rad_dens_hist, bins = traj.compute_RadNumDens()
    np.savez(savename+'_RadNumDens.npz', hist=rad_dens_hist, bins=bins)

if args.bondlen:
    bondlen_hist,bins=traj.compute_BondLenDist()
    np.savez(savename+'_bondlens.npz', hist=bondlen_hist, bins=bins)

if args.MSD:
    msd, msd_com=traj.compute_MSD_chains(chains=False, COM=False)
    np.save(savename+'_MSD.npy', msd)


