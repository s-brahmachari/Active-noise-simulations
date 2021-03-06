import AnalyzeTrajectory
import numpy as np
import argparse as arg
import time

#==============================#
#Parse command line arguments  #
#==============================#
parser=arg.ArgumentParser()
parser.add_argument('-top',default=None,dest='top',type=str)
parser.add_argument('-seq',default=None,dest='seq',type=str)
parser.add_argument('-datapath',default='./',dest='datapath',type=str)
parser.add_argument('-f',default=None,dest='datafile',type=str)
parser.add_argument('-s',default='./',dest='savedest',type=str)
parser.add_argument('-rep',default='1',dest='rep',type=str)

parser.add_argument('-gyr',action='store_true')
parser.add_argument('-IPD',action='store_true')
parser.add_argument('-RDP',action='store_true')
parser.add_argument('-VCV',action='store_true')
parser.add_argument('-MSD',action='store_true')
parser.add_argument('-HiC',action='store_true')
parser.add_argument('-SXp',action='store_true')
parser.add_argument('-relMSD',action='store_true')
parser.add_argument('-comRDP',action='store_true')
parser.add_argument('-bondlen',action='store_true')

args=parser.parse_args()
start=time.time()

traj=AnalyzeTrajectory.AnalyzeTrajectory(datapath=args.datapath, datafile=args.datafile,
                           top_file=args.top, discard_init_steps=1000, seq_file=args.seq,
                           beadSelection='all')

trajA=None
trajB=None
# trajA=AnalyzeTrajectory.AnalyzeTrajectory(datapath=args.datapath, datafile=args.datafile, 
#                             top_file=args.top, discard_init_steps=20000, seq_file=args.seq,
#                             beadSelection='A')

# trajB=AnalyzeTrajectory.AnalyzeTrajectory(datapath=args.datapath, datafile=args.datafile,
#                            top_file=args.top, discard_init_steps=20000, seq_file=args.seq,
#                            beadSelection='B')

savename=args.savedest+traj.savename+'_rep{}'.format(args.rep)

if args.gyr:
    (gyr_eigs, rg, asph, acyl)=traj.compute_GyrationTensor()
    np.save(savename+'_GyrEigs.npy', gyr_eigs)
    np.savez(savename+'_shape_descriptors.npz',rg=rg,asph=asph,acyl=acyl)

if args.comRDP:
    rad_dens_hist, bins = traj.compute_comRadNumDens(dr=0.25)
    np.savez(savename+'_comRadNumDens.npz', hist=rad_dens_hist, bins=bins)

if args.RDP:
    rad_dens_hist, bins = traj.compute_RadNumDens(dr=0.25)
    np.savez(savename+'_RadNumDens.npz', hist=rad_dens_hist, bins=bins)

    if trajA is not None:
        rad_dens_hist, bins = trajA.compute_RadNumDens(dr=0.25)
        np.savez(savename+'_RadNumDens_A.npz', hist=rad_dens_hist, bins=bins)
    if trajB is not None:
        rad_dens_hist, bins = trajB.compute_RadNumDens(dr=0.25)
        np.savez(savename+'_RadNumDens_B.npz', hist=rad_dens_hist, bins=bins)

if args.bondlen:
    bondlen_hist,bins=traj.compute_BondLenDist()
    np.savez(savename+'_bondlens.npz', hist=bondlen_hist, bins=bins)

if args.MSD:
    msd, msd_com=traj.compute_MSD_chains(chains=False, COM=False)
    np.save(savename+'_MSD.npy', msd)

    if trajA is not None:
        msd, msd_com=trajA.compute_MSD_chains(chains=False, COM=False)
        np.save(savename+'_MSD_A.npy', msd)
    if trajB is not None:
        msd, msd_com=trajB.compute_MSD_chains(chains=False, COM=False)
        np.save(savename+'_MSD_B.npy', msd)

if args.relMSD:
    msd25,msd50,msd75=traj.compute_rel_MSD_chains()
    np.savez(savename+'_RelMSD.npz', msd25=msd25,msd50=msd50,msd75=msd75)

if args.HiC:
    hic=traj.compute_HiC(mu=3,rc=1.5)
    np.save(savename+'_HiC.npy', hic)

if args.IPD:
    rij_hist, bins=traj.compute_InterParticleDist()
    np.savez(savename+'_IPD.npz', hist=rij_hist, bins=bins)

if args.VCV:
    vcv_hist,vcv_bins=traj.compute_VoronoiCellVol(Vmax=300., dv=1.0)
    np.savez(savename+'_VCV.npz', hist=vcv_hist, bins=vcv_bins)

if args.SXp:
    sxp,modes=traj.compute_ModeAutocorr(modes=[1,2,3,4,7,10,15,20])
    np.savez(savename+'_SXp.npz',sxp=sxp,modes=modes)
        
print('=====================')
print("\m/Finished!!\m/")
print('Saved data at {}'.format(args.savedest))
print("--------------")
print("Total run time: {:.0f} secs".format(time.time()-start))
print("--------------")
