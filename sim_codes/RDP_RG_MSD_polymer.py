#==================#
#Import libraries  #
#==================#
from urllib.parse import parse_qs
from OpenMiChroM.CndbTools import cndbTools
import numpy as np
import os
import time
from scipy.integrate import simps
import argparse as arg

cndbTools=cndbTools()
start=time.time()

#==============================#
#Parse command line arguments  #
#==============================#
parser=arg.ArgumentParser()
parser.add_argument('-top',default=None,dest='top',type=str)
parser.add_argument('-f',default=None,dest='datafile',type=str)
parser.add_argument('-s',default='./',dest='savedest',type=str)
parser.add_argument('-if_RG',default=False,dest='if_RG',type=bool)
parser.add_argument('-if_RDP',default=False,dest='if_RDP',type=bool)
parser.add_argument('-R0',default=100,dest='R0',type=float)
# parser.add_argument('-bins',default=100,dest='bins',type=int)
parser.add_argument('-if_MSD',default=False,dest='if_MSD',type=bool)
parser.add_argument('-if_HiC',default=False,dest='if_HiC',type=bool)
parser.add_argument('-if_SXp',default=False,dest='if_SXp',type=bool)
parser.add_argument('-if_rot_relax',default=False,dest='if_rot_relax',type=bool)
parser.add_argument('-save_chains',default=False,dest='save_chains',type=bool)
args=parser.parse_args()

#=======================#
# Analysis Functions    #
#=======================#

def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

#r is an (T,3) ndarray: [time stamps,dof]
def msd_fft(r):
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1, S2


#=============================================#
#Load trajectory and chromosome topology file #
#=============================================#
try:
    all_files=os.listdir('./')

    #Check for topology file and load
    if args.top !=None:
        print('=====================')
        print('Loading topology file: ', args.top) 
        chrm_top=np.loadtxt(
                args.top,
                delimiter=' ',
                dtype=int,
                )

    elif args.top==None:
        count_top=sum(1 for ff in all_files
                        if 'top.txt' in ff) 
        if count_top!=1:
            print('There are either NO or MORE THAN ONE .top files. \n\
                Specify the topology file using cmd line argument: -top <filename>')
            raise IOError
        elif count_top==1:
            for fname in all_files:
                if 'top.txt' in fname:
                    print('=====================')
                    print('Loading topology file: ', fname)
                    chrm_top=np.loadtxt(
                                fname,
                                delimiter=' ',
                                dtype=int,
                                )
    #Check for trajectory file and load
    if args.datafile!=None:
        if '.cndb' in args.datafile:
            print('Loading .cndb trajectory: {} ...'.format(args.datafile))
            # cndb_traj=cndbTools.load(args.datafile)
            # all_traj = cndbTools.xyz(
            #             frames=[99,cndb_traj.Nframes,1],
            #             beadSelection='all',
            #             XYZ=[0,1,2]
            #             )
            # savename='analyze_'+args.datafile.replace('.cndb','')
            # del cndb_traj

        elif '.npy' in args.datafile:
            print('Loading .npy trajectory: {} ...'.format(args.datafile))
            all_traj = np.load(args.datafile)
            savename='analyze_'+args.datafile.replace('.npy','')


    elif args.datafile==None:
        #no input given check if there is a unique trajectory file in current directory
        count_cndb=sum(1 for ff in all_files 
                        if '.cndb' in ff)
        if count_cndb!=1:
            count_npy=sum(1 for ff in all_files
                        if 'positions.npy' in ff)
            if count_npy==1:
                for fname in all_files:
                    if 'positions.npy' in fname:
                        print('Loading .npy trajectory: {} ...'.format(fname))
                        all_traj=np.load(fname)
                        savename='analyze_'+fname.replace('_positions.npy','')

            else:
                print('There are either NO or MORE THAN ONE .cndb/.npy files. \n\
                    Specify the data file using cmd line argument: -f <filename>')
                raise IOError
    
        elif count_cndb==1:
            for fname in all_files:
                if '.cndb' in fname:
                    print('Loading .cndb trajectory: {} ...'.format(fname))
            #         cndb_traj=cndbTools.load(fname)
            #         all_traj = cndbTools.xyz(
            #                 frames=[99,cndb_traj.Nframes,1], 
            #                 beadSelection='all', 
            #                 XYZ=[0,1,2]
            #                 )
            #         savename='analyze_'+fname.replace('.cndb','')

            # del cndb_traj
    
    #Number of particles
    Np=all_traj.shape[1]
    #total time
    T=all_traj.shape[0] 
    print('=====================')
    print('Total time steps: ', all_traj.shape[0])
    print('Total number of particles: ', Np)
    print('Number of chains: ', chrm_top.shape[0])
    print('=====================',flush=True)
    # save_chains=False
    def gyr_tensor(X):
        rcm=np.mean(X, axis=1,keepdims=True)
        ws=[]
        N=X.shape[1]
        for val in X-rcm:
            S=np.matmul(np.transpose(val),val)/N
            ws.append(np.sort(np.linalg.eig(S)[0]))
        return np.array(ws)    
    eigs=[]
    for chrm in chrm_top:
        eigs.append(gyr_tensor(all_traj[:,chrm[0]:chrm[1]+1,:]))
    eigs=np.array(eigs)
    #print(eigs.shape)    
    np.save(args.savedest+savename+'_gyr.npy', eigs)        

    #============================#
    #Compute radius of gyration  #
    #============================#
    if args.if_RG == True:
        print('Computing Radius of gyration...\n',flush=True)
        RG=cndbTools.compute_RG(all_traj)
        for chrm in chrm_top:
            RG=np.vstack((RG, 
                    cndbTools.compute_RG(all_traj[:,chrm[0]:chrm[1]+1,:])))
    
        np.savez(args.savedest+savename+'_RadGyr.npz', 
                    genome=RG[0], chrms=RG[1:])
    else: pass

    #=================================================================#
    # Compute Mean-Squared displacement and Position Autocorrelations #
    #=================================================================#
    if args.if_MSD == True:
        print('Computing Mean squared displacements...\n',flush=True)
    
        #MSD and autocorr averaged over all particles
        Pos_autocorr=[]
        MSD=[]
        for p in range(Np):
            S1,S2 = msd_fft(all_traj[:,p,:])
            Pos_autocorr.append(2*S2/S1)
            MSD.append(S1-2*S2)

        MSD=np.array(MSD)
        Pos_autocorr=np.array(Pos_autocorr)

        #average over all particles
        MSD_av=np.mean(MSD,axis=0)
        Pos_autocorr_av=np.mean(Pos_autocorr,axis=0)

        if args.save_chains == True:
    
            #MSD of individual chains
            #MSD_av[0] contains average over all particles
            #MSD_av[i>0] contains MSD averaged over chain i
            for xx in chrm_top:
                MSD_av=np.vstack((MSD_av, 
                            np.mean(MSD[xx[0]:xx[1]+1], axis=0)))
                
                Pos_autocorr_av=np.vstack((Pos_autocorr_av, 
                            np.mean(Pos_autocorr[xx[0]:xx[1]+1], axis=0)))
    
        np.savez(args.savedest+savename+'_MSD-PosAutocorr.npz', 
                        MSD=MSD_av,PAC=Pos_autocorr_av)
    
        #MSD of COM
        S1,S2=msd_fft(np.mean(all_traj,axis=1))
        MSD_COM=S1-2*S2
        PAC_COM=2*S2/S1
        
        for xx in chrm_top:
            S1,S2=msd_fft(np.mean(all_traj[:,xx[0]:xx[1]+1,:],axis=1))
            MSD_COM=np.vstack((MSD_COM,S1-2*S2))
            PAC_COM=np.vstack((PAC_COM,2*S2/S1))
    
        np.savez(args.savedest+savename+'_MSD-PosAutocorr-COM.npz', 
                        MSD_COM=MSD_COM, PAC_COM=PAC_COM)
    else: pass


    #==========================#
    # Normal mode correlations #
    #==========================#
    if args.if_SXp==True:
        print("Computing Normal mode correlations ... \n",flush=True)
        
        modes_p=[1,2,3,4,7,10,15,20]
        Sp_all=[]
        for hh, chrm in enumerate(chrm_top):
            pol_xyz=all_traj[:,chrm[0]:chrm[1]+1,:]
            N_pol=pol_xyz.shape[1]
        
            Sp=[]
            for ii,p in enumerate(modes_p):
                Xp_t = (1/N_pol)*np.einsum('ijk,j->ik',pol_xyz,
                                        np.cos((p*np.pi/N_pol)*(np.arange(1,N_pol+1,1)-0.5)))                
                Sp.append(msd_fft(Xp_t)[1])
            Sp_all.append(Sp)
        
        SXp=np.array(Sp_all)
        
        if args.save_chains == False:
            SXp=np.mean(SXp, axis=0)
        
        # print(SXp.shape)
        np.savez(args.savedest+savename+'_SXp.npz', 
                        SXp=SXp, p=modes_p,)
    else: pass

    #=======================#
    # Rotational relaxation #
    #=======================#
    if args.if_rot_relax == True:
        print('Computing Rotational relaxation ...\n',flush=True)
        S_ee=[]
        for hh, chrm in enumerate(chrm_top):
            Ree=all_traj[:,chrm[1],:]-all_traj[:,chrm[0],:]
            S_ee.append(msd_fft(Ree)[1])
        See_av=np.mean(np.array(S_ee),axis=0)

        np.save(args.savedest+savename+'_rot_relax_av.npy', See_av)
    else: pass

    
    #==================#
    #Compute HiC maps  #
    #==================#
    
    if args.if_HiC == True:
        print('Computing HiC map ...\n',flush=True)
        hic_map=cndbTools.traj2HiC(all_traj)
        np.save(args.savedest+savename+'_HiC.npy',
                hic_map)
    else: pass

    #=====================================#
    #Compute radial distribution profile  #
    #=====================================#
    if args.if_RDP == True:
        R0=float(args.R0)
        # bin_no=int(args.bins)
        dr=0.1
        print('Computing Radial distribution profile...\n')
        print('Rmax={:.0f}, dr={:.2f}'.format(R0,dr),flush=True)
        bin_edges=np.arange(0,R0+8,dr)
        bin_mids=(bin_edges[:-1]+bin_edges[1:])/2
        rdp_tot = np.zeros(len(bin_mids))
        rdp_i=[]

        for hh, chrm in enumerate(chrm_top):
            rdp_i.append(np.zeros(len(bin_mids)))
            # print(chrm)
            for snap in all_traj:
                rdp_tot += np.histogram(np.linalg.norm(snap,axis=1), 
                                bins=bin_edges, density=False)[0]/T
                if args.save_chains is True:
                    rdp_i[hh] += np.histogram(np.linalg.norm(snap[chrm[0]:chrm[1]+1,:],axis=1), 
                                    bins=bin_edges, density=False)[0]/T
        
        # norm=simps(rdp_tot,bin_mids)
        bin_vol=(4/3)*np.pi*(bin_edges[1:]**3-bin_edges[:-1]**3)
        
        RDP_val=np.vstack(( bin_mids, rdp_tot/bin_vol))

        if args.save_chains is True:
            for val in rdp_i:
                RDP_val=np.vstack((RDP_val, val/bin_vol))
        
        np.savez(args.savedest+savename+'_RDPs.npz', 
                    hist=RDP_val[1:], bins=RDP_val[0],)
                
    else: pass
    
    #==========#
    #FINISHED  #
    #==========#
    print('=====================')
    print("\m/Finished!!\m/")
    print('Saved data at {}'.format(args.savedest))
    print("--------------")
    print("Total run time: {:.0f} secs".format(time.time()-start))
    print("--------------")
    
except (IOError):
    print('EXITING!!!')
    pass

