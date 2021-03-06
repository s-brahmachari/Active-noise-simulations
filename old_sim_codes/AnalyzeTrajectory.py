
from OpenMiChroM.CndbTools import cndbTools
import numpy as np
import os
from scipy.spatial import distance, Voronoi, ConvexHull

cndbT=cndbTools()

def _autocorrFFT(x):
        N=len(x)
        F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res= (res[:N]).real   #now we have the autocorrelation in convention B
        n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
        return res/n #this is the autocorrelation in convention A

def _msd_fft(r):
    #r is an (T,3) ndarray: [time stamps,dof]
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([ _autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1, S2


class AnalyzeTrajectory():

    def __init__(self, datapath='./', datafile=None, top_file=None, seq_file=None, discard_init_steps=0, beadSelection='all'):
        try:
            all_files=os.listdir(datapath)
            print('=====================\nLoading {} beads:\n'.format(beadSelection))
            #print(all_files)
            #Check for topology file and load
            if top_file is not None:
                print('Loading topology file: {} ...'.format(top_file) , end=' ')
                chrm_top=np.loadtxt(top_file, delimiter=' ',dtype=int, ndmin=2)
                print('done!\n')

            elif top_file is None:
                count_top=sum(1 for ff in all_files if 'top.txt' in ff) 
                if count_top!=1:
                    print('There are either NO or MORE THAN ONE .top files. \n\
                        Specify the topology file using cmd line argument: -top <filename>')
                    raise IOError

                elif count_top==1:
                    for fname in all_files:
                        if 'top.txt' in fname:
                            print('Loading topology file: {} ...'.format(fname), end=' ')
                            chrm_top=np.loadtxt(datapath+fname,delimiter=' ',dtype=int, ndmin=2)
                            print('done!\n')

            if seq_file is not None:
                print('Loading sequence file: {} ...'.format(seq_file) , end=' ')
                chr_seq=np.loadtxt(seq_file, delimiter=' ',dtype=str,)
                print('done!\n')

            elif seq_file is None:
                count_seq=sum(1 for ff in all_files if 'seq' in ff) 
                if count_seq!=1:
                    print('There are either NO or MORE THAN ONE .top files. \n\
                        Specify the topology file using cmd line argument: -top <filename>')
                    raise IOError

                elif count_seq==1:
                    for fname in all_files:
                        if 'seq' in fname:
                            print('Loading sequence file: {} ...'.format(fname), end=' ')
                            chr_seq=np.loadtxt(datapath+fname,delimiter=' ',dtype=str,)
                            print('done!\n')

            #Check for trajectory file and load
            if datafile is not None:
                
                if '.cndb' in datafile:
                    print('Loading .cndb trajectory: {} ...'.format(datafile), end=' ', flush=True)
                    cndb_traj=cndbT.load(datapath+datafile)
                    all_traj = cndbTools.xyz(frames=[discard_init_steps,cndb_traj.Nframes,1],
                                beadSelection='all', XYZ=[0,1,2])
                    savename='analyze_'+datafile.replace('.cndb','')
                    del cndb_traj
                    print('done!\n', flush=True)

                elif '.npy' in datafile:
                    print('Loading .npy trajectory: {} ...'.format(datafile), end=' ',flush=True)
                    all_traj = np.load(datapath+datafile)[discard_init_steps:,:,:]
                    savename='analyze_'+datafile.replace('.npy','')
                    print('done!\n', flush=True)


            elif datafile is None:
                #no input given check if there is a unique trajectory file in current directory
                count_cndb=sum(1 for ff in all_files if '.cndb' in ff)
                if count_cndb!=1:
                    count_npy=sum(1 for ff in all_files if 'positions.npy' in ff)
                    if count_npy==1:
                        for fname in all_files:
                            if 'positions.npy' in fname:
                                print('Loading .npy trajectory: {} ...'.format(fname), end=' ',flush=True)
                                all_traj=np.load(datapath+fname)[discard_init_steps:,:,:]
                                savename='analyze_'+fname.replace('_positions.npy','')
                        
                        print('done!\n', flush=True)

                    else:
                        print('There are either NO or MORE THAN ONE .cndb/.npy files. \n\
                            Specify the data file using cmd line argument: -f <filename>')
                        raise IOError
            
                elif count_cndb==1:
                    for fname in all_files:
                        if '.cndb' in fname:
                            print('Loading .cndb trajectory: {} ...'.format(fname), end=' ',flush=True)
                            cndb_traj=cndbT.load(datapath+fname)
                            all_traj = cndbT.xyz(frames=[discard_init_steps,cndb_traj.Nframes,1], 
                                    beadSelection='all', XYZ=[0,1,2])
                            savename='analyze_'+fname.replace('.cndb','')
                            del cndb_traj

                            print('done!\n', flush=True)
            
            if beadSelection !='all':
                select_traj=[]
                for ii,line in enumerate(chr_seq):
                    #print(line)
                    if beadSelection==line[1]:
                        select_traj.append(all_traj[:,ii,:])
                N_select=len(select_traj)
                select_traj=np.array(select_traj).reshape(all_traj.shape[0],N_select,3)
                all_traj=select_traj

            self.xyz = all_traj
            self.N = all_traj.shape[1]
            self.T = all_traj.shape[0]
            self.top = chrm_top
            self.seq = chr_seq
            self.savename = savename
            
            print("----------------\nTrajectory object created\n----------------")
            print("Number of particles: {}".format(self.N))
            print("Total time steps: {}\n----------------\n\n".format(self.T))
 
        except (IOError,):
            print("--------\nERROR!!! Could not load trajectory or top file. EXITING!\n-------")
            
    def compute_GyrationTensor(self):

        def gyr_tensor(X):
            rcm=np.mean(X, axis=1,keepdims=True)
            ws=[]
            n=X.shape[1]
            for val in X-rcm:
                S_mat=np.matmul(np.transpose(val),val)/n
                ws.append(np.sort(np.linalg.eig(S_mat)[0]))
            return np.array(ws)    

        print('Computing Gyration Tensor Eigenvalues ...',flush=True, end=' ')

        eigs=[]
        for chrm in self.top:
            eigs.append(gyr_tensor(self.xyz[:,chrm[0]:chrm[1]+1,:]))
        eigs=np.array(eigs)
        flat_eigs=eigs.reshape(-1,3)

        #compute RG_hist
        rg_vals=np.ravel(np.sqrt(np.sum(flat_eigs, axis=1)))
        rg_hist,bin_edges=np.histogram(rg_vals, bins=np.arange(0,rg_vals.max()+5,0.5), density=True)
        rg_bins=0.5*(bin_edges[:-1]+bin_edges[1:])
        
        #compute asphericity
        asph_vals = np.ravel(flat_eigs[:,2] - 0.5*(flat_eigs[:,0]+flat_eigs[:,1]))
        asph_hist,bin_edges=np.histogram(asph_vals, bins=np.logspace(-4,6,400), density=True)
        asph_bins=0.5*(bin_edges[:-1]+bin_edges[1:])

        #compute acylindricity
        acyl_vals=np.ravel(flat_eigs[:,1]-flat_eigs[:,0])
        acyl_hist,bin_edges=np.histogram(acyl_vals, bins=np.logspace(-4,6,400), density=True)
        acyl_bins=0.5*(bin_edges[:-1]+bin_edges[1:])

        print('done!\n', flush=True,)

        return (eigs,(rg_hist, rg_bins),(asph_hist, asph_bins),(acyl_hist,acyl_bins))
            
    def compute_RG_chains(self):
        
        def compute_RG(xyz):
            rcm=np.mean(xyz, axis=1,keepdims=True)
            xyz_rel_to_cm= xyz - np.tile(rcm,(xyz.shape[1],1))
            rg=np.sqrt(np.mean(np.linalg.norm(xyz_rel_to_cm,axis=2)**2,axis=1))
            return rg

        print('Computing Radius of gyration...',flush=True, end=' ')

        rg_chains=compute_RG(self.xyz)
        for chrm in self.top:
            rg_chains=np.vstack((rg_chains, compute_RG(self.xyz[:,chrm[0]:chrm[1]+1,:])))
        
        print('done!\n', flush=True)

        return rg_chains

    def compute_MSD_chains(self,chains=False, COM=False, pos_autocorr=False):

        print('Computing Mean squared displacements...',flush=True, end=' ')
    
        #MSD and autocorr averaged over all particles
        # Pos_autocorr=[]
        msd=[]
        for p in range(self.N):
            s1,s2 = _msd_fft(self.xyz[:,p,:])
            # Pos_autocorr.append(2*S2/S1)
            msd.append(s1-2*s2)

        msd=np.array(msd)
        # Pos_autocorr=np.array(Pos_autocorr)

        #average over all particles
        msd_av=np.mean(msd,axis=0)
        # Pos_autocorr_av=np.mean(Pos_autocorr,axis=0)

        if chains == True:
            #MSD of individual chains
            #MSD_av[0] contains average over all particles
            #MSD_av[i>0] contains MSD averaged over chain i
            for xx in self.top:
                msd_av=np.vstack((msd_av, np.mean(msd[xx[0]:xx[1]+1], axis=0)))
                # Pos_autocorr_av=np.vstack((Pos_autocorr_av, np.mean(Pos_autocorr[xx[0]:xx[1]+1], axis=0)))
        
        msd_COM=None
        if COM==True:
            s1,s2=_msd_fft(np.mean(self.xyz,axis=1))
            msd_COM=s1-2*s2
            # PAC_COM=2*S2/S1
            for xx in self.top:
                s1,s2=_msd_fft(np.mean(self.xyz[:,xx[0]:xx[1]+1,:],axis=1))
                msd_COM=np.vstack((msd_COM,s1-2*s2))
                # PAC_COM=np.vstack((PAC_COM,2*S2/S1))
        print('done!\n', flush=True)

        return (msd_av,msd_COM)

    #need to check this!!
    def compute_rel_MSD_chains(self,):

        print('Computing relative Mean squared displacements...',flush=True, end=' ')
        msd_25,msd_50, msd_75=[],[],[]

        for chrm in self.top:
            mi,mf=chrm[0], chrm[1]
            mono_mid,mono_1,mono_2=mi+int((mf-mi)/2), mi+int((mf-mi)/4), mi+int((mf-mi)*3/4)
            mono_3,mono_4=mi+int((mf-mi)*0.1),mi+int((mf-mi)*0.85)

            s1,s2=_msd_fft(self.xyz[:,mono_mid,:]-self.xyz[:,mono_1])
            msd_25.append(s1-2*s2)
            
            s1,s2=_msd_fft(self.xyz[:,mono_1,:]-self.xyz[:,mono_2])
            msd_50.append(s1-2*s2)

            s1,s2=_msd_fft(self.xyz[:,mono_3,:]-self.xyz[:,mono_4])
            msd_75.append(s1-2*s2)
            

        msd_25=np.mean(msd_25,axis=0)
        msd_50=np.mean(msd_50,axis=0)
        msd_75=np.mean(msd_75,axis=0)

        print('done!\n')
        
        return (msd_25,msd_50,msd_75)
        
    def compute_ModeAutocorr(self, modes=[1,2,4,7,10], chains=False):

        print("Computing Normal mode correlations ... ",flush=True, end=' ')
        Sp_all=[]
        for hh, chrm in enumerate(self.top):
            pol_xyz=self.xyz[:,chrm[0]:chrm[1]+1,:]
            N_pol=pol_xyz.shape[1]
        
            Sp=[]
            for ii,p in enumerate(modes):
                Xp_t = (1/N_pol)*np.einsum('ijk,j->ik',pol_xyz,
                                        np.cos((p*np.pi/N_pol)*(np.arange(1,N_pol+1,1)-0.5)))                
                Sp.append(_msd_fft(Xp_t)[1])
            Sp_all.append(Sp)
        
        SXp=np.array(Sp_all)
        
        if chains == False:
            SXp=np.mean(SXp, axis=0)
        
        print('done!\n',flush=True)

        return (SXp,modes)

    def compute_InterParticleDist(self, dr=1, Rmax=50):

        R"""
        Calculates the radial number density of monomers; which when integrated over 
        the volume (with the appropriate kernel: 4*pi*r^2) gives the total number of monomers.
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  

            dr (float, required):
                mesh size of radius for calculating the radial distribution. 
                can be arbitrarily small, but leads to empty bins for small values.
                bins are computed from the maximum values of radius and dr.
            
            ref (string):
                defines reference for centering the disribution. It can take three values:
                
                'origin': radial distance is calculated from the center

                'centroid' (default value): radial distributioin is computed from the centroid of the cloud of points at each time step

                'custom': user defined center of reference. 'center' is required to be specified when 'custom' reference is chosen

            center (list of float, len 3):
                defines the reference point in custom reference. required when ref='custom'
                       
        Returns:
            num_density:class:`numpy.ndarray`:
                the number density
            
            bins:class:`numpy.ndarray`:
                bins corresponding to the number density

        """
        print('Computing inter-particle distance distribution ...',flush=True,)

        bins=np.arange(0,Rmax,dr)
        dist_all=np.zeros(shape=(len(bins)-1))
        ii=0
        for kk,pos in enumerate(self.xyz):
            if kk%100==0:
                dist=distance.cdist(pos,pos, 'euclidean')
                rij_hist,bin_edges=np.histogram(np.ravel(dist), bins=bins, density=True)
                dist_all+=rij_hist
                ii+=1
                if kk%20000==0: print('frame: ',kk, flush=True)
        # dist_all=np.ravel(dist_all)    
        # rij_hist,bin_edges=np.histogram(dist_all, bins=bins, density=True)

        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        print('done!\n', flush=True)
        return (dist_all/ii, bin_mids)

    def compute_VoronoiCellVol(self, dv=1.0, Vmax=300.0):

        print('Computing Voronoi cell volumes ...',flush=True,)
        bins=np.arange(0,Vmax,dv)
        vol_avg=np.zeros(shape=(len(bins)-1))
        norm=0
        for ii, pos in enumerate(self.xyz):
            if ii%500!=0: continue
            if ii%20000: print('frame ', ii, flush=True)
            v = Voronoi(pos)
            vol = []
            for reg_num in v.point_region:
                indices = v.regions[reg_num]
                if -1 in indices: # some regions can be opened
                    vol.append(np.nan)
                else:
                    vol.append(ConvexHull(v.vertices[indices]).volume)
        
            hist,bin_edges=np.histogram(np.array(vol)[~np.isnan(vol)],bins=bins, density=True)
            norm+=1
            vol_avg+=hist
        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        print('done!\n')

        return (vol_avg/norm, bin_mids)
        

    def compute_RadNumDens(self, dr=1.0, ref='origin',center=None):

        R"""
        Calculates the radial number density of monomers; which when integrated over 
        the volume (with the appropriate kernel: 4*pi*r^2) gives the total number of monomers.
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  

            dr (float, required):
                mesh size of radius for calculating the radial distribution. 
                can be arbitrarily small, but leads to empty bins for small values.
                bins are computed from the maximum values of radius and dr.
            
            ref (string):
                defines reference for centering the disribution. It can take three values:
                
                'origin': radial distance is calculated from the center

                'centroid' (default value): radial distributioin is computed from the centroid of the cloud of points at each time step

                'custom': user defined center of reference. 'center' is required to be specified when 'custom' reference is chosen

            center (list of float, len 3):
                defines the reference point in custom reference. required when ref='custom'
                       
        Returns:
            num_density:class:`numpy.ndarray`:
                the number density
            
            bins:class:`numpy.ndarray`:
                bins corresponding to the number density

        """
        print('Computing radial number density with reference={} ...'.format(ref),flush=True, end=' ')

        if ref=='origin':
            rad_vals = np.ravel(np.linalg.norm(self.xyz,axis=2))

        elif ref=='centroid':
            rcm=np.mean(self.xyz,axis=1, keepdims=True)
            rad_vals = np.ravel(np.linalg.norm(self.xyz-rcm,axis=2))

        elif ref == 'custom':
            try:
                if len(center)!=3: raise TypeError
                center=np.array(center,dtype=float)
                center_nd=np.tile(center,(self.xyz.shape[0],1,1))
                rad_vals=np.ravel(np.linalg.norm(self.xyz-center_nd,axis=2))


            except (TypeError,ValueError):
                print("FATAL ERROR!!\n Invalid 'center' for ref='custom'.\n\
                        Please provide a valid center: [x0,y0,z0]")
                return ([0],[0])
        else:
            print("FATAL ERROR!! Unvalid 'ref'\n\
                'ref' can take one of three values: 'origin', 'centroid', and 'custom'")
            return ([0],[0])
        
        rdp_hist,bin_edges=np.histogram(rad_vals, bins=np.arange(0,rad_vals.max()+1,dr), density=False)
        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        bin_vols = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        num_density = rdp_hist/(self.xyz.shape[0]*bin_vols)
        print('done!\n', flush=True)
        return (num_density, bin_mids)

    def compute_comRadNumDens(self, dr=1.0, ref='origin',center=None):

        R"""
        Calculates the radial number density of center of mass of chromosome chains; which when integrated over 
        the volume (with the appropriate kernel: 4*pi*r^2) gives the total number of chromosomes.
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the :code: `xyz()` function.  

            dr (float, required):
                mesh size of radius for calculating the radial distribution. 
                can be arbitrarily small, but leads to empty bins for small values.
                bins are computed from the maximum values of radius and dr.
            
            ref (string):
                defines reference for centering the disribution. It can take three values:
                
                'origin': radial distance is calculated from the center

                'centroid' (default value): radial distributioin is computed from the centroid of the cloud of points at each time step

                'custom': user defined center of reference. 'center' is required to be specified when 'custom' reference is chosen

            center (list of float, len 3):
                defines the reference point in custom reference. required when ref='custom'
                       
        Returns:
            num_density:class:`numpy.ndarray`:
                the number density
            
            bins:class:`numpy.ndarray`:
                bins corresponding to the number density

        """
        print('Computing radial number density with reference={} ...'.format(ref),flush=True, end=' ')
        rad_vals=[]
        for chrm in self.top:
            rcms=np.mean(self.xyz[:,chrm[0]:chrm[1]+1,:],axis=1, keepdims=True)
            rad_vals.append(np.linalg.norm(rcms,axis=2))
        rad_vals=np.ravel(rad_vals)
        
        rdp_hist,bin_edges=np.histogram(rad_vals, bins=np.arange(0,rad_vals.max()+1,dr), density=False)
        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        bin_vols = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        num_density = rdp_hist/(self.xyz.shape[0]*bin_vols)
        print('done!\n', flush=True)
        return (num_density, bin_mids)


    def compute_BondLenDist(self,dx=0.1):
        print('Computing bond length distribution ... ', flush=True, end=' ')

        bondlen_vals=[]
        for chrm in self.top:
            bondlen_vals.append(np.linalg.norm(self.xyz[:,chrm[0]:chrm[1],:] - self.xyz[:,chrm[0]+1:chrm[1]+1,:], axis=2))

        bondlen_vals=np.ravel(bondlen_vals)
        bins=np.arange(0,bondlen_vals.max()+1,dx)
        bondlen_hist, bin_edges=np.histogram(bondlen_vals, bins=bins, density=True)
        bin_mids=0.5*(bin_edges[1:]+bin_edges[:-1])
        print('done!\n', flush=True)
        return (bondlen_hist, bin_mids)


    def compute_HiC(self, mu=3.0, rc=1.5, avg_all=True):
        print('Computing probability of contact versus contour distance')
        if avg_all:
            def calc_prob(data, mu, rc):
                return 0.5 * (1.0 + np.tanh(mu * (rc - distance.cdist(data, data, 'euclidean'))))
            size=self.top[0][1] - self.top[0][0]+1
            Prob = np.zeros((size, size))
            for chrm in self.top:
                for i in range(self.T):
                    Prob += calc_prob(self.xyz[i,chrm[0]:chrm[1]+1,:], mu, rc)
                    if i % 50000 == 0:
                        print("Reading frame {:} of {:}".format(i, len(self.xyz)))

            Prob=Prob/(self.T*self.top.shape[0])

            return Prob


    def traj2HiC(self, mu=3.22, rc = 1.78):
            R"""
            Calculates the *in silico* Hi-C maps (contact probability matrix) using a chromatin dyamics trajectory.   
            
            The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
            
            Args:

                mu (float, required):
                    Parameter in the probability of crosslink function. (Default value = 3.22).
                rc (float, required):
                    Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
            
            Returns:
                :math:`(N, N)` :class:`numpy.ndarray`:
                    Returns the *in silico* Hi-C maps (contact probability matrix).
            """
            def calc_prob(data, mu, rc):
                return 0.5 * (1.0 + np.tanh(mu * (rc - distance.cdist(data, data, 'euclidean'))))
            print('Computing HiC ...', flush=True)
            size = len(self.xyz[0])
            P = np.zeros((size, size))
            Ntotal = 0

            for i in range(len(self.xyz)):
                data = self.xyz[i]
                P += calc_prob(data, mu, rc)
                Ntotal += 1
                if i % 500 == 0:
                    print("Reading frame {:} of {:}".format(i, len(self.xyz)))

            return(np.divide(P , Ntotal))
