import sys
  
sys.path.append('../')

import ActivePolymer
import AnalyzeTrajectory
import numpy as np

class testActivePolymer():

    def runDefault(self):
        sim=ActivePolymer.ActivePolymer(name='test', time_step=0.001, collision_rate=0.1, temperature=200, 
        active_corr_time=100.0, activity_amplitude=1.0,
        outpath='test_out/', init_struct=None, seq_file=100,)

        ActivePolymer.addRadialConfinement(sim, R0=10, method='FlatBottomHarmonic', kr=50)
        ActivePolymer.addHarmonicBonds(sim,top_file='../input_files/chr_top.txt', kb=10.0, d=1.0)
        ActivePolymer.addSelfAvoidance(sim, E0=4, method='exp')
        # ActivePolymer.addCustomTypes(sim,mu=3.,rc=2.,TypesTable='../input_files/type_table.csv')

        ActivePolymer.runSims(sim, nblocks=200, blocksize=100, )

    def runAnalyze(self):
        traj=AnalyzeTrajectory.AnalyzeTrajectory(datapath='test_out/', datafile=None, 
                            top_file='../input_files/chr_top.txt', discard_init_steps=0)
        print(traj.savename)
        savename='test_out/'+traj.savename

        hic=traj.traj2HiC(mu=3, rc=1.5)

        np.save(savename+'hic.npy', hic)

        print('Finished')


run=testActivePolymer()
# run.runDefault()
run.runAnalyze()
