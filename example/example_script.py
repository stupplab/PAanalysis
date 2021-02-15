"""
Example calculating some of the quantities using PAanalysis
"""


import mdtraj

# No need to do this if PAanalysis is already installed
import sys
sys.path.insert(1, '../')

import PAanalysis


itpfile  = 'PA.itp'
topfile  = 'PA.top'
grofile  = 'PA_water_min.gro'
trajfile = 'PA_water_eq.xtc'  # xtc is used here because it is smaller in size than trr


traj = mdtraj.load(trajfile, top=grofile)
print('Number of frames in PA_water_eq.xtc: ', len(traj))


frame_iterator = range(300,500,10)
max_cluster_size, L1, L2, L3 = PAanalysis.gyration_moments([itpfile], topfile, grofile, trajfile, frame_iterator)
print('\n----------------- Using PAanalysis.gyration_moments -----------------')
print('\n Number of molecules in the largest cluster: ', max_cluster_size)
print('\n Eigenvalues of gyration tensor: ', L1, L2, L3)
print('\n Aspericity (L1-0.5*(L2+L3)): ', L1-0.5*(L2+L3))
print('\n Acylindricity (L2-L3): ', (L2+L3))


frame_iterator = range(300,500,10)
radius = 0.8 # nm
res_names, hydration, global_water_density = PAanalysis.hydration_profile([itpfile], topfile, grofile, trajfile, radius, frame_iterator)
print('\n----------------- Using PAanalysis.hydration_profile -----------------')
print('\n Global water density (Each MARTINI water lumps 4 H2O): ', global_water_density)
print('\n Hydration per residue: ', list(zip(res_names, hydration)))


frame_iterator = range(300,500,10)
radius = 0.8 # nm
res_names, residence_time = PAanalysis.residence_time(itpfile, topfile, grofile, trajfile, radius, frame_iterator)
print('\n----------------- Using PAanalysis.residence_time -----------------')
print('\n Residence time per residue: ', list(zip(res_names, residence_time)))
