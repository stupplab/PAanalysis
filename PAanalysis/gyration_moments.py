
import numpy as np
import os
from .utils import *
from . import quaternion

import scipy.linalg

import freud
import mdtraj



def gyration_moments(itpfile, topfile, grofile, trajfile, frame_iterator):
    """For the given MARTINI system files, 
    calculates the principal compoments (eigenvalues) of the largest alkyl tail cluster.
    
    Largest cluster is calculated using Freud software | https://freud.readthedocs.io/

    Returns: maximum cluster size in number of molecules, L1, L2, L3
    
    L1>=L2>=L3 are eigenvalues of gyration tensor - G
    Gij = 1/N SUM_k[1,N] (ri^k - <ri>)*(rj^k - <rj>)

    Gyration moments can be used to calculate shape descriptors such as
    
    Asphericity = L1 - (L2+L3)/2
    Acylindricity = L2 - L3

    trajfile: <.trr/.xtc>
    """
    

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trajfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)
    shape = positions.shape
    nmol_W         = get_num_molecules(topfile, 'W')
    positions_W    = get_positions(grofile, trajfile, (num_atoms*nmol, num_atoms*nmol+nmol_W))
    
    num_frames      = positions.shape[0]

    # Get Lx, Ly, Lz
    traj = mdtraj.load(trajfile, top=grofile)
    Lx, Ly, Lz = traj.unitcell_lengths[-1]
    box = dict(Lx=Lx, Ly=Ly, Lz=Lz)

    # atom indices from itpfile
    pep_indices = []
    PAM_indices = []
    res_names = []
    start = False
    with open(itpfile, 'r') as f:
        for line in f:
            if '[ atoms ]' in line:
                start = True
                continue
            if start:
                words = line.split()
                if words == []:
                    break
                if words[3] == 'PAM':
                    PAM_indices += [int(words[0])-1]
                else:
                    pep_indices += [int(words[0])-1]
                    res_names += [words[3]+'\n'+words[4]]

    # reverse indices of PAM
    atom_indices = PAM_indices[::-1] + pep_indices
    res_names = ['PAM']*len(PAM_indices) + res_names
    

    L1 = []
    L2 = []
    L3 = []
    maxcluster_sizes = []
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    # query_args = dict(mode='nearest', num_neighbors=1, r_max=radius, exclude_ii=True)
    for frame in frame_iterator:
        points = positions[frame,:,:len(PAM_indices)].reshape(-1,3)
        
        # cluster
        cl = freud.cluster.Cluster()
        cl.compute((box, points), neighbors={'r_max': 0.8})
        sizes = [len(c) for c in cl.cluster_keys]
        argmax = np.argmax(sizes)
        
        points_largestcluster = points[cl.cluster_keys[argmax]]
        
        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute((box, points_largestcluster), [0]*sizes[argmax])
        G = cl_props.gyrations[0]

        L3_, L2_, L1_ = np.sort(scipy.linalg.eigvalsh(G))

        L1 += [L1_]
        L2 += [L2_]
        L3 += [L3_]
        maxcluster_sizes += [sizes[argmax]]
        

    L1_avg = np.mean(L1)
    L2_avg = np.mean(L2)
    L3_avg = np.mean(L3)
    
    maxcluster_size = np.mean(maxcluster_sizes) / len(PAM_indices)  # number of molecules in max cluster

    return maxcluster_size, L1_avg, L2_avg, L3_avg


