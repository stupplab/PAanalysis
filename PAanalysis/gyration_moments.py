
import numpy as np
import os
from .utils import *
from . import quaternion

import scipy.linalg

import freud
import mdtraj



def gyration_moments(itpfiles, topfile, grofile, trajfile, frame_iterator):
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
    
    itpnames = []
    bb_bonds_permols = []
    num_atomss    = []
    nmols         = []

    for itpfile in itpfiles:
        itpname = os.path.basename(itpfile).strip('.itp')
        itpnames += [itpname]
        bb_bonds_permols += [get_backbone_bonds_permol(itpfile)]
        num_atomss    += [get_num_atoms(itpfile)]
        nmols         += [get_num_molecules(topfile, itpname)]
    
    
    start_index = 0
    num_atoms_ = 0
    positionss = []
    for i in range(len(itpfiles)):
        num_atoms_ += num_atomss[i]*nmols[i]
        positionss += [get_positions(grofile, trajfile, (start_index, num_atoms_))]
        start_index = num_atoms_
    # positions = positions.reshape(-1,nmol,num_atoms,3)
    # shape = positions.shape
    nmol_W         = get_num_molecules(topfile, 'W')
    positions_W    = get_positions(grofile, trajfile, (start_index, start_index+nmol_W))
    
    
    # Get Lx, Ly, Lz
    traj = mdtraj.load(trajfile, top=grofile)
    Lx, Ly, Lz = traj.unitcell_lengths[-1]
    box = dict(Lx=Lx, Ly=Ly, Lz=Lz)

    # atom indices from itpfile
    PAM_indicess = []
    for itpfile in itpfiles:
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
        
        PAM_indicess += [PAM_indices]



    L1 = []
    L2 = []
    L3 = []
    maxcluster_sizes = []
    positionss = [positions - [Lx/2,Ly/2,Lz/2] for positions in positionss]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    # query_args = dict(mode='nearest', num_neighbors=1, r_max=radius, exclude_ii=True)
    for frame in frame_iterator:
        points = np.empty((0,3))
        for i,positions in enumerate(positionss):
            positions_ = positions.reshape(-1,nmols[i],num_atomss[i],3)
            positions_ = positions_[frame,:,PAM_indicess[i]].reshape(-1,3)
            points = np.append(points,positions_, axis=0)
            
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
    
    maxcluster_size = np.mean(maxcluster_sizes) / 4  # number of molecules in max cluster

    return maxcluster_size, L1_avg, L2_avg, L3_avg


