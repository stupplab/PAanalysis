
import numpy as np
import os
from .utils import *
from . import quaternion

import scipy.linalg

import freud
import mdtraj



def gyration_moments(itpfiles, topfile, grofile, trajfile, frame_iterator, residue_indices=[]):
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

    residue_indices: list of lists containing residues indices for each molecule for clustering.
    len(residue_indices)==len(itpfiles) or residue_indices=[] (default: clusters all residues)
    For example, residue_indices=[[0,1,2,3],[]] clusters 1st four residues of the first molecule 
    and avoids second molecule.
    """
    

    if (residue_indices!=[]) and (len(residue_indices)!=len(itpfiles)):
        raise ValueError('len(residue_indices) must be equal to len(itpfiles)')

    
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


    if residue_indices==[]:
        for i in range(len(itpfiles)):
            residue_indices += [range(num_atomss[i])]



    L1 = []
    L2 = []
    L3 = []
    maxcluster_sizes = []
    
    # query_args = dict(mode='nearest', num_neighbors=1, r_max=radius, exclude_ii=True)
    for frame in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[frame]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        points = np.empty((0,3))
        for i,positions in enumerate(positionss):
            positions_ = positions.reshape(-1,nmols[i],num_atomss[i],3)
            positions_ = positions_[frame,:,residue_indices[i]].reshape(-1,3)
            points = np.append(points,positions_, axis=0)
        
        points -= [Lx/2,Ly/2,Lz/2]

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
    
    maxcluster_size = np.mean(maxcluster_sizes) # number of residues in max cluster

    return maxcluster_size, L1_avg, L2_avg, L3_avg


