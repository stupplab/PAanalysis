
import numpy as np
import os
from .utils import *
from . import quaternion

import freud
import mdtraj



def res_res_separation(itpfiles, topfile, grofile, trajfile, radius, frame_iterator, whichitp=0):
    """For the given MARTINI system files, 
    For each residue on a PA, calculate it's average separation from the closest residue.
    
    separation is calculated as the average distance between a residue and its similar neighbors
    calculated over 6 closest neighbors for each frame in frame_iterator.

    Returns: residue names, residence time per residue

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
    
    nmol_W         = get_num_molecules(topfile, 'W')
    positions_W    = get_positions(grofile, trajfile, (start_index, start_index+nmol_W))
    
    # Get Lx, Ly, Lz
    traj = mdtraj.load(trajfile, top=grofile)


    # atom indices from itpfile
    PAM_indicess = []
    res_namess = []
    pep_indicess = []
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
        res_namess += [res_names]
        pep_indicess += [pep_indices]

    # selection for a particular itp <whichitp>
    positions = positionss[whichitp]
    nmol = nmols[whichitp]
    num_atoms = num_atomss[whichitp]
    positions = positions.reshape(-1,nmol,num_atoms,3)
    shape = positions.shape
    PAM_indices = PAM_indicess[whichitp]
    res_names = res_namess[whichitp]
    pep_indices = pep_indicess[whichitp]
    atom_indices = PAM_indices[::-1] + pep_indices
    

    # calculate res-res separation

    res_res_sep  = []
    query_args = dict(mode='nearest', num_neighbors=6, r_max=radius, exclude_ii=True)
    for atom_index in atom_indices:
        res_res_sep_  = []
        for frame in frame_iterator:
            Lx, Ly, Lz = traj.unitcell_lengths[frame]
            box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
            points = np.copy(positions[frame,:,atom_index])
            neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)
            neighbor_pairs = neighborhood.query(points, query_args).toNeighborList()[:]
            
            # unwrap points
            seps = points[neighbor_pairs[:,1]]-points[neighbor_pairs[:,0]]
            for i,sep in enumerate(seps):
                if sep[0]>Lx/2:
                    seps[i,0] -= Lx
                elif sep[0]<-Lx/2:
                    seps[i,0] += Lx
                if sep[1]>Ly/2:
                    seps[i,1] -= Ly
                elif sep[1]<-Ly/2:
                    seps[i,1] += Ly
                if sep[2]>Lz/2:
                    seps[i,2] -= Lz
                elif sep[2]<-Lz/2:
                    seps[i,2] += Lz

            res_res_sep_ += [ np.mean(np.linalg.norm(seps,axis=1)) ]
        
        res_res_sep += [ np.mean(res_res_sep_) ]

    return res_names, res_res_sep


