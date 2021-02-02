
import numpy as np
import os
from .utils import *
from . import quaternion

import freud
import mdtraj




def residence_time(itpfile, topfile, grofile, trajfile, radius, frame_iterator):
    """For the given MARTINI system files, 
    calculates the residence time (RT) of each residue around itself. 
    
    RT is calculated as the average time taken by the closest neighbor to 
    travel <radius> distance away from the residue.
    For instance, RT of an ALA residue at ith location is time taken 
    by the closest ALA residue at ith location to travel <radius> distance away.
    
    RT unit depends on the frame_iterator.
    If frame_iterator contains frames after each nanosecond, then RT is in nanoseconds.

    Returns: residue names, residence time per residue

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



    # Calculate RT
    RT = []
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    query_args = dict(mode='nearest', num_neighbors=1, r_max=radius, exclude_ii=True)
    for atom_index in atom_indices:
        pairs_sampled = [] # list of all [res1, res2, startframe]
        RT_ = []
        closest_neigh_dist_ = []
        for frame in frame_iterator:
            points = positions[frame,:,atom_index]
            neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)
            neighbor_pairs = np.array(neighborhood.query(points, query_args).toNeighborList())
            
            # Remove pairs that are in cross-periodic images
            sep = np.linalg.norm(points[neighbor_pairs[:,1]]-points[neighbor_pairs[:,0]], axis=1)
            neighbor_pairs = np.compress(sep<min([Lx,Ly,Lz])/2, neighbor_pairs, axis=0)
            sep = np.compress(sep<min([Lx,Ly,Lz])/2, sep, axis=0)
            
            # Track these neighbor pairs to caluclate RT
            args = list(range(len(neighbor_pairs)))
            for frame2 in range(frame+1, num_frames):
                if args == []:
                    break

                points2 = positions[frame2,:,atom_index]
                sep2 = np.linalg.norm(points[neighbor_pairs[:,0]] - points2[neighbor_pairs[:,1]], axis=1)

                # remove pairs with neighbor in different box images
                args = np.compress(sep2[args] < min([Lx,Ly,Lz])/2, args, axis=0)

                # remove neighbors that moved away radius distance
                # add the time to RT
                condition = sep2[args]<radius+sep[args]
                num_pairs_separated_this_time = len(args)-np.sum(condition)
                RT_ += [(frame2-frame)]*num_pairs_separated_this_time
                args = np.compress(condition, args, axis=0)
                # args_ = np.compress(sep2[args]>=radius, args, axis=0)

        RT += [np.mean(RT_)]

    return res_names, np.array(RT)
    

