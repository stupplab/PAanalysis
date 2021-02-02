
import numpy as np
import os
from .utils import *
from . import quaternion

import freud
import mdtraj



def hydration_profile(itpfile, topfile, grofile, trajfile, radius, frame_iterator):
    """For the given MARTINI system files, 
    calculates the water density per MARTINI residue within the shell of radius <radius>
    relative to the global water density.
    
    Returns: residue names, hydration profile, global water density

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
    
    # Get Lx, Ly, Lz
    traj = md.load(trajfile, top=grofile)
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

    # calculate hydration profile
    global_water_density = nmol_W / (Lx*Ly*Lz)

    hydration  = []
    positions   -= [Lx/2,Ly/2,Lz/2]
    positions_W -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    query_args = dict(mode='ball', r_max=radius, exclude_ii=False)
    for atom_index in atom_indices:
        num_water = []
        for frame in frame_iterator:
            points_W = positions_W[frame]
            query_points = positions[frame,:,atom_index]
            neighborhood = freud.locality.LinkCell(box, points_W, cell_width=radius)
            neighbor_pairs = neighborhood.query(query_points, query_args).toNeighborList()
            num_water += [ len(neighbor_pairs) / nmol ]
        hydration += [ np.mean(num_water) /(4/3*np.pi*radius**3) / global_water_density ]
        # hydration += [ np.mean(num_water) *4/(4/3*np.pi*radius**3)] # convert to #H2O / nm^3


    return res_names, np.array(hydration), global_water_density

