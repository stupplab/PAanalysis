
import numpy as np
import os
from .utils import *
from . import quaternion

import freud
import mdtraj



def hydration_profile(itpfiles, topfile, grofile, trajfile, radius, frame_iterator, whichitp=0):
    """For the given MARTINI system files, 
    calculates the water density per MARTINI residue within the shell of radius <radius>
    relative to the global water density.
    
    Returns: residue names, hydration profile, global water density

    trajfile: <.trr/.xtc>
    itpfiles should be in order of how the corresponding molecules appear in the grofile
    whichitp is the index of the molecules itp in <itpfiles> that will be used to calculate hydration on
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
    

    # calculate hydration profile
    
    
    hydration  = []
    query_args = dict(mode='ball', r_max=radius, exclude_ii=False)
    for atom_index in atom_indices:
        num_water = []
        global_water_density=[]
        for frame in frame_iterator:
            Lx, Ly, Lz = traj.unitcell_lengths[frame]
            box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
            points_W = positions_W[frame]
            points_W -= [Lx/2,Ly/2,Lz/2]
            query_points = positions[frame,:,atom_index]
            query_points -= [Lx/2,Ly/2,Lz/2]
            neighborhood = freud.locality.LinkCell(box, points_W, cell_width=radius)
            neighbor_pairs = neighborhood.query(query_points, query_args).toNeighborList()
            num_water += [ len(neighbor_pairs) / nmol ]
            global_water_density += [nmol_W / (Lx*Ly*Lz)]

        hydration += [ np.mean((np.array(num_water)/(4/3*np.pi*radius**3)) / np.array(global_water_density)) ]
        # hydration += [ np.mean(num_water) *4/(4/3*np.pi*radius**3)] # convert to #H2O / nm^3


    return res_names, np.array(hydration), global_water_density

