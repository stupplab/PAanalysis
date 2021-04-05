
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
            points_W = np.copy(positions_W[frame])
            points_W -= [Lx/2,Ly/2,Lz/2]
            query_points = np.copy(positions[frame,:,atom_index])
            query_points -= [Lx/2,Ly/2,Lz/2]
            neighborhood = freud.locality.LinkCell(box, points_W, cell_width=radius)
            neighbor_pairs = neighborhood.query(query_points, query_args).toNeighborList()
            num_water += [ len(neighbor_pairs) / nmol ]
            global_water_density += [nmol_W / (Lx*Ly*Lz)]
        
        hydration += [ np.mean((np.array(num_water)/(4/3*np.pi*radius**3)) / np.array(global_water_density)) ]
        # hydration += [ np.mean(num_water) *4/(4/3*np.pi*radius**3)] # convert to #H2O / nm^3

    return res_names, np.array(hydration), np.mean(global_water_density)






def hydration_profile_atomistic(grofile, trajfile, radius, frame_iterator, num_atoms, num_molecules, start_index):
    """
    Residue-wise hydration profile for atomistic simulations (tested for GROMACS simulations)
    num_atoms, num_molecules, start_index correspond to the molecule of interest.
    Assumes that all the identical molecules are consecutive
    Averages over only the backbone atoms vs all atoms
    Alkyl residue name should be added in this code. Currently only C16 and 12C
    """

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz


    # Get water indices
    W_indices = []
    for residue in traj.top.residues:
        if residue.is_water:
            W_indices += [residue.atom('O').index]
    nmol_W = len(W_indices)
    
    # Get atom_indices for N CA C O for all residues
    atom_indices = []
    start_res_id = traj.top.atom(start_index).residue.index
    end_res_id   = traj.top.atom( start_index + num_atoms*num_molecules - 1 ).residue.index
    num_residues_permol = (end_res_id - start_res_id + 1) / num_molecules
    if num_residues_permol % 1 != 0:
        raise ValueError('Number of residues per mol is calculated as a non-integer')
    else:
        num_residues_permol = int(num_residues_permol)    
    
    res_names = []
    for i in range(start_res_id, start_res_id+num_residues_permol):
        res_names += [ traj.top.residue(i).name ]

    # Caluclate atom_indices for each residue
    for j in range(start_res_id, end_res_id+1):
        res = traj.top.residue(j)
        atom_indices_ = [ a.index for a in res.atoms ]
        atom_indices += [atom_indices_]

    
    # Place water into grid cells using freud and generate the neighborhood for each frame
    query_args = dict(mode='ball', r_max=radius, exclude_ii=False)
    global_water_density = []
    neighborhoods = []
    neighbor_pairs = []
    for frame in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[frame]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points_W = np.copy(positions[frame, W_indices])
        points_W -= [Lx/2,Ly/2,Lz/2]
        neighborhood = freud.locality.LinkCell(box, points_W, cell_width=radius)
        neighborhoods += [ neighborhood ]
        global_water_density += [nmol_W / (Lx*Ly*Lz)]
        


    # calculate hydration profile over residues of each molecule
    hydration  = []
    hydration_ = []
    n=0
    for atom_indices_ in atom_indices:
        n=n+1
        
        num_water = []
        for k,frame in enumerate(frame_iterator):
            query_points = np.copy(positions[frame,atom_indices_])
            query_points -= [Lx/2,Ly/2,Lz/2]
            neighborhood = neighborhoods[k]
            neighbor_pairs = neighborhood.query(query_points, query_args).toNeighborList()
            num_water += [ len(neighbor_pairs) / (num_molecules*len(atom_indices_)) ]

        
        hydration_ += [ np.mean((np.array(num_water)/(4/3*np.pi*radius**3)) / np.array(global_water_density)) ]

        if n == num_residues_permol:
            hydration += [ hydration_ ]
            hydration_ = []
            n = 0

    hydration = np.mean(hydration, axis=0)

    return res_names, hydration, np.mean(global_water_density)



