
import numpy as np
import itertools
import os, subprocess
from . import utils
from .utils import *
from . import quaternion

import freud
import mdtraj
import time




def r_C_C(grofile, trajfile, frame_iterator, topfile, molname, residuenames=None, filenamerdf=None, filenameft=None):
    """Calculates avearge distance between O (C=O) and H (NH) and H (HOH)
    r_CO_CO

    r is calculated af the average distance of O (C=O) from the nearest H (NH / HOH)
    """

    #---------------------------------------------------------------------------------

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_atoms = traj.n_atoms
    
    num_molecules = utils.get_num_molecules_fromtop(topfile, molname)

    num_atoms_permol = utils.get_num_atoms_fromtop(topfile, molname)

    radius = 2 # rvdw and rcoulomb

    #---------------------------------------------------------------------------------

    # Get atom indices of C,O,N,H(NH) from grofile
    C_indices = []
    O_indices = []
    N_indices = []
    H_indices = []
    molids_C = np.empty(0, dtype=int)
    with open(grofile, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if ' 1 ' in line:
            line_start = i
            break
    for i,line in enumerate(lines):
        if ' C ' in line:
            C_indices += [i-line_start]
            molids_C = np.append(molids_C, int( i // num_atoms_permol ) )
        elif ' O ' in line:
            O_indices += [i-line_start]
        elif ' N ' in line:
            N_indices += [i-line_start]
        elif ' HN ' in line:
            H_indices += [i-line_start]
    indices = dict( 
        C = np.array(C_indices), 
        O = np.array(O_indices),
        N = np.array(N_indices),
        H = np.array(H_indices) )
    
    
    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
                residue_indices += [atom.index]
        
        args = list(set(residue_indices) & set(indices['C']))
    else:
        args = indices['C']
    

    # Get water indices using mdtraj
    HOH_indices = []
    for residue in traj.top.residues:
        if residue.is_water:
            HOH_indices += [residue.atom('O').index, residue.atom('O').index+1, residue.atom('O').index+2]
    indices['HOH'] = np.array(HOH_indices)

    #---------------------------------------------------------------------------------

    query_args = {}
    query_args['C'] = dict(mode='nearest', num_neighbors=8, r_max=radius, exclude_ii=True)

    r_C_C = np.empty(0)

    for frame in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[frame]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        
        points = positions[frame, args]
        if len(points) == 0:
            continue
        neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)

        query_points = positions[frame, args]
        neighbor_pairs = np.array(
            neighborhood.query(
                query_points, query_args['C']).toNeighborList(
                sort_by_distance=True)[:])
        
        # Exclude neighbor pairs that are in the same mol and choose the nearest one
        pairs_ = []
        n=-1
        for pair in neighbor_pairs:
            if molids_C[ pair[0] ] == molids_C[ pair[1] ]:
                continue
            if n < pair[0]:
                pairs_ += [pair]
                n = pair[0]
        neighbor_pairs = np.array(pairs_)
        
        unwrapped_points = utils.unwrap_points(points[neighbor_pairs[:,1]], query_points[neighbor_pairs[:,0]], Lx, Ly, Lz)
        r = unwrapped_points - query_points[neighbor_pairs[:,0]]
        r_norm = np.linalg.norm(r, axis=-1)

        r_C_C = np.append(r_C_C, r_norm, axis=0)

    if len(r_C_C)!=0:
        r_C_C_mean = np.mean(r_C_C)
        r_C_C_std = np.std(r_C_C)
    else:
       r_C_C_mean = None
       r_C_C_std = None
    if r_C_C_mean == 0:
        r_C_C_mean = None
        r_C_C_std = None

    return np.array([r_C_C_mean, r_C_C_std])

    # #---------------------------------------------------------------------------------

    # # r_C_C
    
    # bins = 50
    # r_min = 0.1
    # r_max = 2
    # num_CO_permol = int(np.sum(molids == 0))
    # rdf=[]
    # for i in range(num_CO_permol):
    #     rdf += [ freud.density.RDF(bins=bins, r_min=r_min, r_max=r_max) ]
    

    # for frame in frame_iterator:
    #     Lx, Ly, Lz = traj.unitcell_lengths[frame]
    #     box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

    #     for i in range(num_CO_permol):
    #         args = indices['C'][i::num_CO_permol]
    #         points = positions[frame, args]
    #         rdf[i].compute(system=(box, points), reset=False)

    # r_C_C = []
    # rdf_mean = 0
    # for i in range(num_CO_permol):
    #     r_C_C += [ np.sum(rdf[i].rdf*rdf[i].bin_centers) / np.sum(rdf[i].rdf) ]
    #     rdf_mean += rdf[i].rdf
    # rdf_mean /= num_CO_permol



    # # Plot mean rdf
    # if type(filenamerdf) != type(None):
        
    #     import matplotlib.pyplot as plt

    #     x = rdf[0].bin_centers
    #     y = rdf_mean

    #     fig = plt.figure(figsize=(4/1.2,3/1.2))
    #     plt.plot(x, y, marker='')
    #     plt.title('RDF of C (C=O)')
    #     plt.xlabel('Radius')
    #     plt.ylabel('RDF')
    #     plt.subplots_adjust(bottom=0.18, left=0.16)
    #     plt.savefig(filenamerdf, dpi=400)


    # #---------------------------------------------------------------------------------


    # # Fourier transform of positions C
    # # freq_range = (0.1, 2*np.pi/.1)
    # freq_range = (0, 20)
    # x = np.linspace(freq_range[0], freq_range[1], 40)
    # y = np.linspace(freq_range[0], freq_range[1], 40)
    # z = np.linspace(freq_range[0], freq_range[1], 40)
    # q_vectors = np.array(list(itertools.product(x,y,z)))
    # q_norm = np.linalg.norm(q_vectors, axis=-1)

    # num_CO_permol = int(np.sum(molids == 0))

    # bins = np.linspace(freq_range[0],20,200)
    # indices_q = np.digitize(q_norm, bins)-1 # -1 shifts the indices from left to right edge 

    # Sq_binned = []
    # for n in range(num_CO_permol):
        
    #     args = indices['C'][n::num_CO_permol]
        
    #     Sq_frame = []
    #     for frame in frame_iterator:
    #         R = positions[frame, args]
    #         ft = np.sum(np.exp(-1j*R.dot(q_vectors.T)), axis=0)
    #         Sq_frame += [np.abs(ft)/R.shape[0]]
    #     Sq = np.mean(Sq_frame, axis=0)
        

    #     Sq_binned_ = np.zeros(len(bins))
    #     num_i = np.zeros(len(bins))
    #     for i in indices_q:
    #         Sq_binned_[i] += Sq[i]
    #         num_i[i] += 1
    #     Sq_binned_ /= num_i
        
    #     Sq_binned += [Sq_binned_]
    
    
    # Sq_binned = np.mean(Sq_binned, axis=0)

    
    
    # # Plot FT
    # if type(filenameft) != type(None):
        
    #     import matplotlib.pyplot as plt

    #     x = bins
    #     y = Sq_binned

    #     fig = plt.figure(figsize=(4/1.2,3/1.2))
    #     plt.plot(x, y, marker='')
    #     plt.title('FT of C (C=O)')
    #     plt.xlabel(r'2$\pi$/$\lambda$')
    #     plt.ylabel('FT')
    #     plt.subplots_adjust(bottom=0.18, left=0.16)
    #     plt.savefig(filenameft, dpi=400)
        
        
    # #---------------------------------------------------------------------------------


    # return r_C_C, np.mean(r_C_C)



def r_O_H(grofile, trajfile, frame_iterator, topfile, molname, residuenames=None, filenamerdf=None, filenameft=None):
    """Calculates avearge distance between O (C=O) and H (NH) and H (HOH)
    r_CO_HN
    r_CO_HOH

    r is calculated af the average distance of O (C=O) from the nearest H (NH / HOH)
    """

    #---------------------------------------------------------------------------------

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_atoms = traj.n_atoms
    
    num_molecules = utils.get_num_molecules_fromtop(topfile, molname)

    num_atoms_permol = utils.get_num_atoms_fromtop(topfile, molname)


    radius = 2 # rvdw and rcoulomb

    #---------------------------------------------------------------------------------

    # Get atom indices of C,O,N,H(NH) from grofile
    C_indices = []
    O_indices = []
    N_indices = []
    H_indices = []
    molids_O = np.empty(0, dtype=int)
    molids_H = np.empty(0, dtype=int)
    with open(grofile, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if ' 1 ' in line:
            line_start = i
            break
    for i,line in enumerate(lines):
        if ' C ' in line:
            C_indices += [i-line_start]
        elif ' O ' in line:
            O_indices += [i-line_start]
            molids_O = np.append(molids_O, int( i // num_atoms_permol ) )
        elif ' N ' in line:
            N_indices += [i-line_start]
        elif ' HN ' in line:
            H_indices += [i-line_start]
            molids_H = np.append(molids_H, int( i // num_atoms_permol ) )
    indices = dict( 
        C = np.array(C_indices), 
        O = np.array(O_indices),
        N = np.array(N_indices),
        H = np.array(H_indices) )
    
    
    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
                residue_indices += [atom.index]
        
        indices['C'] = list(set(residue_indices) & set(indices['C']))
        indices['H'] = list(set(residue_indices) & set(indices['H']))
        indices['O'] = list(set(residue_indices) & set(indices['O']))
        indices['H'] = list(set(residue_indices) & set(indices['H']))
        
    # Get water indices using mdtraj
    HOH_indices = []
    for residue in traj.top.residues:
        if residue.is_water:
            HOH_indices += [residue.atom('O').index, residue.atom('O').index+1, residue.atom('O').index+2]
    indices['HOH'] = np.array(HOH_indices)


    #---------------------------------------------------------------------------------

    query_args = {}
    query_args['HOH'] = dict(mode='nearest', num_neighbors=1, r_max=radius, exclude_ii=False)
    query_args['NH'] = dict(mode='nearest',  num_neighbors=8, r_max=radius, exclude_ii=True)

    r_CO_NH = np.empty(0)
    r_CO_HOH = np.empty(0)

    for frame in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[frame]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        points = positions[frame, indices['H']]
        if len(points) == 0:
            continue
        neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)

        query_points = positions[frame, indices['O']]
        neighbor_pairs = np.array(
            neighborhood.query(
                query_points, query_args['NH']).toNeighborList(
                sort_by_distance=True)[:])

        
        # Exclude neighbor pairs that are in the same mol
        pairs_ = []
        n=-1
        for pair in neighbor_pairs:
            if molids_O[ pair[0] ] == molids_H[ pair[1] ]:
                continue
            if n < pair[0]:
                pairs_ += [pair]
                n = pair[0]
        neighbor_pairs = np.array(pairs_)

        unwrapped_points = utils.unwrap_points(points[neighbor_pairs[:,1]], query_points[neighbor_pairs[:,0]], Lx, Ly, Lz)
        r = unwrapped_points - query_points[neighbor_pairs[:,0]]
        r_norm = np.linalg.norm(r, axis=-1)
        

        r_CO_NH = np.append(r_CO_NH, r_norm, axis=0)


        points = positions[frame, indices['HOH']]
        neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)

        query_points = positions[frame, indices['O']]
        neighbor_pairs = np.array(
            neighborhood.query(
                query_points, query_args['HOH']).toNeighborList(
                sort_by_distance=True)[:])

        unwrapped_points = utils.unwrap_points(points[neighbor_pairs[:,1]], query_points[neighbor_pairs[:,0]], Lx, Ly, Lz)
        r = unwrapped_points - query_points[neighbor_pairs[:,0]]
        r_norm = np.linalg.norm(r, axis=-1)
        
        r_CO_HOH = np.append(r_CO_HOH, r_norm, axis=0)


    if len(r_CO_NH)!=0:
        r_CO_NH_mean = np.mean(r_CO_NH)
        r_CO_NH_std = np.std(r_CO_NH)
    else:
        r_CO_NH_mean = None
        r_CO_NH_std = None
    if len(r_CO_HOH)!=0:
        r_CO_HOH_mean = np.mean(r_CO_HOH)
        r_CO_HOH_std = np.std(r_CO_HOH)
    else:
        r_CO_HOH_mean = None
        r_CO_HOH_std = None

    if r_CO_NH_mean == 0:
        r_CO_NH_mean = None
        r_CO_NH_std = None
    if r_CO_HOH_mean == 0:
        r_CO_HOH_mean = None
        r_CO_HOH_std = None

    return np.array([r_CO_NH_mean, r_CO_NH_std, r_CO_HOH_mean, r_CO_HOH_std])

    # #---------------------------------------------------------------------------------

    # # r_CO_NH using rdf
    
    # bins = 50
    # r_min = 0.1
    # r_max = 2
    # num_CO_permol = int(np.sum(molids == 0))
    # rdf=[]
    # for i in range(num_CO_permol):
    #     rdf += [ freud.density.RDF(bins=bins, r_min=r_min, r_max=r_max) ]
    

    # for frame in frame_iterator:
    #     Lx, Ly, Lz = traj.unitcell_lengths[frame]
    #     box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

    #     for i in range(num_CO_permol):
    #         args = indices['H'][i::num_CO_permol]
    #         points = positions[frame, args]
    #         args = indices['O'][i::num_CO_permol]
    #         query_points = positions[frame, args]
    #         rdf[i].compute(system=(box, points), query_points=query_points, reset=False)

    # r_CO_NH = []
    # rdf_mean = 0
    # for i in range(num_CO_permol):
    #     r_CO_NH += [ np.sum(rdf[i].rdf*rdf[i].bin_centers) / np.sum(rdf[i].rdf) ]
    #     rdf_mean += rdf[i].rdf
    # rdf_mean /= num_CO_permol



    # # Plot mean rdf
    # if type(filenamerdf) != type(None):
        
    #     import matplotlib.pyplot as plt

    #     x = rdf[0].bin_centers
    #     y = rdf_mean

    #     fig = plt.figure(figsize=(4/1.2,3/1.2))
    #     plt.plot(x, y, marker='')
    #     plt.title('RDF C=O-NH')
    #     plt.xlabel('Radius')
    #     plt.ylabel('RDF')
    #     plt.subplots_adjust(bottom=0.18, left=0.16)
    #     plt.savefig(filenamerdf, dpi=400)


    # #---------------------------------------------------------------------------------

    # # Fourier transform of positions C
    # # freq_range = (0.1, 2*np.pi/.1)
    # freq_range = (0, 20)
    # x = np.linspace(freq_range[0], freq_range[1], 40)
    # y = np.linspace(freq_range[0], freq_range[1], 40)
    # z = np.linspace(freq_range[0], freq_range[1], 40)
    # q_vectors = np.array(list(itertools.product(x,y,z)))
    # q_norm = np.linalg.norm(q_vectors, axis=-1)

    # num_CO_permol = int(np.sum(molids == 0))

    # bins = np.linspace(freq_range[0],20,200)
    # indices_q = np.digitize(q_norm, bins)-1 # -1 shifts the indices from left to right edge 

    # Sq_binned = []
    # for n in range(num_CO_permol):
        
    #     args = np.append(indices['O'][n::num_CO_permol], indices['H'][n::num_CO_permol], axis=0)
        
    #     Sq_frame = []
    #     for frame in frame_iterator:
    #         R = positions[frame, args]
    #         ft = np.sum(np.exp(-1j*R.dot(q_vectors.T)), axis=0)
    #         Sq_frame += [np.abs(ft)/R.shape[0]]
    #     Sq = np.mean(Sq_frame, axis=0)


    #     Sq_binned_ = np.zeros(len(bins))
    #     num_i = np.zeros(len(bins))
    #     for i in indices_q:
    #         Sq_binned_[i] += Sq[i]
    #         num_i[i] += 1
    #     Sq_binned_ /= num_i
        
    #     Sq_binned += [Sq_binned_]
    
    
    # Sq_binned = np.mean(Sq_binned, axis=0)

    
    
    # # Plot FT
    # if type(filenameft) != type(None):
        
    #     import matplotlib.pyplot as plt

    #     x = bins
    #     y = Sq_binned

    #     fig = plt.figure(figsize=(4/1.2,3/1.2))
    #     plt.plot(x, y, marker='')
    #     plt.title('FT of O-H (C=O-NH)')
    #     plt.xlabel(r'2$\pi$/$\lambda$')
    #     plt.ylabel('FT')
    #     plt.subplots_adjust(bottom=0.18, left=0.16)
    #     plt.savefig(filenameft, dpi=400)

    
    # #---------------------------------------------------------------------------------


    # return r_CO_NH_mean, r_CO_HOH_mean, r_CO_NH, np.mean(r_CO_NH)




def electrostatic_potential(grofile, trajfile, topfile, molname, frame_iterator, rtpfiles):
    """

    Calculating electrostatic potential on C O N H due to HOH and O(C=O) H(NH)
    to correlate with frequency shift of the amide I peak
    https://doi.org/10.1063/1.1580807
    
    Tested using charmm36-jul2020.

    U_non-bonded = U_LJ + U_elec
    U_LJ_ij = 4*eps_ij * [ (sigma_ij/rij)^12 - (sigma_ij/rij)^6 ] ]
    F_LJ_ij = -4*eps_ij *[ 12*sigma_ij/rij^13 - 6*sigma_ij/rij^7 ]
    U_elec_ij = 1/4*pi*eps0 * qi*qj / (eps_r*rij) ]      1/4pieps0 = 138.935458   eps_r = 1 by default
    F_elec_ij_i = - 1/4*pi*eps0 * qi*qj / (eps_r*ri^2j) * unit_vec(n_ij)    | rij=rj-ri
    E_elect_ij_i = - 1/4*pi*eps0 * qj / (eps_r*ri^2j) * unit_vec(n_ij)
    More details given in: https://www.charmmtutorial.org/index.php/The_Energy_Function

    mdp params
    ; vdw
    vdwtype                 = cutoff
    vdw-modifier            = force-switch
    rvdw-switch             = 1.0
    rvdw                    = 1.2       ; short-range van der Waals cutoff (in nm)
    ; Electrostatics
    coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
    rcoulomb                = 1.2
    pme_order               = 4         ; cubic interpolation
    fourierspacing          = 0.16      ; grid spacing for FFT
    
    IONS used: [NA, CL]
    TERs used: [NH2]

    ASSUMES:
    - Continugous molecules in the grofile
    - only one type of protein chain
    """

    #---------------------------------------------------------------------------------

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_atoms = traj.n_atoms
    
    # modify script to add more ions and ters
    IONS = ['NA', 'CL']
    TERS = ['NH2']

    radius = 1.2 # rvdw and rcoulomb

    #---------------------------------------------------------------------------------

    # collect sigma and epsilon from ffnonbondedfile
    residues_unique = np.unique([res.name for res in traj.top.residues])
    
    atom_types = {} # per residue
    for rtpfile in rtpfiles:
        with open(rtpfile, 'r') as f:
            lines = f.readlines()
        start = False
        for i,line in enumerate(lines):
            words = line.split()
            if (len(words)==0) or (words[0]==';'):
                continue
            words_before = lines[i-1].split()
            
            if (not start) and ('[ atoms ]' in line) and (len(words_before)==3) and (words_before[1] in residues_unique):
                start = True
                residue = words_before[1]
                atom_types[residue] = {}
                continue

            if (line.split()[0]=='['):
                start = False
            
            if start:
                words = line.split()
                if words[0] == 'HN': # convert HNs to H as given in .arn files of the charmmff
                    words[0] = 'H'
                atom_types[residue][words[0]] = [words[1], float(words[2])]


    # BUG patch: mdtraj seems to use HB3 instead of HB1 for GLU
    # remove this patch if the bug is resolved
    if 'GLU' in atom_types.keys():
        atom_types['GLU']['HB3'] = atom_types['GLU']['HB1']
        atom_types['GLU']['HG3'] = atom_types['GLU']['HG1']



    # Ions - no van der waal interaction, names NA and CL are placeholder
    if 'NA' in residues_unique:
        atom_types['NA']={'NA': ['NA', 1]}
    if 'CL' in residues_unique:
        atom_types['CL']={'CL': ['CL', -1]}
    
    # TER atoms - values are manually copied from the modified charmm36-jul2020.ff
    atom_types['NH2'] = dict(NT = ['NH2', -0.62], HT1=['H', 0.30], HT2=['H', 0.32], H=['H', 0.30], H2=['H', 0.32])
    # atom_types['COOH'] = dict()
    # atom_types['COO-'] = dict()
    
    # protonation H, basically for GLU and ASP
    atom_types['proton'] = {'HE2': ['H', 0]}

    # water
    atom_types['HOH']['O'] = atom_types['HOH']['OW']
    atom_types['HOH']['H1'] = atom_types['HOH']['HW1']
    atom_types['HOH']['H2'] = atom_types['HOH']['HW2']

    #---------------------------------------------------------------------------------

    # Set index wise atom types and charges of all atoms in the system
    system_atom_types = []
    system_atom_charges = []
    for atom in traj.top.atoms:
        res = atom.residue.name
        try:
            atom_types[res][atom.name]
            system_atom_types += [atom_types[res][atom.name][0]]
            system_atom_charges += [atom_types[res][atom.name][1]]
        except KeyError: # check atom in TERs
            found = False
            for ter in TERS:
                if atom.name in atom_types[ter].keys():
                    system_atom_types += [atom_types[ter][atom.name][0]]
                    system_atom_charges += [atom_types[ter][atom.name][1]]
                    found = True
                    break
            if not found:
                if atom.name in atom_types['proton'].keys():
                    system_atom_types += [atom_types['proton'][atom.name][0]]
                    system_atom_charges += [atom_types['proton'][atom.name][1]]
                    found = True
            if not found:
                raise ValueError(f'{atom.name} index {atom.index} not found in {res}, TERS, water, proton or IONS')
    
    system_atom_types = np.array(system_atom_types)
    system_atom_charges = np.array(system_atom_charges)
    atom_types_unique = np.unique(system_atom_types)
    
    
    #---------------------------------------------------------------------------------
    

    # atom Get indices of C,O,N,H(NH) from grofile
    C_indices = []
    O_indices = []
    N_indices = []
    H_indices = []
    with open(grofile, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if ' 1 ' in line:
            line_start = i
            break
    for i,line in enumerate(lines):
        if ' C ' in line:
            C_indices += [i-line_start]
        elif ' O ' in line:
            O_indices += [i-line_start]
        elif ' N ' in line:
            N_indices += [i-line_start]
        elif ' HN ' in line:
            H_indices += [i-line_start]
    indices = dict( 
        C = np.array(C_indices), 
        O = np.array(O_indices),
        N = np.array(N_indices),
        H = np.array(H_indices) )
    
    
    # Get water indices using mdtraj
    HOH_indices = []
    for residue in traj.top.residues:
        if residue.is_water:
            HOH_indices += [residue.atom('O').index, residue.atom('O').index+1, residue.atom('O').index+2]
    indices['HOH'] = np.array(HOH_indices)

  
    #---------------------------------------------------------------------------------
    # Calculate Van der Waal force and electrostatic field is calculated for C, H, O, NH1
    # F_LJ_ij = -4*eps_ij *[ 12*sigma_ij/rij^13 - 6*sigma_ij/rij^7 * unit_vec(n_ij)
    # phi_elect_ij = - 1/4*pi*eps0 * qj / (eps_r*rij)
        
    # Electrostatic potential
    phi = dict(HOH={}, H={}, O={})
    for B in ['HOH', 'H', 'O']:
        phi[B] = dict(
                C = [],
                O = [],
                N = [],
                H = [] )


    X = dict(                 # Position
        C = np.empty((0,3)),
        O = np.empty((0,3)),
        N = np.empty((0,3)),
        H = np.empty((0,3)) )
        

    query_args = {}
    query_args['HOH'] = dict(mode='ball', r_max=radius, exclude_ii=False)
    query_args['H'] = dict(mode='ball', r_max=radius, exclude_ii=True)
    query_args['O'] = dict(mode='ball', r_max=radius, exclude_ii=True)
    for frame in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[frame]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        for B in ['HOH', 'H', 'O']:
            points = positions[frame, indices[B]]

            neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)
            
            for A in ['C','O','N','H']:
                query_points = positions[frame, indices[A]]
                neighbor_pairs = np.array(neighborhood.query(query_points, query_args[B]).toNeighborList()[:])

                unwrapped_points = utils.unwrap_points(points[neighbor_pairs[:,1]], query_points[neighbor_pairs[:,0]], Lx, Ly, Lz)
                r = unwrapped_points - query_points[neighbor_pairs[:,0]]
                r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
                r_unit = r / r_norm
                
                q = system_atom_charges[indices[B][neighbor_pairs[:,1]]].reshape(-1,1)
                
                eps_r = 1
                phi_coul = 138.935458 * q / (eps_r * r_norm)
                
                phi_ = np.zeros(len(query_points))
                for i,pair in enumerate(neighbor_pairs):
                    phi_[pair[0]] += phi_coul[i]
                
                phi[B][A] = np.append(phi[B][A], phi_, axis=0)

               
    phi_mean = dict(HOH={}, H={}, O={})
    phi_std = dict(HOH={}, H={}, O={})
    for B in ['HOH', 'H', 'O']:
        for A in ['C','O','N','H']:
            phi_mean[B][A] = np.round(np.mean(phi[B][A]), 2)
            phi_std[B][A] = np.round(np.std(phi[B][A]), 2)
    
    return phi_mean, phi_std




def CO_bond_length_properties(grofile, trajfile, frame_iterator, window_duration_autocorr, freq_range, window_duration_ft, autocorrfilename=None, ftfilename=None, gaussian_sigma=None):
    """
    Calculates 
    - Mean and variance of the CO bond length
    - Autocorrelation of the CO bond length and 
    its characteristic time = ( SUM_t |auto(t)| )
    - Fourier transform of the temporal graph of the CO bond length and 


    Autocorr and FT plots are plotted and saved if filenames are given
    gaussian_sigma: smoothing parameter for the FT plot 
    freq_range: frequency range for the fourier transform calculation

    frame_iterator: slices the traj in the beginnning 
    """

    traj = mdtraj.load(trajfile, top=grofile)
    traj = traj[frame_iterator]

    positions = traj.xyz

    num_frames = traj.n_frames


    # Get indices of C,O from grofile
    C_indices = []
    O_indices = []
    with open(grofile, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if ' 1 ' in line:
            line_start = i
            break
    for i,line in enumerate(lines):
        if ' C ' in line:
            C_indices += [i-line_start]
        elif ' O ' in line:
            O_indices += [i-line_start]
    C_indices = np.array(C_indices)
    O_indices = np.array(O_indices)

    r =[]
    for f in range(num_frames):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        posO = utils.unwrap_points(positions[f,O_indices], positions[f,C_indices], Lx, Ly, Lz)
        r += [ posO - positions[f,C_indices] ]
    r = np.array(r)
    CO_length = np.linalg.norm( r, axis=-1 )

    # print(CO_length[:800,0])
    # import matplotlib.pyplot as plt
    # plt.plot(range(800), CO_length[:800,0])
    # plt.show()
    # raise

    CO_mean = np.mean( CO_length )
    CO_std = np.std( CO_length )


    # Calculate Autocorrelation
    mu = np.mean(CO_length, axis=0, keepdims=True)
    CO_length_ = CO_length - mu
    sigma2 = np.mean( CO_length_**2, axis=0 )
    auto = [1]
    for i in range(1,window_duration_autocorr):
        auto_ = np.mean( CO_length_[:-i]*CO_length_[i:], axis=0 ) / sigma2
        auto += [ np.mean(auto_) ]
    auto = np.array(auto)

    
    tau = np.sum( np.abs(auto) )

    # Typical freq_range = [2*np.pi/100, 2*np.pi]
    # Fourier transform of autocorrelation
    w = np.linspace(freq_range[0], freq_range[1], 100, endpoint=False)
    t = np.arange(window_duration_ft)
    ft = []
    for w_ in w:
        ft += [ np.sum( auto[:window_duration_ft] * np.exp( -1j * w_ * t ), axis=0 )]
    ft = np.abs(ft)
    
    w_ft_peak = w[ np.argmax(ft) ]

    
    # Apply smoothening if gaussian_sigma is provided
    if type(gaussian_sigma) != type(None):

        from scipy.ndimage import gaussian_filter
        
        auto = gaussian_filter(auto, sigma=gaussian_sigma)
        ft = gaussian_filter(ft, sigma=gaussian_sigma)

        
    # Plots
    if type(autocorrfilename) != type(None):
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(range(len(auto)), auto, marker='')
        plt.title('Autocorrelation of C=O bond length')
        plt.xlabel('Frame')
        plt.ylabel('Autocorrelation')
        plt.subplots_adjust(bottom=0.18, left=0.22)
        plt.savefig(autocorrfilename, dpi=400)    

        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(range(len(CO_length[:,2])), CO_length[:,1], marker='')
        plt.title('Bond length evolution')
        plt.xlabel('Frame')
        plt.ylabel('Bond length')
        plt.subplots_adjust(bottom=0.18, left=0.22)
        plt.savefig('CO_bondlength_evolution.png', dpi=400)    
        

    if type(ftfilename) != type(None):
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(w, ft, marker='')
        plt.title('Fourier Transform of C=O bond length')
        plt.xlabel(r'2$\pi$/$\lambda$')
        plt.ylabel('FT')
        plt.subplots_adjust(bottom=0.18, left=0.22)
        plt.savefig(ftfilename, dpi=400)


    return CO_mean, CO_std, auto, tau, ft, w_ft_peak



def force_on_CO_bond(grofile, trajfile, forcetrajfile, window_duration, filename=None, autocorrfilename=None):
    """
    Calculates RMS force on the CO bond in the direction of the CO bond. 
    Uses net atomic forces generated by the gromacs.
    force RMS = < (f_C - f_O)•(r_O - r_C) >

    Autocorrelation of this force (f_C - f_O)•(r_O - r_C) is also calculated

    Tested for atomistic simulations with dt = 100 fs
    """

    import pytrr

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_atoms = traj.n_atoms    

    num_frames = traj.n_frames


    # read forces
    forces = []
    with open(forcetrajfile, 'rb') as f:
        for _ in range(num_frames):
            header = pytrr.read_trr_header(f)
            data = pytrr.read_trr_data(f, header)
            forces += [ data['f'] ]
    forces = np.array(forces)

    
    # atom Get indices of C,O from grofile
    C_indices = []
    O_indices = []
    with open(grofile, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if ' 1 ' in line:
            line_start = i
            break
    for i,line in enumerate(lines):
        if ' C ' in line:
            C_indices += [i-line_start]
        elif ' O ' in line:
            O_indices += [i-line_start]
    
    C_indices = np.array(C_indices)
    O_indices = np.array(O_indices)


    # Plot force in C=O group
    CO_force_undirectional = forces[:,C_indices] - forces[:,O_indices]
    r =[]
    for f in range(num_frames):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        posO = utils.unwrap_points(positions[f,O_indices], positions[f,C_indices], Lx, Ly, Lz)
        r += [ posO - positions[f,C_indices] ]
    r = np.array(r)
    r_unit = r / np.linalg.norm(r, axis=-1, keepdims=True)
    
    CO_force = np.sum(CO_force_undirectional*r_unit, axis=-1) # per frame and per bond

    # CO_force = np.abs(CO_force)

    min_f = np.min(CO_force)
    max_f = np.max(CO_force)
    hist, bin_edges = np.histogram(CO_force.reshape(-1), bins=50, range=(min_f,max_f))
    dist = hist
    force = 0.5*(bin_edges[:-1]+bin_edges[1:])

    
    x = force
    y = dist

    COforce_rms = np.sqrt(np.mean(CO_force**2))
    COforce_std = np.std(np.abs(CO_force))
    
    mu = np.mean(CO_force, axis=0, keepdims=True)
    CO_ = CO_force - mu
    sigma2 = np.mean( CO_**2, axis=0 )
    auto = [1] # collect auto for each window
    for i in range(1,window_duration):
        auto_ = np.mean( CO_[:-i]*CO_[i:], axis=0 ) / sigma2
        auto += [ np.mean(auto_) ]


    if type(filename) != type(None):
        
        import matplotlib.pyplot as plt
            
        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(x, y, marker='')
        plt.title('Force Distribution for C=O')
        plt.xlabel(r'Force')
        plt.ylabel('Distribution')
        plt.subplots_adjust(bottom=0.18, left=0.18)
        plt.savefig(filename, dpi=400)    

    
    if type(autocorrfilename) != type(None):
        
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(range(len(auto)), auto, marker='')
        plt.title('Autocorrelation of force on C=O')
        plt.xlabel('Frame')
        plt.ylabel('Autocorrelation')
        plt.subplots_adjust(bottom=0.18, left=0.22)
        plt.savefig(autocorrfilename, dpi=400)    


    return COforce_rms, COforce_std, auto




def vibrational_spectra2(pdbfile, filename=None):
    """ 
    NOTE: Doesn't seem to produce consistent and reliable results. BUMMER

    Uses exciton Model given at 
    http://www.2d-ir-spectroscopy.com/

    pdbfile name should not be more than 4 letters

    """
    
    pdb = pdbfile.replace('.pdb','')
    cmd = f'./peptide {pdb}'
    subprocess.run(cmd, shell=True).check_returncode()
    with open(f'{pdb}_lin.dat', 'r') as f:
        lines = f.readlines()

    freq = []
    spectra = []
    for line in lines:
        words = line.split()
        freq += [ float(words[0]) ]
        spectra += [ float(words[1]) ]

    if type(filename) != type(None):
        
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(freq, spectra, marker='')
        plt.title('Vibrational Spectra')
        plt.xlabel(r'Wavenumber (cm$^{-1}$)')
        plt.ylabel('Absorption')
        plt.subplots_adjust(bottom=0.18, left=0.16)
        plt.savefig(filename, dpi=400)




def vibrational_spectra(grofile, trajfile, topfile, molname, frame_iterator, rtpfiles, ffnonbondedfile):
    """
    NOTE: Doesn't seem to produce consistent and reliable results. BUMMER

    Calculating vibrational spectrum, peak position and half max full width properties 
    Empirical Map method used is given in: https://doi.org/10.1021/jp412827s
    
    Tested using charmm36-jul2020.

    U_non-bonded = U_LJ + U_elec
    U_LJ_ij = 4*eps_ij * [ (sigma_ij/rij)^12 - (sigma_ij/rij)^6 ] ]
    F_LJ_ij = -4*eps_ij *[ 12*sigma_ij/rij^13 - 6*sigma_ij/rij^7 ]
    U_elec_ij = 1/4*pi*eps0 * qi*qj / (eps_r*rij) ]      1/4pieps0 = 138.935458   eps_r = 1 by default
    F_elec_ij_i = - 1/4*pi*eps0 * qi*qj / (eps_r*ri^2j) * unit_vec(n_ij)    | rij=rj-ri
    E_elect_ij_i = - 1/4*pi*eps0 * qj / (eps_r*ri^2j) * unit_vec(n_ij)
    More details given in: https://www.charmmtutorial.org/index.php/The_Energy_Function

    mdp params
    ; vdw
    vdwtype                 = cutoff
    vdw-modifier            = force-switch
    rvdw-switch             = 1.0
    rvdw                    = 1.2       ; short-range van der Waals cutoff (in nm)
    ; Electrostatics
    coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
    rcoulomb                = 1.2
    pme_order               = 4         ; cubic interpolation
    fourierspacing          = 0.16      ; grid spacing for FFT
    
    IONS used: [NA, CL]
    TERs used: [NH2]

    ASSUMES:
    - Continugous molecules in the grofile
    - only one type of protein chain
    """

    #---------------------------------------------------------------------------------

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_atoms = traj.n_atoms
    
    # modify script to add more ions and ters
    IONS = ['NA', 'CL']
    TERS = ['NH2']

    radius = 1.2 # rvdw and rcoulomb

    #---------------------------------------------------------------------------------

    # collect sigma and epsilon from ffnonbondedfile
    residues_unique = np.unique([res.name for res in traj.top.residues])
    
    atom_types = {} # per residue
    for rtpfile in rtpfiles:
        with open(rtpfile, 'r') as f:
            lines = f.readlines()
        start = False
        for i,line in enumerate(lines):
            words = line.split()
            if (len(words)==0) or (words[0]==';'):
                continue
            words_before = lines[i-1].split()
            
            if (not start) and ('[ atoms ]' in line) and (len(words_before)==3) and (words_before[1] in residues_unique):
                start = True
                residue = words_before[1]
                atom_types[residue] = {}
                continue

            if (line.split()[0]=='['):
                start = False
            
            if start:
                words = line.split()
                if words[0] == 'HN': # convert HNs to H as given in .arn files of the charmmff
                    words[0] = 'H'
                atom_types[residue][words[0]] = [words[1], float(words[2])]


    # BUG patch: mdtraj seems to use HB3 instead of HB1 for GLU
    # remove this patch if the bug is resolved
    if 'GLU' in atom_types.keys():
        atom_types['GLU']['HB3'] = atom_types['GLU']['HB1']
        atom_types['GLU']['HG3'] = atom_types['GLU']['HG1']



    # Ions - no van der waal interaction, names NA and CL are placeholder
    if 'NA' in residues_unique:
        atom_types['NA']={'NA': ['NA', 1]}
    if 'CL' in residues_unique:
        atom_types['CL']={'CL': ['CL', -1]}
    
    # TER atoms - values are manually copied from the modified charmm36-jul2020.ff
    atom_types['NH2'] = dict(NT = ['NH2', -0.62], HT1=['H', 0.30], HT2=['H', 0.32], H=['H', 0.30], H2=['H', 0.32])
    # atom_types['COOH'] = dict()
    # atom_types['COO-'] = dict()
    
    # protonation H, basically for GLU and ASP
    atom_types['proton'] = {'HE2': ['H', 0]}

    # water
    atom_types['HOH']['O'] = atom_types['HOH']['OW']
    atom_types['HOH']['H1'] = atom_types['HOH']['HW1']
    atom_types['HOH']['H2'] = atom_types['HOH']['HW2']

    #---------------------------------------------------------------------------------

    # Set index wise atom types and charges of all atoms in the system
    system_atom_types = []
    system_atom_charges = []
    for atom in traj.top.atoms:
        res = atom.residue.name
        try:
            atom_types[res][atom.name]
            system_atom_types += [atom_types[res][atom.name][0]]
            system_atom_charges += [atom_types[res][atom.name][1]]
        except KeyError: # check atom in TERs
            found = False
            for ter in TERS:
                if atom.name in atom_types[ter].keys():
                    system_atom_types += [atom_types[ter][atom.name][0]]
                    system_atom_charges += [atom_types[ter][atom.name][1]]
                    found = True
                    break
            if not found:
                if atom.name in atom_types['proton'].keys():
                    system_atom_types += [atom_types['proton'][atom.name][0]]
                    system_atom_charges += [atom_types['proton'][atom.name][1]]
                    found = True
            if not found:
                raise ValueError(f'{atom.name} index {atom.index} not found in {res}, TERS, water, proton or IONS')
    
    system_atom_types = np.array(system_atom_types)
    system_atom_charges = np.array(system_atom_charges)
    atom_types_unique = np.unique(system_atom_types)
    
    #---------------------------------------------------------------------------------

    # Get van der waal params, sigma and epsilon, from ffnonbondedfile
    num = len(atom_types_unique)
    vdw_params_epsilon = np.zeros((num,num))-1 # -ive is there for usage in combination rule, see later
    vdw_params_sigma = np.zeros((num,num))
    vdw_params_atomindex = {}
    for i,atom_type in enumerate(atom_types_unique):
        vdw_params_atomindex[atom_type] = [i]

    with open(ffnonbondedfile, 'r') as f:
        lines = f.readlines()
    start = False
    for line in lines:
        if len(line.split()) == 0:
            continue
        if '[ pairtypes ]' in line:
            start = True
            continue
        if start:
            words = line.split()
            try:
                type1 = words[0]
                type2 = words[1]
                vdw_params_atomindex[type1]
                vdw_params_atomindex[type2]
                sigma = float(words[3])
                epsilon = float(words[4])
                type1_id = vdw_params_atomindex[type1]
                type2_id = vdw_params_atomindex[type2]
                vdw_params_epsilon[type1_id, type2_id] = epsilon
                vdw_params_sigma[type1_id, type2_id] = sigma
                vdw_params_epsilon[type2_id, type1_id] = epsilon
                vdw_params_sigma[type2_id, type1_id] = sigma
            except KeyError:
                pass
    
    for line in lines:
        if (len(line.split())==0) or (line[0]==';') or (line[0]=='#'):
            continue
        if '[ atomtypes ]' in line:
            start = True
            continue
        
        if start:
            words = line.split()
            if words[0]=='[':
                break
            try:
                type_ = words[0]
                vdw_params_atomindex[type_]
                sigma = float(words[5][:-1])
                epsilon = float(words[6][:-1])
                type_id = vdw_params_atomindex[type_]
                vdw_params_epsilon[type_id, type_id] = epsilon
                vdw_params_sigma[type_id, type_id] = sigma
            except KeyError:
                continue

    # Set vdw with all ions to be 0
    for ion in IONS:
        try:
            type_ = ion
            vdw_params_atomindex[type_]      
            type_id = vdw_params_atomindex[type_]
            vdw_params_epsilon[type_id, type_id] = 0
            vdw_params_sigma[type_id, type_id] = 0
        except:
            pass

    
    # use combination rule to calculate rest of the vdw_params missed in pairtypes
    for i,j in itertools.product(range(vdw_params_epsilon.shape[0]), range(vdw_params_epsilon.shape[1])):
        if (vdw_params_epsilon[i,j] < 0) or (vdw_params_epsilon[j,i] < 0):
            epsilon1 = vdw_params_epsilon[i,i]
            epsilon2 = vdw_params_epsilon[j,j]
            sigma1 = vdw_params_sigma[i,i]
            sigma2 = vdw_params_sigma[j,j]
            epsilon = np.sqrt( epsilon1 * epsilon2 )
            sigma = ( sigma1 + sigma2 ) / 2
            vdw_params_epsilon[i,j] = epsilon
            vdw_params_sigma[i,j] = sigma
            vdw_params_epsilon[j,i] = epsilon
            vdw_params_sigma[j,i] = sigma


    #---------------------------------------------------------------------------------
    
    # atom Get indices of C,O,N,H(NH) from grofile
    C_indices = []
    O_indices = []
    N_indices = []
    H_indices = []
    with open(grofile, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        if ' 1 ' in line:
            line_start = i
            break
    for i,line in enumerate(lines):
        if ' C ' in line:
            C_indices += [i-line_start]
        elif ' O ' in line:
            O_indices += [i-line_start]
        elif ' N ' in line:
            N_indices += [i-line_start]
        elif ' HN ' in line:
            H_indices += [i-line_start]
    indices = dict( 
        C = np.array(C_indices), 
        O = np.array(O_indices),
        N = np.array(N_indices),
        H = np.array(H_indices) )
    
    
    # Get bonds for the molecule from the topfile
    bonds_permol = []
    with open(topfile, 'r') as f:
        lines = f.readlines()
    start = False
    for line in lines:
        words = line.split()
        if not start:
            if '[ bonds ]' in line:
                start = True
                continue
        if start:
            if len(words)==0:
                continue
            if words[0][0]=='[':
                start=False
                break
            if not words[0].isdigit():
                continue
            bonds_permol += [ [int(words[0])-1, int(words[1])-1] ]
    bonds_permol = np.array(bonds_permol)

    num_atoms_permol = np.max(bonds_permol.reshape(-1))+1
    num_molecules = get_num_molecules_fromtop(topfile, molname) # only the protein chain

    
    neighbors12 = [[] for _ in range(num_atoms_permol)]
    for bond in bonds_permol:
        neighbors12[bond[0]] += [bond[1]]
        neighbors12[bond[1]] += [bond[0]]
    
    neighbors13 = [[] for _ in range(num_atoms_permol)]
    for i,neighbors_ in enumerate(neighbors12):
        for j in neighbors_:
            neighbors13[i] += neighbors12[j]
        neighbors13[i] = [k for k in neighbors13[i] if k!=i]
        
    
    # turn neigbors12 13 into arrays
    nneigh = [len(neigh) for neigh in neighbors12]
    max_nneigh = np.max(nneigh)
    neighbors12_ = []
    for i,neighbors_ in enumerate(neighbors12):
        neighbors12_ += [ neighbors_+[np.nan]*(max_nneigh - nneigh[i]) ]
    neighbors12 = np.array(neighbors12_)

    nneigh = [len(neigh) for neigh in neighbors13]
    max_nneigh = np.max(nneigh)
    neighbors13_ = []
    for i,neighbors_ in enumerate(neighbors13):
        neighbors13_ += [ neighbors_+[np.nan]*(max_nneigh - nneigh[i]) ]
    neighbors13 = np.array(neighbors13_)

    #---------------------------------------------------------------------------------
    # Calculate Van der Waal force and electrostatic field is calculated for C, H, O, NH1
    # F_LJ_ij = -4*eps_ij *[ 12*sigma_ij/rij^13 - 6*sigma_ij/rij^7 * unit_vec(n_ij)
    # E_elect_ij_i = - 1/4*pi*eps0 * qj / (eps_r*rij^2) * unit_vec(n_ij)
    

    F = dict(                 # vdw force 
        C = np.empty((0,3)),
        O = np.empty((0,3)),
        N = np.empty((0,3)),
        H = np.empty((0,3)) ) 
    E = dict(                 # Electrostatic field
        C = np.empty((0,3)),
        O = np.empty((0,3)),
        N = np.empty((0,3)),
        H = np.empty((0,3)) )

    X = dict(                 # Position
        C = np.empty((0,3)),
        O = np.empty((0,3)),
        N = np.empty((0,3)),
        H = np.empty((0,3)) )

    r_CO = np.empty((0,3))   # C=O bond vector
        


    All_indices = np.empty(0, dtype=int)
    for A in indices:
        All_indices = np.append(All_indices, indices[A], axis=0)

    query_args = dict(mode='ball', r_max=radius, exclude_ii=False)
    for frame in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[frame]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points = positions[frame]
        neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)
        
        query_points = positions[frame, All_indices]
        neighbor_pairs = np.array(neighborhood.query(query_points, query_args).toNeighborList()[:])
        
        num_neighbor_pairs = len(neighbor_pairs)
        
        # Neighbor exclusion
        atoms_in_protein = np.arange(num_neighbor_pairs)[neighbor_pairs[:,1] < num_molecules * num_atoms_permol]
        mol_indices = All_indices[neighbor_pairs[atoms_in_protein,0]] // num_atoms_permol
        atom_indices = All_indices[neighbor_pairs[atoms_in_protein,0]] % num_atoms_permol
        mol_indices_neigh = neighbor_pairs[atoms_in_protein,1] // num_atoms_permol
        atom_indices_neigh = neighbor_pairs[atoms_in_protein,1] % num_atoms_permol

        # exclude 1-1
        filtr = np.ones(len(neighbor_pairs), dtype=bool)
        same_mol = mol_indices == mol_indices_neigh
        filtr[atoms_in_protein] *= ~( same_mol * (atom_indices == atom_indices_neigh) )
        # exclude 1-2
        for i in range(neighbors12.shape[1]):
            filtr[atoms_in_protein] *= ~( same_mol * (atom_indices == neighbors12[atom_indices_neigh,i]) )
        # exclude 1-3
        for i in range(neighbors13.shape[1]):
            filtr[atoms_in_protein] *= ~( same_mol * (atom_indices == neighbors13[atom_indices_neigh,i]) )

        neighbor_pairs = np.compress(filtr, neighbor_pairs, axis=0)
        
        # Force and field calculation
        type1 = system_atom_types[All_indices[neighbor_pairs[:,0]]]
        type2 = system_atom_types[neighbor_pairs[:,1]]
        type1_ids = [ vdw_params_atomindex[t] for t in type1 ]
        type2_ids = [ vdw_params_atomindex[t] for t in type2 ]
        
        epsilon = vdw_params_epsilon[type1_ids, type2_ids]
        sigma = vdw_params_sigma[type1_ids, type2_ids]


        unwrapped_points = utils.unwrap_points(points[neighbor_pairs[:,1]], query_points[neighbor_pairs[:,0]], Lx, Ly, Lz)
        r = unwrapped_points - query_points[neighbor_pairs[:,0]]
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
        r_unit = r / r_norm

        F_lj = -4 * epsilon * ( 12*sigma/r_norm**13 - 6*sigma/r_norm**7 ) * r_unit
        
        
        q = system_atom_charges[neighbor_pairs[:,1]].reshape(-1,1)
        eps_r = 1
        E_coul = - 138.935458 * q / (eps_r*r_norm**2) * r_unit
        
        
        F_ = np.zeros((len(query_points), 3))    
        E_ = np.zeros((len(query_points), 3))
        for i,pair in enumerate(neighbor_pairs):
            F_[pair[0]] += F_lj[i]
            E_[pair[0]] += E_coul[i]
        

        i=0
        X_ = {}
        E__ = {}
        for A in indices: # A: C,O,N,H
            F[A] = np.append(F[A], F_[i:i+len(indices[A])], axis=0)
            
            E__[A] = E_[i:i+len(indices[A])]
            E[A] = np.append(E[A], E__[A], axis=0)
            
            X_[A] = query_points[i:i+len(indices[A])]
            X[A] = np.append(X[A], X_[A], axis=0)

            i += len(indices[A])
        

        X_O = utils.unwrap_points(X_['O'], X_['C'], Lx, Ly, Lz)
        r = X_O - X_['C']
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
        r_unit = r / r_norm
        r_CO = np.append( r_CO, r_unit, axis=0 )

        # E_CO = np.sum( (E__['C'] - E__['O']) * r_unit, axis=-1 )
        # E_CO_mean = np.mean(np.abs(E_CO))
        # E_CO_std = np.std(np.abs(E_CO))
        # print(E_CO_mean, E_CO_std)

    E_CO = np.sum( (E['C'] - E['O']) * r_CO, axis=-1 )
    E_CO_mean = np.mean(np.abs(E_CO))
    E_CO_std = np.std(np.abs(E_CO))

    F_CO = np.sum( (F['C'] - F['O']) * r_CO, axis=-1 )
    F_CO_mean = np.mean(np.abs(F_CO))
    F_CO_std = np.std(np.abs(F_CO))
    
    return E_CO_mean, E_CO_std, F_CO_mean, F_CO_std
    


    # Conversion to a.u. unit from kj/mol/nm 
    # Unit: kJ mol−1 nm−1    1nm = 0.1890 au | 1au = 2625.5 kJ/mol
    # Conversion to a.u.: 1/2625.5/0.1890 = 0.002015237
    # 1/4*pi*eps: 8.987 × 10^9 kJ * m / C^2 = 8.987 * 10^18 kJ nm / C^2 = 8.987 * 10^18 * 2.5669*10^-38 = 23.0693 *10^-20  kJ nm/e^2 = 138.935458 kJ nm / mol /e^2
    for A in indices:
        F[A] *= 0.002015237
        E[A] *= 0.002015237

    
    # align C, O, N in X-Y plane, align C,O bond in Y-axis
    # ASSUMING same index C O N H correspond to the same peptide bond 
    X_ = []
    num = min([len(X['C']),len(X['O']),len(X['N']),len(X['H'])])
    for i in range(num):
        C = X['C'][i]
        O = X['O'][i]
        N = X['N'][i]
        H = X['H'][i]

        v1 = O-C
        v2 = [0,1,0]
        q1 = quaternion.q_between_vectors(v1, v2)
        O = quaternion.qv_mult(q1, O-C) + C
        N = quaternion.qv_mult(q1, N-C) + C
        H = quaternion.qv_mult(q1, H-C) + C

        v1 = np.cross(O-C, N-C)
        v2 = [0,0,-1]
        q2 = quaternion.q_between_vectors(v1, v2)
        q = quaternion.qq_mult(q2,q1)

        F['C'][i] = quaternion.qv_mult(q, F['C'][i])
        F['O'][i] = quaternion.qv_mult(q, F['O'][i])
        F['N'][i] = quaternion.qv_mult(q, F['N'][i])
        F['H'][i] = quaternion.qv_mult(q, F['H'][i])

        E['C'][i] = quaternion.qv_mult(q, E['C'][i])
        E['O'][i] = quaternion.qv_mult(q, E['O'][i])
        E['N'][i] = quaternion.qv_mult(q, E['N'][i])
        E['H'][i] = quaternion.qv_mult(q, E['H'][i])

        
    
    # Calculate freq peak and half-max-full-width
    map1 = dict(
        w0 = 1674.55, 
        Cy = -103.6,
        Oy = 1320.4,
        Nx = -632.7,
        Ny = 3932.5,
        Hx = -1390.8,
        Hy = -1740.4,
        Oy_vdw = 0,
        Nx_vdw = -7905.1,
        Ny_vdw = -858.3)
             
    map2 = dict(
        w0 = 1674.64, 
        Cy = -374.3,  
        Oy = 1787.4,
        Nx = -286.1,
        Ny = 3643.2,
        Hx = -1598.1,
        Hy = -1412.3,
        Oy_vdw = -705.9,
        Nx_vdw = -8233.2,
        Ny_vdw = 0)

    m=map1
    w_peak1 = m['w0'] + \
        m['Cy']*np.mean(E['C'][:,1]) + \
        m['Oy']*np.mean(E['O'][:,1]) + \
        m['Nx']*np.mean(E['N'][:,0]) + \
        m['Ny']*np.mean(E['N'][:,1]) + \
        m['Hx']*np.mean(E['H'][:,0]) + \
        m['Hy']*np.mean(E['H'][:,1]) + \
        m['Oy_vdw']*np.mean(F['O'][:,1]) + \
        m['Nx_vdw']*np.mean(F['N'][:,0]) + \
        m['Ny_vdw']*np.mean(F['N'][:,1])

    
    m=map2
    w_peak2 = m['w0'] + \
        m['Cy']*np.mean(E['C'][:,1]) + \
        m['Oy']*np.mean(E['O'][:,1]) + \
        m['Nx']*np.mean(E['N'][:,0]) + \
        m['Ny']*np.mean(E['N'][:,1]) + \
        m['Hx']*np.mean(E['H'][:,0]) + \
        m['Hy']*np.mean(E['H'][:,1]) + \
        m['Oy_vdw']*np.mean(F['O'][:,1]) + \
        m['Nx_vdw']*np.mean(F['N'][:,0]) + \
        m['Ny_vdw']*np.mean(F['N'][:,1])
    
    

    return w_peak1, w_peak2


