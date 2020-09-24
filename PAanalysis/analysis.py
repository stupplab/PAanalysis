""" Analysis routines build for MARTINI simulations of peptide amphiphiles
"""


import numpy as np
import os
import itertools
from .utils import *






def peptide_backbone_alignment(itpfile, topfile, grofile, trrfile):
    """Alignment fluctuation of peptide backbone of the MARTINI molecule
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]

    
    positions = positions.reshape(-1,nmol,num_atoms,3)
    points = positions[:,:,bb_bonds_permol[:,0]]
    ref_points = positions[:,:,bb_bonds_permol[:,1]]
    
    # unwrap bond lengths
    points = unwrap_points(
        points,
        ref_points, 
        Lx, Ly, Lz)

    vectors = points - ref_points


    molalignment_perfiber_std  = np.std(vectors, axis=2)    
    molalignment_distortion_perframe_mean  = np.mean(np.linalg.norm(molalignment_perfiber_std, axis=-1), axis=1)


    return num_frames, molalignment_distortion_perframe_mean





def betasheet_sc_alignment(itpfile, topfile, grofile, trrfile, residue_indices):
    """Alignment deviation of sidechain that forms beta sheet
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile, residue_indices)
    
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]

    positions = positions.reshape(-1,nmol,num_atoms,3)
    points     = positions[:,:,sc_bonds_permol[:,0]]
    ref_points = positions[:,:,sc_bonds_permol[:,1]]
    
    # unwrap bond lengths
    points = unwrap_points(
        points,
        ref_points, 
        Lx, Ly, Lz)

    vectors = points - ref_points

    scalignment_std = np.std(vectors, axis=1) # sc orientation averaged over all molecules
    scalignment_std_perframe = np.sqrt(np.mean(scalignment_std**2, axis=(1,2)))

    
    return num_frames, scalignment_std_perframe





def density_distribution(itpfile, topfile, grofile, trrfile, Lx, Ly, Lz, frame_range, filename):
    """Density distribution
    """

    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)
    shape = positions.shape


    densities = []
    positions    -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    for frame in range(frame_range[0], min([num_frames, frame_range[1]])):
        points = positions[frame].reshape(-1,3)
        # neighborquery = freud.locality.LinkCell(box, points, cell_width=radius)
        gd = freud.density.GaussianDensity(width=(2*Lx,2*Ly,2*Lz), r_max=1, sigma=1)
        densities += list(gd.compute((box,points)).density.reshape(-1))


    

    return densities

    



def structure_factor(itpfile, topfile, grofile, trrfile, frame_range):
    """NEED TO BE TESTED FOR CORRECTNESS
    fourier transform (scattering function) of simulation particles
    S(q) is automatically normalized because of the averaging of points observed
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)
    shape = positions.shape
    

    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    z = np.linspace(0, 10, 20)
    q_vectors = np.array(list(itertools.product(x,y,z)))
        
    

    Sq_frame = []
    for frame in range(frame_range[0], min([num_frames, frame_range[1]]), frame_range[2]):
        R = positions[frame].reshape(-1,3)
        ft = np.sum(np.exp(-1j*R.dot(q_vectors.T)), axis=0)
        Sq_frame += [(ft*ft.conjugate()).real/R.shape[0]]
    Sq = np.mean(Sq_frame, axis=0)


    q_norm = np.linalg.norm(q_vectors, axis=-1)
    bins = np.linspace(0,10,100)
    indices = np.digitize(q_norm, bins)-1 # -1 shifts the indices from left to right edge 

    Sq_binned = np.zeros(len(bins))
    num_i = np.zeros(len(bins))
    for i in indices:
        Sq_binned[i] += Sq[i]
        num_i[i] += 1
    Sq_binned /= num_i


    # Center of mass of peptide residues
    pep_indices = []
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
                if words[3] != 'PAM':
                    pep_indices += [int(words[0])-1]
                    

    Sq_frame = []
    for frame in range(frame_range[0], min([num_frames, frame_range[1]]), frame_range[2]):
        R = np.mean(positions[frame,:,pep_indices], axis=1)
        ft = np.sum(np.exp(-1j*R.dot(q_vectors.T)), axis=0)
        Sq_frame += [(ft*ft.conjugate()).real/R.shape[0]]
    Sq = np.mean(Sq_frame, axis=0)


    q_norm = np.linalg.norm(q_vectors, axis=-1)
    bins = np.linspace(0,10,100)
    indices = np.digitize(q_norm, bins)-1 # -1 shifts the indices from left to right edge 

    Sq_pep_binned = np.zeros(len(bins))
    num_i = np.zeros(len(bins))
    for i in indices:
        Sq_pep_binned[i] += Sq[i]
        num_i[i] += 1
    Sq_pep_binned /= num_i

    

    return bins, Sq_binned, Sq_pep_binned






def radius_of_gyration(itpfile, topfile, grofile, trrfile):
    """radius of gyration of the molecule
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    positions = positions.reshape(-1,nmol,num_atoms,3)
    num_frames = positions.shape[0]

    shape = positions.shape
    mu = np.mean(positions, axis=2).reshape(shape[0],shape[1],1,shape[3])
    

    # unwrap bond lengths
    points = unwrap_points(
        positions,
        np.repeat(mu, num_atoms, axis=2), 
        Lx, Ly, Lz)

    
    rog = np.mean( np.sqrt(np.sum((points - mu)**2, axis=(2,3))/num_atoms), axis =-1)


    
    return num_frames, rog
    




def molecule_rmsd(itpfile, topfile, grofile, trrfile):
    """Root Mean square displacement
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)


    mol_com = np.mean(positions, axis=2)

    rmsd = []
    t = []
    for delta_t in range(1,num_frames):
        rmsd += [np.sqrt(np.mean((mol_com[delta_t:] - mol_com[:-delta_t])**2))]
        t+=[delta_t]

    
    return t, rmsd





def hydration_profile(itpfile, topfile, grofile, trrfile, radius, frame_range):
    """Water density profile as you go from hydrophobic end to hydrophilic end
    """
    
    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)
    shape = positions.shape
    nmol_W         = get_num_molecules(topfile, 'W')
    positions_W    = get_positions(grofile, trrfile, (num_atoms*nmol, num_atoms*nmol+nmol_W))


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



    # calculate hydration
    hydration = []
    positions   -= [Lx/2,Ly/2,Lz/2]
    positions_W -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    for frame in range(frame_range[0], min([num_frames, frame_range[1]])):
        points_W = positions_W[frame]
        neighborhood = freud.locality.LinkCell(box, points_W, cell_width=radius)
        query_args = dict(mode='ball', r_max=radius, exclude_ii=False)
        
        hydration_ = []
        for atom_index in atom_indices:
            points = positions[frame,:,atom_index]
            neighbor_pairs = neighborhood.query(points, query_args).toNeighborList()
            hydration_+= [ len(neighbor_pairs)/nmol ]
        
        hydration += [hydration_]
    
    hydration = np.array(hydration) *4/(4/3*np.pi*radius**3) # convert to #H2O / nm^3

    hydration_profile_mean = np.mean(hydration, axis=0)
    hydration_profile_std = np.std(hydration, axis=0)




    return res_names, hydration_profile_mean, hydration_profile_std






def binding(itpfile_PA, itpfile_pep, topfile, grofile, trrfile, radius, frame_range):
    """Measures the probability of the two molecules - PA and pep - being near to each other
    All neighboring pairs of particles in PA and pep are frame-wise checked 
    to count how many frames their contact last. 
    Then the probability of contact lasting per frame duration is counted.
    """

    import freud


    itpname = os.path.basename(itpfile_PA).replace('.itp','')
    num_atoms_PA    = get_num_atoms(itpfile_PA)
    nmol_PA         = get_num_molecules(topfile, itpname)
    start_index     = 0
    positions_PA    = get_positions(grofile, trrfile, (start_index, num_atoms_PA*nmol_PA))
    num_frames      = positions_PA.shape[0]
    positions_PA    = positions_PA.reshape(-1,nmol_PA,num_atoms_PA,3)
    shape_PA        = positions_PA.shape
    
    itpname = os.path.basename(itpfile_pep).replace('.itp','')
    num_atoms_pep    = get_num_atoms(itpfile_pep)
    nmol_pep         = get_num_molecules(topfile, itpname)
    start_index      = num_atoms_PA*nmol_PA
    positions_pep    = get_positions(grofile, trrfile, (start_index, start_index+num_atoms_pep*nmol_pep))
    positions_pep    = positions_pep.reshape(-1,nmol_pep,num_atoms_pep,3)
    shape_pep        = positions_pep.shape
    


    # Calculate neighboring particle pairs (of PA and pep) for all frames
    neighbor_pairs = []
    neighbor_pairs_duration = []
    neighbor_pairs_duration_stopcount = []
    positions_PA    -= [Lx/2,Ly/2,Lz/2]
    positions_pep   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    for frame in range(frame_range[0], min([num_frames, frame_range[1]])):
        points_pep = positions_pep[frame].reshape(-1,3)
        neighborhood = freud.locality.LinkCell(box, points_pep, cell_width=radius)
                
        query_args = dict(mode='ball', r_max=radius, exclude_ii=False)
        points = positions_PA[frame].reshape(-1,3) # all residues in frame
        this_neighbor_pairs = np.array(list(neighborhood.query(points, query_args).toNeighborList())).tolist()
            

        # count duration of neighbor pairs 
        
        index_wise_pairs = {}
        for i, pair in enumerate(this_neighbor_pairs):
            try:
                index_wise_pairs[pair[0]] += [pair]
            except KeyError:
                index_wise_pairs[pair[0]] = [pair]
        index_wise_pairs_keys = index_wise_pairs.keys()

        if len(neighbor_pairs) != 0:
            for i, pair in enumerate(neighbor_pairs):
                if neighbor_pairs_duration_stopcount[i]:
                    continue
                if pair[0] not in index_wise_pairs_keys:
                    neighbor_pairs_duration_stopcount[i] = True
                    continue
                if pair in index_wise_pairs[pair[0]]:
                    neighbor_pairs_duration[i] += 1
                    this_neighbor_pairs.remove(pair)
                else:
                    neighbor_pairs_duration_stopcount[i] = True

        neighbor_pairs += this_neighbor_pairs
        neighbor_pairs_duration += [0]*len(this_neighbor_pairs)
        neighbor_pairs_duration_stopcount += [False]*len(this_neighbor_pairs)
    

    
    return neighbor_pairs_duration
        
   