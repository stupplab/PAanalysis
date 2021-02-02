""" Analysis routines build for MARTINI simulations of peptide amphiphiles
"""


import numpy as np
import os
import itertools
from .utils import *
from . import quaternion





def peptide_backbone_alignment(itpfile, topfile, grofile, trrfile, box):
    """Alignment fluctuation of peptide backbone of the MARTINI molecule
    box is a di
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']
    
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





def betasheet_sc_alignment(itpfile, topfile, grofile, trrfile, residue_indices, box):
    """Alignment deviation of sidechain that forms beta sheet
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile, residue_indices)
    
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']

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




def res_Sq(itpfile, topfile, grofile, trrfile, frame_iterator, q):
    ''' Residue wise scattering function in z axis
    q is N x 3 array
    '''

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)



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

    
    R = positions[frame_iterator]
    ft = np.sum(np.exp(-1j*R.dot(q.T)), axis=1)
    Sq_ = (ft*ft.conjugate()).real/nmol
    Sq = np.mean(Sq_, axis=0)[atom_indices]
    

    # Old code that uses for loop    
    # Sq = []
    # for i,atom_index in enumerate(atom_indices):
    #     tmp = []
    #     for frame in frame_iterator:
    #         R = positions[frame,:,atom_index]
    #         ft = np.sum(np.exp(-1j*R.dot(q.T)), axis=0)
    #         tmp += [(ft*ft.conjugate()).real/len(R)]
    #     Sq += [ np.mean(tmp, axis=0) ]
    

    return res_names, Sq




def res_Sq_z(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    ''' Residue wise scattering function in z axis
    '''

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']


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




    q  = np.linspace(1, 100, 200).reshape(1,-1)
    Sq = []
    for i,atom_index in enumerate(atom_indices):
        tmp = []
        for frame in frame_iterator:
            z = positions[frame,:,atom_index,2].reshape(-1,1)
            ft = np.sum(np.exp(-1j*z.dot(q)), axis=0)
            tmp += [(ft*ft.conjugate()).real/len(z)]

        Sq += [ np.mean(tmp, axis=0) ]
            

    return res_names, q.reshape(-1), Sq






def res_Sq_azimuthal(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    ''' Residue wise scattering function in azimuthal angle
    '''

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)
    
    
    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']


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



    
    q  = np.linspace(1, 100, 100).reshape(1,-1)
    Sq = []
    for i,atom_index in enumerate(atom_indices):
        tmp = []
        for frame in frame_iterator:
            X = positions[frame,:,atom_index].reshape(-1,3)
            X[:,2]=0
            X -= np.mean(X, axis=0)
            theta = np.array([quaternion.angle_v1tov2([1,0,0], x) for x in X]).reshape(-1,1)
            ft = np.sum(np.exp(-1j*theta.dot(q)), axis=0)
            tmp += [(ft*ft.conjugate()).real/len(theta)]
        Sq += [ np.mean(tmp, axis=0) ]
        
    return res_names, q.reshape(-1), Sq






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
    




def molecule_rmsd(itpfile, topfile, grofile, trrfile, box):
    """Root Mean square displacement with respect to assembly centre
    of the molecule's center of mass
    """

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']



    rmsd = []
    t = []
    positions = positions.reshape(num_frames,-1,3)
    frame_com = np.mean(positions, axis=1, keepdims=True)
    for delta_t in range(1,num_frames):
        r = (positions[delta_t:]-frame_com[delta_t:]) - (positions[:-delta_t]-frame_com[:-delta_t])

        r = r.reshape(-1,3)
        r = np.compress(np.abs(r[:,0])<Lx/2, r, axis=0)
        r = np.compress(np.abs(r[:,1])<Ly/2, r, axis=0)
        r = np.compress(np.abs(r[:,2])<Lz/2, r, axis=0)


        rmsd += [np.sqrt(np.mean(( r )**2))]
        t+=[delta_t]
    
    return t, rmsd





def res_rmsd_AUC(itpfile, topfile, grofile, trrfile, stride, box, delta_frames, start_frame):
    ''' Area under the curve of rmsd divided by number of frames is calculated for each residue
    delta_frames: number of frames over which res_displacement is returned
    '''
    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']

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
    

    positions = positions[start_frame:]
    num_frames = positions.shape[0]

    AUC = []
    frame_com = np.mean(positions, axis=(1,2)).reshape(num_frames,1,3)
    for atom_index in atom_indices:
        tmp = []        
        for delta_t in range(1,num_frames):
            r = (positions[delta_t:,:,atom_index]-frame_com[delta_t:]) - (positions[:-delta_t,:,atom_index]-frame_com[:-delta_t])
            r = r[::stride]
            r = r.reshape(-1,3)
            r = np.compress(np.abs(r[:,0])<Lx/2, r, axis=0)
            r = np.compress(np.abs(r[:,1])<Ly/2, r, axis=0)
            r = np.compress(np.abs(r[:,2])<Lz/2, r, axis=0)

            tmp += [np.sqrt(np.mean(( r )**2))]
        AUC += [ np.sum(tmp[:delta_frames]) ]


    return res_names, np.array(AUC)





def res_rmsd_AUC_freud(itpfile, topfile, grofile, trrfile, box, delta_frames):
    ''' Area under the curve of rmsd divided by number of frames is calculated for each residue
    delta_frames: number of frames over which res_displacement is returned
    '''

    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']

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



    # calculate mean square displacement profile wrt time for each residue
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

    msd = freud.msd.MSD(box=box)

    rmsd = []
    t = []
    for delta_t in range(1,delta_frames):
        r1 = positions[delta_t:]
        r2 = positions[:-delta_t]

        r = (r1-r2)[::stride]
        sq_displacement = np.sum(r**2, axis=-1)
        rmsd += [np.sqrt(np.mean(sq_displacement, axis=(0,1)))]
        t+=[delta_t]


    AUC = np.sum(rmsd[:delta_frames], axis=0)[atom_indices]
    

    return res_names, AUC




def hydration_profile(itpfile, topfile, grofile, trajfile, radius, frame_range, box):
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
    positions_W    = get_positions(grofile, trajfile, (num_atoms*nmol, num_atoms*nmol+nmol_W))

    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']

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





def hydration_rdf(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    
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
    
    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']

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
    rdf = []
    positions   -= [Lx/2,Ly/2,Lz/2]
    positions_W -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    bins = 50; r_max = 1.5
    for atom_index in atom_indices:
        rdf_instance = freud.density.RDF(bins=bins, r_max=r_max, r_min=0)
        for frame in frame_iterator:    
            points_W = positions_W[frame]
            query_points = positions[frame,:,atom_index]
            rdf_instance.compute(system=(box, points_W), query_points=query_points, reset=False)
        
        bin_centers = rdf_instance.bin_centers
        rdf += [ rdf_instance.rdf ]
        

    return res_names, bin_centers, rdf




def binding(itpfile_PA, itpfile_pep, topfile, grofile, trrfile, radius, frame_range, box):
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
    
    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']

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
        
   




def average_molecule_structure(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    ''' NOT COMPLETED
    Calculates the descriptor that characterizes the structure and dynamics
    of the PA assembly.
    Average PA molecule (mean and std of positions) is calculated 
    wrt to its center-of-mass and average orientation

    frame_iterator: frames to calculate the value for


    WHAT AM I DOING SERIOUSLY - AVERAGING POSITIONS MAKES SO SENSE
    Molecule also needs to be oriented in the right direction
    '''
    
    

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms       = get_num_atoms(itpfile)
    nmol            = get_num_molecules(topfile, itpname)
    start_index     = 0
    positions       = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames      = positions.shape[0]
    positions       = positions.reshape(-1,nmol,num_atoms,3)
    shape           = positions.shape


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']



    res_position_mean = []
    res_position_std  = []

    positions = positions[list(frame_iterator)]

    # shift molecules to origin
    mol_centroids = np.mean(positions, axis=2, keepdims=True)
    positions -= mol_centroids
    

    # rotate molecules so that their average orientation points in z-axis
    points = positions[:,:,bb_bonds_permol[:,0]]
    ref_points = positions[:,:,bb_bonds_permol[:,1]]
    points = unwrap_points(points, ref_points, Lx, Ly, Lz)
    vectors = points - ref_points
    mol_average_orientation = np.mean(vectors, axis=2)

    positions_ = []
    for i,frame_positions in enumerate(positions):
        frame_positions_ = []
        for j, mol_positions in enumerate(frame_positions):
            v1 = mol_average_orientation[i,j]
            v2 = [0,0,1]
            q = quaternion.q_between_vectors(v1, v2)

            mol_positions_ = []
            for k, p in enumerate(mol_positions):
                mol_positions_ += [ quaternion.qv_mult(q,p) ]

            frame_positions_ += [ mol_positions_ ]

    positions_ += [ frame_positions_ ]
    positions = np.array(positions_)


    points = positions[:,:,sc_bonds_permol[:,0]]
    ref_points = positions[:,:,sc_bonds_permol[:,1]]
    points = unwrap_points(points, ref_points, Lx, Ly, Lz)
    vectors = points - ref_points
    mol_average_orientation = np.mean(vectors, axis=2)

    positions_ = []
    for i,frame_positions in enumerate(positions):
        frame_positions_ = []
        for j, mol_positions in enumerate(frame_positions):
            v1 = mol_average_orientation[i,j]
            v2 = [1,0,0]
            q = quaternion.q_between_vectors(v1, v2)

            mol_positions_ = []
            for k, p in enumerate(mol_positions):
                mol_positions_ += [ quaternion.qv_mult(q,p) ]

            frame_positions_ += [ mol_positions_ ]

    positions_ += [ frame_positions_ ]
    positions = np.array(positions_)

    

    res_position_mean = np.mean(positions, axis=1)
    res_position_std  = np.std(positions, axis=1)


    return res_position_mean, res_position_std






def res_res_separation(itpfile, topfile, grofile, trrfile, radius, frame_iterator, box):
    '''
    NOT COMPLETED
    For each residue on a PA, calculate it's average separation from the closest residue
    '''

    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms       = get_num_atoms(itpfile)
    nmol            = get_num_molecules(topfile, itpname)
    start_index     = 0
    positions       = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames      = positions.shape[0]
    positions       = positions.reshape(-1,nmol,num_atoms,3)
    shape           = positions.shape


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']


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



    # calculate separation
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

    sep = [[]]*len(atom_indices)
    for frame in frame_iterator:
            
        for i,atom_index in enumerate(atom_indices):
            points = positions[frame,:,atom_index]
            neighborhood = freud.locality.LinkCell(box, points, cell_width=radius)
            query_args = dict(mode='ball', r_max=radius, exclude_ii=True)
            
            neighbor_pairs = neighborhood.query(points, query_args).toNeighborList()[:]
            

            # unwrap bond lengths
            points1 = points[neighbor_pairs[:,0]]
            points2 = points[neighbor_pairs[:,1]]
            
            points1 = unwrap_points(
                points1,
                points2, 
                Lx, Ly, Lz)

            distances = np.linalg.norm(points1-points2, axis=1)
            
            sep[i] += list(distances)
            
            print(np.mean(sep[i]))

    raise


    res_res_sep_mean = []
    res_res_sep_std = []
    for i,atom_index in enumerate(atom_indices):
        res_res_sep_mean += [ np.mean(sep[i]) ]
        res_res_sep_std += [ np.std(sep[i]) ]



    return res_names, res_res_sep_mean, res_res_sep_std






def bondorder(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    '''
    Calculate the bond order of the 
    '''

    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']


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



    # shift positions for freud and create box
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    
    
    n_bins_theta = 100
    n_bins_phi = 100
    bod = freud.environment.BondOrder((n_bins_theta, n_bins_phi))
    
    bod_array=[]
    for frame in frame_iterator:
        points = positions[frame].reshape(-1,3)

        bod_array += [ bod.compute(system=(box, points), neighbors={'num_neighbors': 8, 'r_max': 2}).bond_order]
    bod_array = np.mean(bod_array, axis=0)

    # Clean up polar bins for plotting
    bod_array = np.clip(bod_array, 0, np.percentile(bod_array, 99))
    
    return bod_array.T
    




def localdensity(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    '''
    Calculate the local density of each residue around itself
    '''

    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']


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



    # shift positions for freud and create box
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)


    r_max=2
    atom_diameter = 0.5
    density = freud.density.LocalDensity(r_max=r_max, diameter=atom_diameter)

    res_ld_mean=[]
    res_ld_std=[]
    
    for i,atom_index in enumerate(atom_indices):
        ld_ = []
        for frame in frame_iterator:
            points = positions[frame,:,atom_index]
            ld_ += [ density.compute(system=(box, points), query_points=points, neighbors=dict(mode='ball', exclude_ii=True, r_max=r_max)).density]

        res_ld_mean +=[ np.mean(ld_) ]
        res_ld_std +=[ np.std(ld_) ]
    
    return res_names, res_ld_mean, res_ld_std






def diffraction(itpfile, topfile, grofile, trrfile, frame_iterator, box):
    '''
    NOT COMPLETED
    Calculate the diffraction pattern
    '''

    import freud

    itpname = os.path.basename(itpfile).strip('.itp')
    bb_bonds_permol = get_backbone_bonds_permol(itpfile)
    sc_bonds_permol = get_sidechain_bonds_permol(itpfile)
    num_atoms    = get_num_atoms(itpfile)
    nmol         = get_num_molecules(topfile, itpname)
    start_index  = 0
    positions    = get_positions(grofile, trrfile, (start_index, num_atoms*nmol))
    num_frames = positions.shape[0]
    positions = positions.reshape(-1,nmol,num_atoms,3)


    Lx = box['Lx']
    Ly = box['Ly']
    Lz = box['Lz']


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



    # shift positions for freud and create box
    positions   -= [Lx/2,Ly/2,Lz/2]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)


    dp = freud.diffraction.DiffractionPattern(grid_size=512, output_size=512)

    points = positions[200,:,3]
    orientation = [0.70710678, 0., 0.70710678, 0.]
    dp.compute((box, points), view_orientation=orientation)

    return dp
    # res_ld_mean=[]
    # res_ld_std=[]
    
    # for i,atom_index in enumerate(atom_indices):
    #     sq_ = []
    #     for frame in frame_iterator:
    #         points = positions[frame,:,atom_index]
    #         sq_ += [ dp.compute((box, points), view_orientation=[1, 0, 0, 0])]

        # res_ld_mean +=[ np.mean(ld_) ]
        # res_ld_std +=[ np.std(ld_) ]
    
    # return res_names, res_ld_mean, res_ld_std





