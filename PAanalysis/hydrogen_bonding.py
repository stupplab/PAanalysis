

import numpy as np
import mdtraj
from . import quaternion
from . import utils




def Hbonds(grofile, trajfile, frame_iterator, residuenames=None, freq=0.1):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
    Returns the hbonds distances found in frame_iterator using two criteria:
     - Average distances of hbonds cummulatively found in all the frames in frame_iterator
     - hbonds cummulatively calculated using mdtraj for a particular frequency (freq) avlue
     (See mdtraj documentation for more info)
    Hbond criteria: DHA angle > 2*np.pi/3  and H-A separation < 2.5 A
    """
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
                residue_indices += [atom.index]

    hbonds_all = np.empty((0,3))
    framewise_hbonds = []
    for f in frame_iterator:
        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                if type(residuenames) != type(None):
                    if (traj.top.atom(b[0]).index in residue_indices) or (traj.top.atom(b[2]).index in residue_indices):
                        hbonds_ += [b]
                else:
                    hbonds_ += [b]
        hbonds = np.array(hbonds_)
        
        if len(hbonds) == 0:
            hbonds = hbonds.reshape((0,3))

        framewise_hbonds += [ hbonds ]

        hbonds_all = np.append(hbonds_all, hbonds, axis=0).astype(int)
        
        
    num_hbonds_perframe = np.mean([len(hbonds_f) for hbonds_f in framewise_hbonds])
    
    return num_hbonds_perframe

    
    hbonds_all = np.unique(hbonds_all, axis=0)

    num_hbonds_all = len(hbonds_all)



    r_DH=[]
    r_DA=[]
    r_AH=[]
    r_DH_all=[]
    r_DA_all=[]
    r_AH_all=[]
    theta=[]
    num_hbonds = 0

    
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        hbonds = framewise_hbonds[i]
        if type(residuenames) != type(None):
            hbonds_hash = {}
            for hb in hbonds:
                hbonds_hash[hb[0]] = hb
            args = list(set(residue_indices) & set(hbonds[:,0]))
            hbonds = np.array([hbonds_hash[arg] for arg in args])
            
        if len(hbonds)==0:
            continue
        num_hbonds += len(hbonds)
        posH = positions[f,hbonds[:,1]]
        posD = utils.unwrap_points(positions[f,hbonds[:,0]], posH, Lx, Ly, Lz)
        posA = utils.unwrap_points(positions[f,hbonds[:,2]], posH, Lx, Ly, Lz)
        r_AH_ = np.linalg.norm(posA-posH, axis=-1)
        r_DA_ = np.linalg.norm(posD-posA, axis=-1)
        r_DH_ = np.linalg.norm(posD-posH, axis=-1)
        r_DH += list(r_DH_)
        r_AH += list(r_AH_)
        r_DA += list(r_DA_)
        
        posH = positions[f,hbonds_all[:,1]]
        posD = utils.unwrap_points(positions[f,hbonds_all[:,0]], posH, Lx, Ly, Lz)
        posA = utils.unwrap_points(positions[f,hbonds_all[:,2]], posH, Lx, Ly, Lz)
        r_AH_all_ = np.linalg.norm(posA-posH, axis=-1)
        r_DA_all_ = np.linalg.norm(posD-posA, axis=-1)
        r_DH_all_ = np.linalg.norm(posD-posH, axis=-1)
        r_DH_all += list(r_DH_all_)
        r_AH_all += list(r_AH_all_)
        r_DA_all += list(r_DA_all_)
        
        v1 = posA-posH
        v2 = posD-posH
        theta_ = [ abs(quaternion.angle_v1tov2(v1[i],v2[i])) for i in range(v1.shape[0]) ]
        theta += [ theta_ ]

    num_hbonds_perframe = num_hbonds/len(frame_iterator)
    
    
    return (np.mean(r_DH),
            np.std(r_DH),
            np.mean(r_AH),
            np.std(r_AH), 
            np.mean(r_DA),
            np.std(r_DA),
            num_hbonds_perframe,
            np.mean(r_DH_all),
            np.std(r_DH_all),
            np.mean(r_AH_all),
            np.std(r_AH_all), 
            np.mean(r_DA_all),
            np.std(r_DA_all),
            num_hbonds_all)




def Hbond_chirality():
    """TODO
    """


def CO_orientation_order(grofile, trajfile, frame_iterator):
    """
    NOTE: For atomistic simulations
    
    Calculates the orientation order of the C=O of amide bond wrt to the mean direction
    using http://gisaxs.com/index.php/Orientation_order_parameter
    S = ( 3* <cos^2(theta)> - 1 ) / 2
    theta is angle between the OH orientation and the mean director.
    The first 4 C=O bonds are used
    num_CO: number of CO bonds per PA starting from the alkyl-peptide interface

    Calculates S for all C=Os
    """
    
    import freud 

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------


    # identify the args for C=O bonds using mdtraj
    CO_indices = np.empty((0,2), dtype=int)

    for i, res in enumerate(traj.top.residues):
        id_ = np.array([[None, None]])
        if res.is_protein:
            for atom in res.atoms:
                if atom.is_backbone:
                    if atom.name == 'C':
                        id_[0,0] = atom.index
                    elif atom.name == 'O':
                        id_[0,1] = atom.index
            CO_indices = np.append(CO_indices, id_, axis=0)

    CO_indices = CO_indices.astype(int)

    # Calculation
    costhetas = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])


        rCO = positions_f[CO_indices[:,1]] - positions_f[CO_indices[:,0]]
        rCO /= np.linalg.norm(rCO, axis=1, keepdims=True)

        # Calculate projection costheta of rOH wrt to its mean
        rCO_mean = np.mean(rCO, axis=0)
        rCO_mean /= np.linalg.norm(rCO_mean)

        costhetas = np.append(costhetas, rCO.dot(rCO_mean.reshape(-1,1)).reshape(-1))
        
    S = ( 3 * np.mean(costhetas**2) - 1 ) /2

    return S


def CO_degree_of_alignment(grofile, trajfile, frame_iterator):
    """
    NOTE: For atomistic simulations
    
    Calculates the asphericity using the eigenvalues of the gyration tensor
    formed by the CO vectors
    """
    
    import freud 
    import scipy

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------


    # identify the args for C=O bonds using mdtraj
    CO_indices = np.empty((0,2), dtype=int)

    for i, res in enumerate(traj.top.residues):
        id_ = np.array([[None, None]])
        if res.is_protein:
            for atom in res.atoms:
                if atom.is_backbone:
                    if atom.name == 'C':
                        id_[0,0] = atom.index
                    elif atom.name == 'O':
                        id_[0,1] = atom.index
            CO_indices = np.append(CO_indices, id_, axis=0)

    CO_indices = CO_indices.astype(int)

    # Calculation
    L1 = []
    L2 = []
    L3 = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])


        rCO = positions_f[CO_indices[:,1]] - positions_f[CO_indices[:,0]]
        rCO /= np.linalg.norm(rCO, axis=1, keepdims=True)

        # Calculate eigenvalues
        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute((box, rCO), np.zeros(len(rCO)))
        G = cl_props.gyrations[0]
        L3_, L2_, L1_ = np.sort(scipy.linalg.eigvalsh(G))
        L1 += [L1_]
        L2 += [L2_]
        L3 += [L3_]
        
    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)

    asphericity = np.mean( L1 - ( L2 + L3 ) / 2 )

    return asphericity


def CO_nematic_order(grofile, trajfile, frame_iterator, residuenames=None):
    """
    NOTE: For atomistic simulations
    
    Calculates the nematic order using freud
    formed by the CO vectors
    """
    
    import freud

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------

    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
                residue_indices += [atom.index]
        

    # identify the args for C=O bonds using mdtraj
    CO_indices = np.empty((0,2), dtype=int)

    for i, res in enumerate(traj.top.residues):
        id_ = np.array([[None, None]])
        if res.is_protein:
            for atom in res.atoms:
                if atom.is_backbone:
                    if atom.name == 'C':
                        id_[0,0] = atom.index
                    elif atom.name == 'O':
                        id_[0,1] = atom.index
            CO_indices = np.append(CO_indices, id_, axis=0)

    CO_indices = CO_indices.astype(int)

    if type(residuenames) != type(None):
        CO_hash = {}
        for id_ in CO_indices:
            CO_hash[id_[0]] = id_
        indices = list(set(residue_indices) & set(CO_indices[:,0]))
        CO_indices = np.array([CO_hash[id_] for id_ in indices], dtype=int)


    # Calculation
    S = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        rCO = positions_f[CO_indices[:,1]] - positions_f[CO_indices[:,0]]
        rCO /= np.linalg.norm(rCO, axis=1, keepdims=True)

        # Calculate nematic order
        director = np.array([1,0,0])
        orientations = [quaternion.q_between_vectors(director, v2) for v2 in rCO]
        nematic = freud.order.Nematic(director)
        nematic.compute(orientations)
        S += [nematic.order]
        
    return np.mean(S)



def Hbond_nematic_order(grofile, trajfile, frame_iterator, residuenames=None):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
    
    Calculates the nematic order using freud
    """
    
    import freud 

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------

    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
                residue_indices += [atom.index]


    # Calculation
    S = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        if type(residuenames) != type(None):
            hbonds_hash = {}
            for hb in hbonds:
                hbonds_hash[hb[0]] = hb
            args = list(set(residue_indices) & set(hbonds[:,0]))
            hbonds = np.array([hbonds_hash[arg] for arg in args])
        
        if len(hbonds) == 0:
            continue
        rOH = positions_f[hbonds[:,1]] - positions_f[hbonds[:,2]]
        rOH /= np.linalg.norm(rOH, axis=1, keepdims=True)

        # Calculate nematic order
        director = np.array([1,0,0])
        orientations = [quaternion.q_between_vectors(director, v2) for v2 in rOH]
        nematic = freud.order.Nematic(director)
        nematic.compute(orientations)
        S += [nematic.order]
        
    return np.mean(S)


def local_Hbond_nematic_order(grofile, trajfile, frame_iterator, residuename=None):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
        
    Calculates the nematic order using freud
    """
    
    import freud 

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------

    #------------------------------ filter for residuenames ----------------------
    residue_indices = []
    for atom in traj.top.atoms:
        if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name == residuename:
            residue_indices += [atom.index]

    

    #--------------- select points clusters from the 1st frame ---------------
    
    radius = 0.4
    frame = 0
    Lx, Ly, Lz = traj.unitcell_lengths[frame_iterator[frame]]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    neighborhood = freud.locality.LinkCell(
        box, points[frame], cell_width=radius)
    query_args = dict(
        mode='nearest', num_neighbors=12, r_max=radius, exclude_ii=False)
    query_points = points[frame]
    
    if type(fraction_of_points) != type(None):
        args = np.random.choice(
            len(query_points), 
            size=int(fraction_of_points * len(query_points)))
        query_points = query_points[args]
    
    neighbors = []
    for query_point in query_points:
        neighbor_pairs = np.array(neighborhood.query(
            np.array([query_point]), query_args).toNeighborList()[:])
        neighbors += [neighbor_pairs[:,1]]
    
    cluster_points = []
    for args in neighbors:
        cluster_points += [np.copy(points[:,args])]

    #------------------------------------------------------------------------



    # Calculation
    S = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        if type(residuename) != type(None):
            hbonds_hash = {}
            for hb in hbonds:
                hbonds_hash[hb[0]] = hb
            args = list(set(residue_indices) & set(hbonds[:,0]))
            hbonds = np.array([hbonds_hash[arg] for arg in args])
        
        if len(hbonds) == 0:
            continue
        rOH = positions_f[hbonds[:,1]] - positions_f[hbonds[:,2]]
        rOH /= np.linalg.norm(rOH, axis=1, keepdims=True)

        # Calculate nematic order
        director = np.array([1,0,0])
        orientations = [quaternion.q_between_vectors(director, v2) for v2 in rOH]
        nematic = freud.order.Nematic(director)
        nematic.compute(orientations)
        S += [nematic.order]
        
    return np.mean(S)



def Hbond_degree_of_alignment(grofile, trajfile, frame_iterator):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
    
    Calculates the asphericity using the eigenvalues of the gyration tensor
    formed by the Hbond vectors
    
    """
    
    import freud 
    import scipy

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------


    # Calculation
    L1 = []
    L2 = []
    L3 = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        if len(hbonds) == 0:
            continue
        
        rOH = positions_f[hbonds[:,1]] - positions_f[hbonds[:,2]]
        rOH /= np.linalg.norm(rOH, axis=1, keepdims=True)

        # Calculate eigenvalues
        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute((box, rOH), np.zeros(len(rOH)))
        G = cl_props.gyrations[0]
        L3_, L2_, L1_ = np.sort(scipy.linalg.eigvalsh(G))
        L1 += [L1_]
        L2 += [L2_]
        L3 += [L3_]
        
    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)

    asphericity = np.mean( L1 - ( L2 + L3 ) / 2 )

    return asphericity




def Hbond_orientation_order(grofile, trajfile, frame_iterator):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
    
    Calculates the orientation order of the H-Bonds wrt to the mean direction
    using http://gisaxs.com/index.php/Orientation_order_parameter
    S = ( 3* <cos^2(theta)> - 1 ) / 2
    theta is angle between the OH orientation and the mean director.
    """
    
    import freud 

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------


    # Calculation
    costhetas = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        if len(hbonds) == 0:
            continue
        rOH = positions_f[hbonds[:,1]] - positions_f[hbonds[:,2]]
        rOH /= np.linalg.norm(rOH, axis=1, keepdims=True)

        # Calculate projection costheta of rOH wrt to its mean
        rOH_mean = np.mean(rOH, axis=0)
        rOH_mean /= np.linalg.norm(rOH_mean)

        costhetas = np.append(costhetas, rOH.dot(rOH_mean.reshape(-1,1)).reshape(-1))
        
    S = ( 3 * np.mean(costhetas**2) - 1 ) /2

    return S




def Hbond_orientation2(grofile, trajfile, frame_iterator):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
    
    Calculates the fluctuation of the direction O-H bond as 
    theta is angle between the OH orientation and it's average.
    """
    import freud 

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames


    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions, unitcell_lengths)

    #------------------------------------------------------------------------


    # Calculation
    costhetas = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        if len(hbonds) == 0:
            continue
        rOH = positions_f[hbonds[:,1]] - positions_f[hbonds[:,2]]
        rOH /= np.linalg.norm(rOH, axis=1, keepdims=True)

        # Calculate fluctuation of rOH wrt to its mean
        rOH_mean = np.mean(rOH, axis=0)
        rOH_mean /= np.linalg.norm(rOH_mean)

        costhetas = np.append(costhetas, rOH.dot(rOH_mean.reshape(-1,1)).reshape(-1))
        

    return np.mean(np.abs(costhetas)), np.mean(costhetas)




def Hbond_orientation(grofile, trajfile, frame_iterator, eig_index):
    """
    NOTE: For atomistic simulations
    Calculates baker-hubbard hydrogen bond for frame_iterator
    
    Calculates the fluctuation of the direction O-H bond as 
    theta is angle between the OH orientation and it's average.

    Calculates Hbond orientation wrt to its molecule and wrt to the fiber axis
    eig_index: {0,1,2} 0: largest eigenvalue 2: lowest eigenvalue 
    eigenvectors are ussed for calculating the fiber axis 
    and are calculated using the peptide backbone.

    spatial_fluctuation 

    Hbond criteria: DHA angle > 2*np.pi/3  and H-A separation < 2.5 A
    """
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    # all hbonds that ever occured in the traj
    hbonds_all = np.empty((0,3))
    framewise_hbonds = []
    for f in frame_iterator:
        hbonds = mdtraj.baker_hubbard(traj[f])
        # FILTER Hbonds that are only between NH and C=O
        hbonds_=[]
        for b in hbonds:
            if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
                hbonds_ += [b]
        hbonds = np.array(hbonds_)
        framewise_hbonds += [ hbonds ]
        hbonds_all = np.append(hbonds_all, hbonds, axis=0).astype(int) 
    hbonds_all = np.unique(hbonds_all, axis=0)
    
    rOH = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]

        posH = positions[f,hbonds_all[:,1]]
        posA = utils.unwrap_points(positions[f,hbonds_all[:,2]], posH, Lx, Ly, Lz)

        rOH += [ posH - posA ]

    rOH = np.array(rOH)
    
    
    #------------------------------------ Gyration eigenvectors ------------------------
    
    points = np.empty((0,3))
    for p in backbone_positionss:
        points = np.append(points, p, axis=0)
    
    center = np.mean(points, axis=0)

    # Calculate gyration eigenvectors using only backbone atoms
    
    # Gij = 1/N SUM_n:1_N [ rn_i * rn_j ]
    points -= center
    G = np.zeros((3,3))
    for i,j in itertools.product(range(3),range(3)):
        G[i,j] = np.mean(points[:,i]*points[:,j])
    

    w,v = scipy.linalg.eig(G)

    args = np.argsort(w)[::-1]

    eigvec = v[:,args]

    #------------------------------------ Calculate ------------------------

    # Calculate spatial fluctuation, averaged over all frames
    fluc_spatial = []
    rOH_mean = np.mean(rOH, axis=1)
    for i in range(rOH.shape[0]):
        thetas = [ quaternion.angle_v1tov2(rOH_, rOH_mean[i]) for rOH_ in rOH[i] ]

        fluc_spatial += [ np.sqrt(np.mean(np.array(thetas)**2)) ]

    fluc_spatial = np.mean(fluc_spatial)
    

    # Calculate temporal fluctuation, averaged over all hbonds
    fluc_temporal = []
    rOH_mean = np.mean(rOH, axis=0)
    for i in range(rOH.shape[1]):
        thetas = [ quaternion.angle_v1tov2(rOH_, rOH_mean[i]) for rOH_ in rOH[:,i] ]

        fluc_temporal += [ np.sqrt(np.mean(np.array(thetas)**2)) ]

    fluc_temporal = np.mean(fluc_temporal)
    
    return fluc_spatial, fluc_temporal



def Hbond_autocorrelation(grofile, trajfile, frame_iterator, window_duration, frame_interval, filename=None):
    """
    NOTE: For atomistic simulations
    Autocorrelation is calculated as
    auto(t) = < kij(t0) * k(t0+t) / <kij(t0)^2> >
    kij: binary variable defining H-bond presence between i, j atoms
    auto is averaged over all the hbonds and multiple time windows
    
    frame_iterator: traj frames to use for calculating auto
    window_duration: max time up till which auto is calculated
    frame_interval: number of frames between start of consecutive time windows

    autos are calculated after every frame_interval and then averaged
    """
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    framewise_hbonds = []
    all_hbonds = np.empty((0,3))
    k = {}  # key is (D,H,A)
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        hbonds = mdtraj.baker_hubbard(traj[f])
        framewise_hbonds += [ hbonds ]

        all_hbonds = np.append(all_hbonds, hbonds, axis=0)
        
    all_hbonds = np.unique(all_hbonds, axis=0)

    hbond_id_dict = {}
    for i,hbond in enumerate(all_hbonds):
        hbond_id_dict[tuple(hbond)] = i

    # k is the array that stores hbond presence
    k = np.zeros((len(all_hbonds),len(frame_iterator)))
    for f, hbonds in enumerate(framewise_hbonds):
        for hbond in hbonds:
            k[hbond_id_dict[tuple(hbond)],f] = 1


    # So <kij(t0)^2> = 1 because kij(t0) is 1 for all hbonds in that frame
    # Caluclate average < kij(t0) * k(t0+t) >
    # auto = np.empty((0,window_duration)) # collect auto for each window
    # for i,_ in enumerate(frame_iterator[::frame_interval]):
    #     hbonds0 = framewise_hbonds[i]
    #     args = [hbond_id_dict[tuple(hbond)] for hbond in hbonds0]
    #     if len(frame_iterator[i:])>=window_duration:
    #         auto_ = k[args,i:i+1]*k[args,i:i+window_duration]
    #         auto = np.append(auto, auto_, axis=0)
    # auto = np.mean(auto, axis=0)

    auto = [1]
    args = []
    for i in range(1,len(frame_iterator)):
        hbonds0 = framewise_hbonds[i]
        args += [ [ hbond_id_dict[tuple(hbond)] for hbond in hbonds0 ] ]
    args = np.array(args)

    auto = [1]
    for i in range(1,len(frame_iterator)):
        auto_ = []
        for j in range(len(frame_iterator)-i):
            auto_ += [ np.mean(k[args[j],j]*k[args[j],j+i]) ]
        auto += [ np.mean(auto_) ]
    auto = np.array(auto)


    if type(filename) != type(None):
        fig = plt.figure(figsize=(4/1.2,3/1.2))
        plt.plot(range(len(auto)), auto, marker='o')
        plt.title('H-bond Autocorrelation')
        plt.xlabel('Frame Number')
        plt.ylabel('Autocorrelation')
        plt.subplots_adjust(bottom=0.14, left=0.15)
        plt.savefig(filename, dpi=400)

    return auto 
    



def SASA(grofile, trajfile, frame_iterator):
    """
    TOO SLOW | NOT COMPLETED
    Calculate the SASA - Solvent Accessible Surface Area
    """    
    
    traj = mdtraj.load(trajfile, top=grofile)

    sasa = mdtraj.shrake_rupley(
        traj[frame_iterator], 
        probe_radius=0.14, 
        n_sphere_points=960, 
        mode='residue', 
        change_radii=None, 
        get_mapping=True)

    print(sasa)

    return sasa



