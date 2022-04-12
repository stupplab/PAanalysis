
import numpy as np
import os
from . import utils
from . import quaternion

import freud
import mdtraj
import scipy.linalg
import itertools
    
    
def RMSF(grofile, trajfile, frame_iterator, make_orientation_invariate=True):
    """
    atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration
    
    """
    
    #------------------------------------ Config -------------------------------
    traj0 = mdtraj.load(grofile)
    frame0 = traj0.xyz
    traj = mdtraj.load(trajfile, top=grofile)
    positions = np.append(frame0, traj.xyz, axis=0)
    num_frames = len(positions)
    unitcell_lengths = [traj0.unitcell_lengths[0]] + [traj.unitcell_lengths[f] for f in range(len(traj))]
    #------------------------------------------------------------------------
    
    #------------------------------------ Non-water atoms ------------------------
    backbone_args = []
    not_water_args = []
    for atom in traj.top.atoms:
        if atom.residue.name in ['C16', '12C', 'HOH', 'NA', 'CL', 'ION']:
            continue
        if atom.name in ['C', 'N', 'CA']:
            backbone_args += [atom.index]
        not_water_args += [atom.index]
    backbone_args = np.array(backbone_args)
    not_water_args = np.array(not_water_args)
    
    args = not_water_args
    #------------------------------------------------------------------------

    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    images = utils.find_box_images(positions[:,args], unitcell_lengths)
    #------------------------------------------------------------------------
    
    points = positions[frame_iterator][:,args]

    #-------------------------- Unwrap and centralize -----------------------
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points_f = points[n]
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])
        points_f -= np.mean(points_f, axis=0)
        points[n] = points_f

    #------------------------------------------------------------------------
    
    #---------------------- Make orientational invariant --------------------
    if make_orientation_invariate:
        for n,f in enumerate(frame_iterator):
            points_f = np.copy(points[n])
            
            # gyration vectors
            w,eigvec = utils.gyration(points_f)
            w = np.abs(w)
            wargs = np.argsort(w)[::-1]
            w = w[wargs]
            eigvec = eigvec[:,wargs]
            if n!=0: # flip eigvec near to eigvec_prev
                dotproducts = np.sum( eigvec * eigvec_prev, axis=0, keepdims=True )
                eigvec *= np.sign(dotproducts)
            eigvec_prev = np.copy(eigvec)

            # # rotate system towards cartesian
            # v = eigvec[:,0]
            # qx = quaternion.q_between_vectors( v, [1,0,0] )
            # v = quaternion.qv_mult( qx, eigvec[:,1] )
            # qy = quaternion.q_between_vectors( v, [0,1,0] )
            # q = quaternion.qq_mult( qy, qx )
            # v = quaternion.qv_mult( q, eigvec[:,2] )
            # qz = quaternion.q_between_vectors( v, [0,0,1] )
            # q = quaternion.qq_mult( qz, q )
            # points_f = [ quaternion.qv_mult(q,p) for p in points_f]
            
            # project each point on the eigenvectors
            points_f = np.array([ p.dot(eigvec)/np.linalg.norm(eigvec,axis=0) for p in points_f])
            points[n] = points_f
    #------------------------------------------------------------------------

    # Calculate RMSF
    points_mean = np.mean(points, axis=0, keepdims=True)
    
    rmsf = np.sqrt(np.mean((points - points_mean)**2))

    return rmsf



def RMSF_specific_residues(grofile, trajfile, frame_iterator, residuenames, make_orientation_invariate=True, residuenames_for_invariance=None):
    """
     - atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration
    
    residuenames: atom names as given in the coPA_water.gro file
    alkyl_or_backbone='alkyl' uses alkyl tail for invariant eigenvectors
    """
    
    #------------------------------------ Config -------------------------------
    traj0 = mdtraj.load(grofile)
    frame0 = traj0.xyz
    traj = mdtraj.load(trajfile, top=grofile)
    positions = np.append(frame0, traj.xyz, axis=0)
    num_frames = len(positions)
    unitcell_lengths = [traj0.unitcell_lengths[0]] + [traj.unitcell_lengths[f] for f in range(len(traj))]
    #------------------------------------------------------------------------

    #------------------------------------ Non-water atoms ------------------------
    backbone_args = []
    not_water_args = []
    # specific_args = []
    alkyl_args = []
    for atom in traj.top.atoms:
        if atom.residue.name in ['HOH', 'NA', 'CL', 'ION']:
            continue
        if atom.residue.name in ['C16', '12C']:
            alkyl_args += [atom.index]
        if atom.name in ['C', 'N', 'CA']:
            backbone_args += [atom.index]
        not_water_args += [atom.index]
        # if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
            # specific_args += [atom.index]
    backbone_args = np.array(backbone_args)
    not_water_args = np.array(not_water_args)
    
    args = not_water_args
    args_for_invariance = np.copy(args)
    #------------------------------------------------------------------------

    #------------------------------ Filter for residuenames ----------------------
    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if (str(atom.residue).replace(atom.residue.name, '')+atom.residue.name) in residuenames:
                residue_indices += [atom.index]
                
        args = list(set(residue_indices) & set(args))
    
    if type(residuenames_for_invariance) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if (str(atom.residue).replace(atom.residue.name, '')+atom.residue.name) in residuenames_for_invariance:
                residue_indices += [atom.index]
                
        args_for_invariance = list(set(residue_indices) & set(args_for_invariance))
    #-----------------------------------------------------------------------------

    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    images = utils.find_box_images(positions[:,args], unitcell_lengths)
    images_for_invariance = utils.find_box_images(positions[:,args_for_invariance], unitcell_lengths)
    #------------------------------------------------------------------------
    
    #---------------------------------- Select points ------------------------
    points = positions[frame_iterator][:,args]
    points_for_invariance = positions[frame_iterator][:,args_for_invariance]
    #------------------------------------------------------------------------
    
    #-------------------------- Unwrap and centralize -----------------------
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points_f = np.copy(points[n])
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])

        points_for_invariance_f = np.copy(points_for_invariance[n])
        points_for_invariance_f -= [Lx/2, Ly/2, Lz/2]
        points_for_invariance_f = box.unwrap(points_for_invariance_f, images_for_invariance[f])
        center = np.mean(points_for_invariance_f, axis=0, keepdims=True)
        
        points_f -= center
        points[n] = points_f
        points_for_invariance_f -= center
        points_for_invariance[n] = points_for_invariance_f    
    #------------------------------------------------------------------------
    
    #---------------------- Make orientational invariant --------------------
    if make_orientation_invariate:
        for n,f in enumerate(frame_iterator):
            points_f = np.copy(points[n])

            points_for_invariance_f = np.copy(points_for_invariance[n])

            # gyration vectors
            w,eigvec = utils.gyration(points_for_invariance_f)
            w = np.abs(w)
            wargs = np.argsort(w)[::-1]
            w = w[wargs]
            eigvec = eigvec[:,wargs]
            if n!=0: # flip eigvec near to eigvec_prev
                dotproducts = np.sum( eigvec * eigvec_prev, axis=0, keepdims=True )
                eigvec *= np.sign(dotproducts)
            eigvec_prev = np.copy(eigvec)
            
            # rotate system towards cartesian
            # v = eigvec[:,0]
            # qx = quaternion.q_between_vectors( v, [1,0,0] )
            # q = qx
            # v = quaternion.qv_mult( q, eigvec[:,1] )
            # qy = quaternion.q_between_vectors( v, [0,1,0] )
            # q = quaternion.qq_mult( qy, q )
            # v = quaternion.qv_mult( q, eigvec[:,2] )
            # qz = quaternion.q_between_vectors( v, [0,0,1] )
            # q = quaternion.qq_mult( qz, q )
            # points_f = [ quaternion.qv_mult(q,p) for p in points_f]
            
            # project each point on the eigenvectors
            points_f = np.array([ p.dot(eigvec)/np.linalg.norm(eigvec,axis=0) for p in points_f])
            points[n] = points_f
    #------------------------------------------------------------------------
    
    #-------------------- calculate rmsf per duration -----------------------
    
    rmsf = np.sqrt(np.mean((points - points[0:1])**2, axis=(1,2)))

    #------------------------------------------------------------------------

    return rmsf


def CC_RMSF(grofile, trajfile, frame_iterator, residuenames=None):
    """
    atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration

    residuenames: atom names as given in the coPA_water.gro file
    fraction_of_point: fraction of randomly selected atoms for local rmsf calculation
    """
    
    #------------------------------------ Config -------------------------------
    traj0 = mdtraj.load(grofile)
    frame0 = traj0.xyz
    traj = mdtraj.load(trajfile, top=grofile)
    positions = np.append(frame0, traj.xyz, axis=0)
    num_frames = len(positions)
    unitcell_lengths = [traj0.unitcell_lengths[0]] + [traj.unitcell_lengths[f] for f in range(len(traj))]
    #------------------------------------------------------------------------

    #-------------------------------- Non-water atoms ---------------------------
    backbone_args = []
    not_water_args = []
    specific_args = []
    for atom in traj.top.atoms:
        if atom.residue.name in ['C16', '12C', 'HOH', 'NA', 'CL', 'ION']:
            continue
        if atom.name in ['C']:
            backbone_args += [atom.index]
        not_water_args += [atom.index]
        # if str(atom.residue).replace(atom.residue.name, '')+atom.residue.name in residuenames:
        #     specific_args += [atom.index]
    backbone_args = np.array(backbone_args)
    not_water_args = np.array(not_water_args)
    
    args = backbone_args
    #-----------------------------------------------------------------------------
    
    #------------------------------ Filter for residuenames ----------------------
    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if (str(atom.residue).replace(atom.residue.name, '')+atom.residue.name) in residuenames:
                residue_indices += [atom.index]

        args = list(set(residue_indices) & set(args))
    #-----------------------------------------------------------------------------
    
    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    images = utils.find_box_images(positions[:,args], unitcell_lengths)
    #------------------------------------------------------------------------
    
    points = positions[frame_iterator][:,args]

    #-------------------------- Unwrap and centralize -----------------------
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points_f = np.copy(points[n])
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])
        points_f -= np.mean(points_f, axis=0)
        points[n] = points_f
    #------------------------------------------------------------------------

    #--------------- select C-C points from the 1st frame ---------------
    radius = 1
    frame = 0
    Lx, Ly, Lz = unitcell_lengths[frame_iterator[frame]]
    # using larger box since points are already unwrapped
    box = freud.box.Box(Lx=2*Lx, Ly=2*Ly, Lz=2*Lz, is2D=False)
    neighborhood = freud.locality.LinkCell(
        box, points[frame], cell_width=radius)
    query_args = dict(
        mode='nearest', num_neighbors=1, r_max=radius, exclude_ii=True)
    query_points_args = np.arange(len(points[frame]))
    

    neighbor_pairs = np.array(neighborhood.query(
        points[frame], query_args).toNeighborList()[:])
    
    CC_points = []
    for args in neighbor_pairs:
        CC_points += [np.copy(points[:,args])]
    #------------------------------------------------------------------------

    #------------ calculate rmsf of C-C separation per duration -------------
    rmsf = []
    for i,this_CC_points in enumerate(CC_points):
        r_s = np.linalg.norm(this_CC_points[:,0] - this_CC_points[:,1], axis=-1)
        mu = np.mean(r_s)
        rmsf_ = np.sqrt( np.mean(( r_s - mu )**2) ) #/ mu
        rmsf += [rmsf_]
        
    rmsf = np.mean(rmsf)
    #------------------------------------------------------------------------

    return rmsf



def CO_rotation(grofile, trajfile, frame_iterator, residuenames):
    """
    Calculates the angular fluctuation of the CO vector
    """
    
    #------------------------------------ Config -------------------------------
    traj0 = mdtraj.load(grofile)
    frame0 = traj0.xyz
    traj = mdtraj.load(trajfile, top=grofile)
    positions = np.append(frame0, traj.xyz, axis=0)
    num_frames = len(positions)
    unitcell_lengths = [traj0.unitcell_lengths[0]] + [traj.unitcell_lengths[f] for f in range(len(traj))]
    #------------------------------------------------------------------------
    
    #-------------------------------- CO indices ---------------------------
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
    #-----------------------------------------------------------------------------

    #------------------------------ filter for residuenames ----------------------
    if type(residuenames) != type(None):
        residue_indices = []
        for atom in traj.top.atoms:
            if (str(atom.residue).replace(atom.residue.name, '')+atom.residue.name) in residuenames:
                residue_indices += [atom.index]
                
        CO_hash = {}
        for id_ in CO_indices:
            CO_hash[id_[0]] = id_
        indices = list(set(residue_indices) & set(CO_indices[:,0]))
        CO_indices = np.array([CO_hash[id_] for id_ in indices], dtype=int)
    #---------------------------------------------------------------------------
    
    #---------------------------------- box images --------------------------
    # Identify images if particles jump more than half the box length
    images = utils.find_box_images(positions, unitcell_lengths)
    #------------------------------------------------------------------------
    
    #------------------------------- calculation ----------------------------    
    rCO = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        positions_f = box.unwrap(positions[f], images[f])

        rCO_ = positions_f[CO_indices[:,1]] - positions_f[CO_indices[:,0]]
        rCO_ /= np.linalg.norm(rCO_, axis=1, keepdims=True)
        rCO += [rCO_]
    rCO = np.array(rCO)
    
    # Calculate nematic order
    S = []
    for i in range(rCO.shape[1]):
        rCO_ = np.copy(rCO[:,i])
        director = np.array([1,0,0])
        orientations = [quaternion.q_between_vectors(director, v2) for v2 in rCO_]
        nematic = freud.order.Nematic(director)
        nematic.compute(orientations)
        S += [nematic.order]
        
    #-----------------------------------------------------------------------------

    return np.mean(S)






