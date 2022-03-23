
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
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    num_frames = traj.n_frames
    
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
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions[:,args], unitcell_lengths)
    #------------------------------------------------------------------------
    
    points = positions[frame_iterator][:,args]

    #-------------------------- Unwrap and centralize -----------------------
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
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

            # rotate system towards cartesian
            v = eigvec[:,0]
            qx = quaternion.q_between_vectors( v, [1,0,0] )
            v = quaternion.qv_mult( qx, eigvec[:,1] )
            qy = quaternion.q_between_vectors( v, [0,1,0] )
            q = quaternion.qq_mult( qy, qx )
            v = quaternion.qv_mult( q, eigvec[:,2] )
            qz = quaternion.q_between_vectors( v, [0,0,1] )
            q = quaternion.qq_mult( qz, q )
            points_f = [ quaternion.qv_mult(q,p) for p in points_f]
            
            points[n] = points_f
    #------------------------------------------------------------------------

    # Calculate RMSF
    points_mean = np.mean(points, axis=0, keepdims=True)
    
    rmsf = np.sqrt(np.mean((points - points_mean)**2))

    return rmsf



def RMSF_specific_residues(grofile, trajfile, frame_iterator, residuenames, make_orientation_invariate=True, alkyl_or_backbone='backbone'):
    """
    atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration

    residuenames: atom names as given in the coPA_water.gro file
    alkyl_or_backbone='alkyl' uses alkyl tail for invariant eigenvectors
    """
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    num_frames = traj.n_frames
    
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
    
    args = backbone_args
    #------------------------------------------------------------------------

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
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions[:,args], unitcell_lengths)
    images_alkyl = utils.find_box_images(positions[:,alkyl_args], unitcell_lengths)
    images_backbone = utils.find_box_images(positions[:,backbone_args], unitcell_lengths)
    #------------------------------------------------------------------------
    
    points = positions[frame_iterator][:,args]
    points_alkyl = positions[frame_iterator][:,alkyl_args]
    points_backbone = positions[frame_iterator][:,backbone_args]

    #-------------------------- Unwrap and centralize -----------------------
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points_f = np.copy(points[n])
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])
        points_f -= np.mean(points_f, axis=0)
        points[n] = points_f

        points_alkyl_f = np.copy(points_alkyl[n])
        points_alkyl_f -= [Lx/2, Ly/2, Lz/2]
        points_alkyl_f = box.unwrap(points_alkyl_f, images_alkyl[f])
        points_alkyl_f -= np.mean(points_alkyl_f, axis=0)
        points_alkyl[n] = points_alkyl_f

        points_backbone_f = np.copy(points_backbone[n])
        points_backbone_f -= [Lx/2, Ly/2, Lz/2]
        points_backbone_f = box.unwrap(points_backbone_f, images_backbone[f])
        points_backbone_f -= np.mean(points_backbone_f, axis=0)
        points_backbone[n] = points_backbone_f
    #------------------------------------------------------------------------

    #---------------------- Make orientational invariant --------------------
    if make_orientation_invariate:
        for n,f in enumerate(frame_iterator):
            points_f = np.copy(points[n])
            points_alkyl_f = np.copy(points_alkyl[n])
            points_backbone_f = np.copy(points_backbone[n])

            # gyration vectors
            if alkyl_or_backbone == 'alkyl':
                w,eigvec = utils.gyration(points_alkyl_f)
            elif alkyl_or_backbone == 'backbone':
                w,eigvec = utils.gyration(points_backbone_f)
            w = np.abs(w)
            wargs = np.argsort(w)[::-1]
            w = w[wargs]
            eigvec = eigvec[:,wargs]
            if n!=0: # flip eigvec near to eigvec_prev
                dotproducts = np.sum( eigvec * eigvec_prev, axis=0, keepdims=True )
                eigvec *= np.sign(dotproducts)
            eigvec_prev = np.copy(eigvec)
            
            # rotate system towards cartesian
            v = eigvec[:,0]
            qx = quaternion.q_between_vectors( v, [1,0,0] )
            q = qx
            v = quaternion.qv_mult( q, eigvec[:,1] )
            qy = quaternion.q_between_vectors( v, [0,1,0] )
            q = quaternion.qq_mult( qy, q )
            v = quaternion.qv_mult( q, eigvec[:,2] )
            qz = quaternion.q_between_vectors( v, [0,0,1] )
            q = quaternion.qq_mult( qz, q )
            points_f = [ quaternion.qv_mult(q,p) for p in points_f]
            
            points[n] = points_f
            # print(points[n,1])
        # print(np.mean((points-np.mean(points, axis=0, keepdims=True))**2))
        # raise
    #------------------------------------------------------------------------

    #-------------------- calculate rmsf per duration -----------------------
    # print(np.sum((points - [points[0]])**2, axis=(1,2))/points.shape[1])
    # raise
    rmsf = np.sqrt(np.mean((points - points[0:1])**2, axis=(1,2)))

    # frame_lengths = np.arange(2, len(points))
    # rmsf = []
    # for frame_length in frame_lengths:
    #     rmsf_ = []
    #     for t in range(0, len(points)-frame_length, frame_length):
    #         points_ = np.copy(points[t:t+frame_length])
    #         points_mean = np.mean(points_, axis=0, keepdims=True)
    #         rmsf_ += [ np.sqrt(np.mean((points_ - points_mean)**2)) ]
        
    #     rmsf += [ np.mean(rmsf_) ]
    #------------------------------------------------------------------------

    # Calculate RMSF
    # points_mean = np.mean(points, axis=0, keepdims=True)
    # print(np.sqrt(np.mean((points - points_mean)**2)))
    
    # rmsf = np.array(rmsf)

    return rmsf


def local_RMSF(grofile, trajfile, frame_iterator, residuenames=None, fraction_of_points=None):
    """
    atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration

    residuenames: atom names as given in the coPA_water.gro file
    fraction_of_point: fraction of randomly selected atoms for local rmsf calculation
    """
    

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    num_frames = traj.n_frames

    #-------------------------------- Non-water atoms ---------------------------
    backbone_args = []
    not_water_args = []
    specific_args = []
    for atom in traj.top.atoms:
        if atom.residue.name in ['C16', '12C', 'HOH', 'NA', 'CL', 'ION']:
            continue
        if atom.name in ['C', 'N', 'CA']:
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
    
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions[:,args], unitcell_lengths)
    #------------------------------------------------------------------------
    
    points = positions[frame_iterator][:,args]

    #-------------------------- Unwrap and centralize -----------------------
    for n,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
        points_f = points[n]
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])
        points_f -= np.mean(points_f, axis=0)
        points[n] = points_f

    #------------------------------------------------------------------------

    #--------------- select points clusters from the 1st frame ---------------
    
    radius = 0.4
    frame = 0
    Lx, Ly, Lz = traj.unitcell_lengths[frame_iterator[frame]]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    neighborhood = freud.locality.LinkCell(
        box, points[frame], cell_width=radius)
    query_args = dict(
        mode='nearest', num_neighbors=2, r_max=radius, exclude_ii=False)
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

    #------------ calculate rmsf per points cluster per duration ------------
    
    frame_lengths = np.arange(2, len(points))
    rmsf = []
    for frame_length in frame_lengths:
        rmsf_ = []
        for cluster_points_ in cluster_points:
            for t in range(0, len(cluster_points_)-frame_length, frame_length):
                cluster_points__ = np.copy(cluster_points_[t:t+frame_length])
                # centralize clusters
                cluster_centers = np.mean(cluster_points__, axis=1, keepdims=True)
                cluster_points__ -= cluster_centers 
                # cal cluster rmsf
                cluster_points__mean = np.mean(cluster_points__, axis=0, keepdims=True)
                rmsf_ += [ np.sqrt(np.mean((cluster_points__ - cluster_points__mean)**2)) ]
        
        rmsf += [ np.mean(rmsf_) ]

    #------------------------------------------------------------------------

    rmsf = np.array(rmsf)

    return frame_lengths, rmsf



def inter_atom_fluctuation(grofile, trajfile, frame_iterator, radius):
    """
    atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration
    
    """
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    num_frames = traj.n_frames
    
    #------------------------------------ Backbone args ------------------------
    
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

    #---------------------------- Calculate fluctuation ------------------------
    args = not_water_args
    Lx, Ly, Lz = traj.unitcell_lengths[frame_iterator[0]]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    query_args = dict(mode='ball', r_max=radius, exclude_ii=True)
    points_f0 = np.copy(positions[frame_iterator[0], args])
    points_f0 -= [Lx/2, Ly/2, Lz/2]
    neighborhood = freud.locality.LinkCell(box, points_f0, cell_width=radius)
    neighbor_pairs = np.array(neighborhood.query(points_f0, query_args).toNeighborList()[:])
    
    points = utils.unwrap_points(points_f0[neighbor_pairs[:,0]], points_f0[neighbor_pairs[:,1]], Lx, Ly, Lz)
    r0 = np.linalg.norm( points - points_f0[neighbor_pairs[:,1]], axis=1 )
    

    fluc = []
    for f in frame_iterator[1:]:
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        points_f = positions[f, args]
        points_f -= [Lx/2, Ly/2, Lz/2]

        points = utils.unwrap_points(points_f[neighbor_pairs[:,0]], points_f[neighbor_pairs[:,1]], Lx, Ly, Lz)
        r = np.linalg.norm( points - points_f[neighbor_pairs[:,1]], axis=1 )
        
        fluc += [ np.sqrt(np.mean((r-r0)**2)) ]


    return np.mean(fluc)




def PA_rotation(grofile, trajfile, frame_iterator):
    """
    atomistic simulation
    Root mean square fluctuation of the peptide backbone
    displacement wrt mean configuration
    
    """
    
    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz
    
    num_frames = traj.n_frames
    
    #------------------------------------ Backbone args ------------------------
    
    backbone_args = []
    for atom in traj.top.atoms:
        if atom.residue.name in ['C16', '12C', 'HOH']:
            continue
        elif atom.name in ['C', 'N', 'CA']:
            backbone_args += [atom.index]
    backbone_args = np.array(backbone_args)
    
    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions[:,backbone_args], unitcell_lengths)

    #---------------------------- Calculate fluctuation ------------------------
    Lx, Ly, Lz = traj.unitcell_lengths[frame_iterator[0]]
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)
    # query_args = dict(mode='ball', r_max=radius, exclude_ii=True)
    points_f0 = np.copy(positions[frame_iterator[0], backbone_args])
    points_f0 = box.unwrap(points_f0, images[f0])

    w,eigvec = utils.gyration(points_f0)
    vec = eigvec[np.argmax(w)]

    


    return rotation
