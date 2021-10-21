
import numpy as np
import os
from . import utils
from . import quaternion

import freud
import mdtraj
import scipy.linalg
import itertools
    
    
def RMSF(grofile, trajfile, frame_iterator):
    """
    NOT COMPLETED
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
