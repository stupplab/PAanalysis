
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
    
    args = not_water_args
    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions[:,args], unitcell_lengths)

    #--------------------- Make orientational and translation invarient -------------------
    
    points = []
    for f in frame_iterator:
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        points_f = positions[f, args]
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])
        points_f -= np.mean(points_f, axis=0)
        _,eigvec = utils.gyration(points_f)
        for i,v in enumerate([[1,0,0],[0,1,0],[0,0,1]]):
            q = quaternion.q_between_vectors(eigvec[:,i], v)
            for j,p in enumerate(points_f):
                points_f[j] = quaternion.qv_mult(q,p)

        points += [ points_f ]
    points = np.array(points)
    
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
