


import numpy as np
import mdtraj
from . import quaternion
from . import utils
import matplotlib.pyplot as plt

import freud





def C16_C_eigs(grofile, trajfile, frame_iterator):
    """
    atomistic simulation:
    calculates the radii of gyration of 
    C backbone of the C16
    
    curvature can be estimated as ((E1+E2)/2-E3) 

    """

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz

    num_frames = traj.n_frames
    
    #------------------------------------ C16 C args ------------------------
    C16_C_args = []
    for atom in traj.top.atoms:
        if atom.residue.name in ['C16', '12C'] and atom.name in ['C']:
            C16_C_args += [atom.index]
    args = np.array(C16_C_args)
    
    #------------------------------------------------------------------------

    #------------------------------------ Box Images ------------------------
    # Identify images if particles jump more than half the box length
    unitcell_lengths = [traj.unitcell_lengths[f] for f in range(num_frames)]
    images = utils.find_box_images(positions[:,args], unitcell_lengths)

    #------------------------------------------------------------------------

    # Calculate C16_C_eigs and curvature estimate
    ws=[]
    curv = []
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]
        box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, is2D=False)

        points_f = positions[f, args]
        points_f -= [Lx/2, Ly/2, Lz/2]
        points_f = box.unwrap(points_f, images[f])
        points_f -= np.mean(points_f, axis=0)
        w,eigvec = utils.gyration(points_f)
        w = np.real(w)
        wargs = np.argsort(w)[::-1]
        w = w[wargs]
        eigvec = eigvec[:,wargs]
        ws += [w]
        e1 = eigvec[:,0]/np.linalg.norm(eigvec[:,0])
        e2 = eigvec[:,1]/np.linalg.norm(eigvec[:,1])
        e3 = eigvec[:,2]/np.linalg.norm(eigvec[:,2])
        curv += [ (w[0]+w[1])/2 - w[2] ]

        # hbonds = mdtraj.baker_hubbard(traj[f])
        # # FILTER Hbonds that are only between NH and C=O
        # hbonds_=[]
        # for b in hbonds:
        #     if traj.top.atom(b[0]).name == 'N' and traj.top.atom(b[2]).name == 'O':
        #         hbonds_ += [b]
        # hbonds = np.array(hbonds_)
        # rOH = positions[f,hbonds[:,1]] - positions[f,hbonds[:,2]]
        # rOH /= np.linalg.norm(rOH, axis=1, keepdims=True)
        # projection_OH_on_e1 = np.mean(np.abs(rOH.dot(e1.reshape(-1,1))))
        # projection_OH_on_e2 = np.mean(np.abs(rOH.dot(e2.reshape(-1,1))))

    w_mean = np.mean(ws, axis=0)
    
    return w_mean, np.mean(curv)
        


