


import numpy as np
import mdtraj
from . import quaternion
from . import utils
import matplotlib.pyplot as plt





def curvature(grofile, trajfile, frame_iterator):
    """Calculates the two principle components of the curvature 1/R1 and 1/R2 
    using onlu C O N H atoms

    by fitting ellipsoid
    """

    traj = mdtraj.load(trajfile, top=grofile)
    
    positions = traj.xyz


    # Get atom indices of C,O,N,H(NH) from grofile
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


    # Calculate curvature
    for i,f in enumerate(frame_iterator):
        Lx, Ly, Lz = traj.unitcell_lengths[f]

        posH = positions[f,hbonds_all[:,1]]
        posA = utils.unwrap_points(positions[f,hbonds_all[:,2]], posH, Lx, Ly, Lz)
