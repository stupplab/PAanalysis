"""
Utility functions to be used by analyze.py
"""

import numpy as np
import mdtraj as md
import itertools
import scipy.linalg

##################################### Secondary methods #####################################



def get_bonds_per_molecule(itpfile):
    """Return array of bonded atom pairs of a MARTINI molecule
    using the itpfile
    """
    
    bonds = []
    start = False
    with open(itpfile, 'r') as f:
        for line in f:
            if ('[ bonds ]' in line) or ('[ constraints ]' in line):
                start = True
                continue
            if start:
                words = line.split()
                if words == []:
                    start = False
                    continue
                if not words[0].isdigit():
                    continue
                bonds = bonds + [[int(words[0]),int(words[1])]]

    # starting the index from zero
    bonds = np.array(bonds)-1
    
    return bonds            

                

def get_backbone_bonds_permol(itpfile):
    """Return array of bonded atom pairs of the backbone of peptide sequence of MARTINI molecule
    using the itpfile
    """
    
    bonds = []
    start = False
    with open(itpfile, 'r') as f:
        for line in f:
            if ('Backbone bonds' in line):
                start = True
                continue
            if start:
                words = line.split()
                if (words == []) or (not words[0].isdigit()):
                    start = False
                    break
                bonds = bonds + [[int(words[0]),int(words[1])]]

    # starting the index from zero
    bonds = np.array(bonds)-1
    
    # add alkyl tail which is categorized in sidechain in itp
    # bonds = np.append(bonds, bonds_alkyl, axis=0)

    return bonds            



def get_sidechain_bonds_permol(itpfile, residue_indices=None):
    """Return array of bonded atom pairs of the sidechain of MARTINI molecule
    using the itpfile
    """
    
    if residue_indices == None:
        residue_indices = []
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
                    if ' SC' in line:
                        if words[3] != 'PAM':
                            residue_indices += [int(words[2])-1]

    
    atom_indices = []
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
                if int(words[2])-1 in residue_indices:
                    atom_indices += [int(words[0])-1]


    bonds = []
    start = False
    with open(itpfile, 'r') as f:
        for line in f:
            if ('Sidechain bonds' in line) or ('[ constraints ]' in line):
                start = True
                continue
            if start:
                words = line.split()
                if (words == []) or (not words[0].isdigit()):
                    start = False
                    continue
                bonds = bonds + [[int(words[0]),int(words[1])]]

    
    # starting the index from zero
    bonds = np.array(bonds)-1

    # filter bonds containing residue_indices
    bonds_ = []
    for bond in bonds:
        if (bond[0] in atom_indices) and (bond[0] in atom_indices):
            bonds_ += [bond]
    bonds = bonds_


    return np.array(bonds)




def get_num_atoms(itpfile):
    """
    Return number of atoms of a MARTINI molecule
    using the itpfile
    """
    
    num_atoms = 0
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
                num_atoms += 1

    
    return num_atoms




def get_num_molecules(topfile, name):
    """Return the number of molecules in the simulation.
    Curretnly returns PA
    """

    with open(topfile, 'r') as f:
        for line in f:
            if line.split() != []:
                if line.split()[0]==name:
                    return int(line.split()[1])




def get_positions(gro, trajfile, atom_range):
    """Returns all the wrapped positions the MARTINI simulation trajectory
    for a molecule. 
    Returns trajectory of atom_range = (start, end)
    """


    traj = md.load(trajfile, top=gro)
    

    positions = traj.xyz[:,atom_range[0]:atom_range[1]]


    return positions



def get_num_atoms_fromGRO(filename):
    # Returns the number in the 2nd line of grofile

    num_atoms = 0
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[1].split()[0])

    return num_atoms


def get_num_molecules_fromtop(topfile, molname):
    # Get the number of Protein from the [ molecules ] section

    with open(topfile, 'r') as f:
        lines = f.readlines()
        start = False
        for i,line in enumerate(lines):
            if '#mols' in line:
                words = lines[i+1].split()
                if molname == words[0]:
                    num_molecules = int(words[1])
                    break
        return num_molecules


def get_num_atoms_fromtop(topfile, molname):
    # Get the number of Protein from the [ molecules ] section

    with open(topfile, 'r') as f:
        lines = f.readlines()
    
    start1 = False
    start2 = False
    num_atoms = 0
    for i,line in enumerate(lines):
        if '[ moleculetype ]' in line:
            if molname in ''.join(lines[i+1:i+3]):
                start1 = True
                continue
        if start1:
            if '[ atoms ]' in line:
                start2 = True
                continue
        if start2:
            words = line.split()
            if len(words) == 0:
                continue
            if words[0].isdigit():
                num_atoms += 1
            if words[0] == '[':
                break
    
    return num_atoms


def unwrap_points(points, ref_points, Lx, Ly, Lz):
    """Accepts two arrays of points in 3D and box dimensions.
    Arrays may have any shape as long as the last axis has 3D coordinates.
    Unwraps the <point> position in case the distance between the <point> 
    and <ref_point> is more than half the box length along the corresponding dimension.
    Box dimensions are (0 to Lx) (0 to Ly) (0 to Lz).
    """

    shape = points.shape

    points = np.copy(points.reshape(-1,3))
    ref_points = np.copy(ref_points.reshape(-1,3))
    
    filtr = points[:,0]-ref_points[:,0] > Lx/2
    points[filtr,0] -= Lx

    filtr = points[:,0]-ref_points[:,0] < -Lx/2
    points[filtr,0] += Lx

    filtr = points[:,1]-ref_points[:,1] > Ly/2
    points[filtr,1] -= Ly

    filtr = points[:,1]-ref_points[:,1] < -Ly/2
    points[filtr,1] += Ly

    filtr = points[:,2]-ref_points[:,2] > Lz/2
    points[filtr,2] -= Lz

    filtr = points[:,2]-ref_points[:,2] < -Lz/2
    points[filtr,2] += Lz


    # for i in range(len(points)):
    #     if points[i,0]-ref_points[i,0] > Lx/2:
    #         points[i,0] -= Lx
    #     elif points[i,0]-ref_points[i,0] < -Lx/2:
    #         points[i,0] += Lx

    #     if points[i,1]-ref_points[i,1] > Ly/2:
    #         points[i,1] -= Ly
    #     elif points[i,1]-ref_points[i,1] < -Ly/2:
    #         points[i,1] += Ly


    #     if points[i,2]-ref_points[i,2] > Lz/2:
    #         points[i,2] -= Lz
    #     elif points[i,2]-ref_points[i,2] < -Lz/2:
    #         points[i,2] += Lz


    return points.reshape(shape)




def find_box_images(positions, unitcell_lengths):
    """
    Go frame by frame from the start and identify particle box images
    by the sudden shift in coordinate
    """
    num_frames = positions.shape[0]
    images = np.zeros((positions.shape[0], positions.shape[1], 3))

    for f in range(1,num_frames):
        Lx, Ly, Lz = unitcell_lengths[f]
        
        images[f] = np.copy(images[f-1])

        filtr = (positions[f,:,0] - positions[f-1,:,0]) < -Lx/2
        images[f,filtr,0] = images[f,filtr,0] + 1        
        filtr = (positions[f,:,0] - positions[f-1,:,0]) > Lx/2
        images[f,filtr,0] = images[f,filtr,0] - 1

        filtr = positions[f,:,1] - positions[f-1,:,1] < -Ly/2
        images[f,filtr,1] = images[f,filtr,1] + 1
        filtr = positions[f,:,1] - positions[f-1,:,1] > Ly/2        
        images[f,filtr,1] = images[f,filtr,1] - 1

        filtr = positions[f,:,2] - positions[f-1,:,2] < -Lz/2
        images[f,filtr,2] = images[f,filtr,2] + 1
        filtr = positions[f,:,2] - positions[f-1,:,2] > Lz/2
        images[f,filtr,2] = images[f,filtr,2] - 1

    return images



def gyration(points):
    # Calculate gyration eigenvectors using only backbone atoms
    # Gij = 1/N SUM_n:1_N [ rn_i * rn_j ]
    # Remember to unwrap  of box images

    points -= np.mean(points, axis=0)
    G = np.zeros((3,3))
    for i,j in itertools.product(range(3),range(3)):
        G[i,j] = np.mean(points[:,i]*points[:,j])
    
    w,eigvec = scipy.linalg.eig(G)
    
    return w,eigvec



def fibonacci_sphere(samples=1):
    '''Generate points on a sphere using fibonacci algorithm
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
        
    return np.array(points)
