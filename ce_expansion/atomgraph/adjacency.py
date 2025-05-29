#!/usr/bin/env python3

# Library for buiding bonding lists/tables/matrices
# James Dean, 2019

import os
import pathlib
from typing import Iterable, List

import numpy as np
import ase
import ase.neighborlist

# TODO: Clean and streamline this function
def get_nneighbors(n: ase.neighborlist.NeighborList):
    """
    Custom function to get number of neighbors (for now it is a copy pase of ASE, just that it doesn't have the annoying deprecation warning)
    """
    nneighbors = sum(indices.size for indices in n.nl.neighbors)
    if n.nl.self_interaction:
        nneighbors -= len(n.nl.neighbors)
    return nneighbors // 2 if n.nl.bothways else nneighbors

def build_bonds_arr(atoms: ase.Atoms, radii: Iterable[float] = None) -> np.ndarray:
    """
    Finds bonds between atoms based on bonding radii
    Default radii to determine bonding: covalent radii * 1.2

    Args:
    atoms: ase.Atoms object used to find bonding

    KArgs:
    radii: array of bonding radii for each atom in <atoms>.
           Defaults to covalent_radii * 1.2
    """
    # remove periodic boundaries
    atoms = atoms.copy()
    atoms.pbc = False

    # use default radius_arr = covalent_radii
    if radii is None:
        radii = ase.neighborlist.natural_cutoffs(atoms, 1.2)

    # create neighborlist object
    n = ase.neighborlist.NeighborList(radii, skin=0, self_interaction=False)
    n.update(atoms)
    # QUESTION: Check in depth this weird call
    if not n.nneighbors:
        return []
    nneighbors = get_nneighbors(n)

    bonds = np.zeros((nneighbors, 2), int)
    spot1 = 0
    for atomi in range(len(atoms)):
        # get neighbors of atomi
        neighs = n.get_neighbors(atomi)[0]

        # find second cutoff in matrix
        spot2 = spot1 + len(neighs)

        # add bonds to matrix
        bonds[spot1:spot2, 0] = atomi
        bonds[spot1:spot2, 1] = neighs

        # shift down matrix
        spot1 = spot2

        # once all bonds have been found break loop
        if spot1 == nneighbors:
            break

    return np.concatenate((bonds, bonds[:, ::-1]))


def build_adjacency_matrix(atoms: ase.Atoms, radii: Iterable[float] = None) -> np.ndarray:
    """
    Sparse matrix representation from an ase atoms object.

    Args:
    atoms: ase.Atoms object used to find bonding

    KArgs:
    radii: array of bonding radii for each atom in <atoms>.
           Defaults to covalent_radii * 1.2

    Returns:
    A numpy array representing the matrix of the ase object
    """
    # Construct the list of bonds
    bonds = build_bonds_arr(atoms, radii)

    # Generate the matrix
    adjacency_matrix = np.zeros((len(atoms), len(atoms))).astype(int)
    adjacency_matrix[bonds[:, 0], bonds[:, 1]] += 1

    return adjacency_matrix


def build_adjacency_list(atoms: ase.Atoms, radii: Iterable[float] = None) -> List[List]:
    """
    Adjacency list representation for an ase atoms object.
    [[atom indices bonded to atom 0], [... bonded to atom 1], ...]

    Args:
    atoms: ase.Atoms object used to find bonding

    KArgs:
    radii: array of bonding radii for each atom in <atoms>.
           Defaults to covalent_radii * 1.2

    Returns:
    the adjacency list of the ase atoms object

    """
    bonds = build_bonds_arr(atoms, radii)
    adj_list = [bonds[bonds[:, 0] == i][:, 1].tolist()
                for i in range(len(atoms))]
    return adj_list
 
if __name__ == '__main__':
    import ase.cluster

    nanoparticle = ase.cluster.Icosahedron('Cu', 2)
    adjacency_list = build_adjacency_list(nanoparticle)
    adjacency_matrix = build_adjacency_matrix(nanoparticle)
    bonds_list = build_bonds_arr(nanoparticle)
    print(bonds_list)
