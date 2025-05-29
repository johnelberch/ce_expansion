import collections.abc
import functools
from copy import deepcopy
from warnings import warn
from typing import Dict, Iterable, List, Optional

import ase
import ase.units
import numpy as np

from ce_expansion.atomgraph import adjacency
from ce_expansion.data import radii, ce_bulk

## NOTE FOR EDITING: 
# Things to change will have the following comment: TODO
# Suggestions and questions will have the following comment: QUESTION


def recursive_update(d: dict, u: dict) -> dict:
    """
    recursively updates 'dict of dicts'
    Ex)
    d = {0: {1: 2}}
    u = {0: {3: 4}, 8: 9}

    recursive_update(d, u) == {0: {1: 2, 3: 4}, 8: 9}

    Args:
    d (dict): the nested dict object to update
    u (dict): the nested dict that contains new key-value pairs

    Returns:
    d (dict): the final updated dict
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_cutoffs(atoms: ase.Atoms, mapping: Dict[str,str], x: float) -> List[float]:
    """
    Custom cutoffs from custom radii
    
    Args:
        atoms (ase.Atoms): The ASE atoms object representing the nanoparticle investigated.
        mapping (dict): Mapping dictionary with (key,value) pairs corresponding to (fake element, original element). e.g., {'Np' : 'Au'} for 'Np' atoms representing 'Au' atoms with adsorbates.
        x (float): Cutoff factor.
    
    Returns:
        (list) : List of radii 
    """

    # Radii modification like in the original code
    # QUESTION: Following Denny's implementation in NP. I multiplied the corresponding radii by x. BUT IT WAS NOT IN THE ORIGINAL (I assume it is a mistake that Denny overwrote in Canela_NP)
    radii['Au'] = 1.47
    radii['Pd'] = 1.38
    radii['Pt'] = 1.38
    
    # Update with user-provided mapping
    for k,v in mapping.items():
        radii[k] = radii[v]

    return [radii[atom_type]*x for atom_type in atoms.symbols]


class BCModelAds:
    def __init__(
            self, atoms: ase.Atoms, 
            gamma_values: Dict[str, Dict[str,float]], 
            bond_list: Optional[Iterable] = None, 
            CN_Method: str = "frac", 
            mapping: Optional[Dict[str,str]] = {},
        ):
        """
        Custom BCM class for calculations involving adsorbates

        Args:
            atoms (ase.Atoms): ASE atoms object which contains the data of the NP being tested      
            gamma_values (dict): Dictionary containing the gamma values for each atom pair. Outer dict keys correspond to element i. Inner dict keys correspond to element j. Inner dict value correspond to gamma_ij

        Kwargs:
            bond_list (Iterable): list of atom indices involved in each bond
            CN_Method (str):  Options "frac" or "int"
            mapping (dict): Mapping dictionary with (key,value) pairs corresponding to (fake element, original element). e.g., {'Np' : 'Au'} for 'Np' atoms representing 'Au' atoms with adsorbates.
        """
        
        # Basic assignments
        self.CN_Method = CN_Method 
        self.bond_list = bond_list
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.syms = atoms.symbols
        self.metal_types = sorted(set(atoms.symbols))
        self.gammas = gamma_values
        self.mapping = mapping
        self.radius = deepcopy(radii)
        for k,v in {'Au':1.47,'Pd':1.38,'Pt':1.38}.items():
            self.radius[k] = v #Custom radii defined in the original code (obtained from DFT)

        # Mapping checks and modifications
        if not mapping:
            warn("BCModelAds class was initiated, but no Mapping dictionary provided. Is this expected behavior?", UserWarning)

        if not all([e in self.metal_types for e in gamma_values.keys()]):
            raise KeyError("Some elements in the atoms object are not present in the Gamma values dictionary")
            
        if not all([{k,v} <= set(self.metal_types) for (k,v) in self.mapping.items()]):
            raise ValueError("Some elements in the mapping dictionary are not present in the atoms object")
        
        if not  all(e == len(gamma_values) for e in [len(L) for L in gamma_values.values()]):
             raise ValueError("The Gamma values dictionary is inconsistent. Make sure all inner dictionaries have the same length and include ALL elements")

        for k,v in self.mapping.items():
            self.radius[k] = radii[v]

        # Custom atoms object for connectivity purposes (safety guard, as some ASE connectivity functions use data dictionaries)
        self.connectivity_atoms = atoms.copy()
        symbols = np.array(self.connectivity_atoms.get_chemical_symbols())
        for k,v in mapping.items():
            symbols = np.char.replace(symbols, k, v)
        self.connectivity_atoms.set_chemical_symbols(symbols)

        # Define bond list
        #TODO: Check if having bond_list is really necessary. If not, merge with the if statement below
        #QUESTION: One idea is to modify the GA algorithm so it updates the bond list with simple slicing. That way, we can avoid calling adjacency.build_bonds_arr every time we need it?
        if self.bond_list is None:
            if CN_Method == 'frac':
                self.radii_bond_list = get_cutoffs(self.connectivity_atoms,self.mapping,1.2)
                self.bond_list = adjacency.build_bonds_arr(self.connectivity_atoms,self.radii_bond_list)
            else:
                self.bond_list = adjacency.build_bonds_arr(self.connectivity_atoms)

        # Values for precomps
        if CN_Method=='int':
            self.inv_radii = np.ones(len(self.metal_types))
            self.cn = np.bincount(self.bond_list[:, 0])
        elif CN_Method == 'frac':
            self.avg_radius =np.mean([self.radius[m] for m in self.syms]) 
            self.inv_radii = np.array([self.radius[m]/self.avg_radius for m in self.metal_types])
            self.cn = np.bincount(self.bond_list[:, 0])

        # get bonded atom columns
        self.a1 = self.bond_list[:, 0]
        self.a2 = self.bond_list[:, 1]

        # Setting precomputed values for quick calculations
        self.ce_bulk = None
        self.precomps = None
        self.cn_precomps = None

        self._get_bcm_params()
        self._get_precomps()


    def __len__(self) -> int:
        return len(self.atoms)


    ### FROM HERE
    def calc_ce(self, orderings: np.ndarray) -> float:
        """
        Calculates the Cohesive energy (in eV / atom) of the ordering given or of the default ordering of the NP

        [Cohesive Energy] = ( [precomp values of element A and B] / sqrt(12 * CN) ) / [num atoms]

        Args:
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        Cohesive Energy (eV / atom)
        """
        if self.CN_Method == 'int':
            return (self.precomps[orderings[self.a1], orderings[self.a2]] / self.cn_precomps).sum() / len(self.atoms)
        elif self.CN_Method == 'frac':
            return (self.precomps[orderings[self.a1], orderings[self.a2]] / self.cn_precomps[self.cn[self.a1], orderings[self.a1]]).sum() / len(self.atoms)


    def calc_ee(self, orderings: np.ndarray) -> float:
        """
        Calculates the Excess energy (in eV / atom) of the ordering given or of the default ordering of the NP

        [Excess Energy] = [CE of NP] - sum([Pure Element NP] * [Comp of Element in NP])

        Args:
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        Excess Energy (eV / atom)
        """

        metals = np.bincount(orderings)

        # obtain atom fractions of each tested element
        x_i = np.zeros(len(self.metal_types)).astype(float)
        x_i[:len(metals)] = metals / metals.sum()

        # calculate energy of tested NP first;
        ee = self.calc_ce(orderings)

        # Then, subtract calculated pure NP energies multiplied by respective
        # fractions to get Excess Energy
        for ele in range(len(self.metal_types)):
            x_ele = x_i[ele]
            o_mono_x = np.ones(len(self), int) * ele

            ee -= self.calc_ce(o_mono_x) * x_ele
        return ee


    def calc_smix(self, orderings: np.ndarray) -> float:
        """
        Uses boltzman constant, orderings, and element compositions to determine the smix of the nanoparticle

        Args:
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        entropy of mixing (smix)

        """

        x_i = np.bincount(orderings) / len(orderings)

        # drop 0s to avoid errors
        x_i = x_i[x_i != 0]

        kb = ase.units.kB

        smix = -kb * sum(x_i * np.log(x_i))

        return smix


    def calc_gmix(self, orderings: np.ndarray, T: float = 298.15) -> float:
        """
        gmix (eV / atom) = self.ee - T * self.calc_smix(ordering)

        Args:
        T: Temperature of the system in Kelvin; Defaults at room temp of 25 C
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        free energy of mixing (gmix)
        """
        return self.calc_ee(orderings) - T * self.calc_smix(orderings)


    def metropolis(self, ordering: np.ndarray, num_steps: int = 1000) -> None:
        """
        Metropolis-Hastings-based exploration of similar NPs

        Args:
        ordering: 1D chemical ordering array
        num_steps: How many steps to simulate for
        """
        # Initialization
        # create new instance of ordering array
        ordering = ordering.copy()
        best_ordering = ordering.copy()
        best_energy = self.calc_ce(ordering)
        prev_energy = best_energy
        energy_history = np.zeros(num_steps)
        energy_history[0] = best_energy

        ordering_indices = np.arange(len(ordering))
        for step in range(1, num_steps):
            prev_ordering = ordering.copy()
            i, j = np.random.choice(ordering_indices, 2, replace=False)
            ordering[i], ordering[j] = ordering[j], ordering[i]

            # Evaluate the energy change
            energy = self.calc_ce(ordering)

            # Metropolis-related stuff
            ratio = energy / prev_energy
            if ratio > np.random.uniform():
                # Commit to the step
                energy_history[step] = energy
                if energy < best_energy:
                    best_energy = energy
                    best_ordering = ordering.copy()
            else:
                # Reject the step
                ordering = prev_ordering.copy()
                energy_history[step] = prev_energy

        return best_ordering, best_energy, energy_history


    @functools.cached_property
    def num_shells(self) -> int:
        """
        Return number of shells in NP
        Use calc_shell_map if user did not define num_shells
        """
        return max(self.shell_map)


    @functools.cached_property
    def shell_map(self) -> Dict[int, Iterable[int]]:
        """
        Map of shell number and atom indices in shell

        0: core atom(s)
        1: shell (layer) 1 over core atom(s)
        etc.

        Returns:
            shell_map (dict): dict of shell number and array of atom indices in shell
        """
        remaining_atoms = set(range(len(self.atoms)))

        shell_map = {}
        cur_shell = 0
        srf = np.where(self.cn < 12)[0] #TODO What about non FCC elements in the core?
        shell_map[cur_shell] = srf
        remaining_atoms -= set(srf)
        coord_dict = {i: set(self.bond_list[self.bond_list[:, 0] == i].ravel())
                      for i in remaining_atoms}
        while remaining_atoms:
            cur_shell -= 1
            shell = [i for i in remaining_atoms
                     if coord_dict[i] - remaining_atoms]
            shell_map[cur_shell] = np.array(shell)
            remaining_atoms -= set(shell)

        shell_map = {k - cur_shell: v for k, v in shell_map.items()}
        return shell_map
    
    
    #NOTE: Modified these two already
    def _get_bcm_params(self) -> None:
        """
        Creates gamma and ce_bulk dictionaries which are then used
        to created precomputed values for the BCM calculation

        Sets:
            ce_bulk (dict): Bulk Cohesive energy values for the elements, accounting for the mapping {element_symbol : ce_bulk value}
        """

        ce_bulk_values = {}

        for M in self.metal_types:
            #Tries getting M from the mapping dictionary keys, if it is not there, use M
            M_use = self.mapping.get(M, M) 

            # Read from ce_bulk data in __init__.py
            ce_bulk_values[M] = ce_bulk[M_use]

        self.ce_bulk = ce_bulk_values

    def _get_precomps(self) -> None:
        """
        Uses the Gamma and ce_bulk dictionaries to create a precomputed
        BCM matrix of gammas and ce_bulk values

        [precomps] = [gamma of element 1] * [ce_bulk of element 1 to element 2]

        Sets:
            precomps (np.ndarray): Precomp Matrix (gamma x ce_bulk) of shape (len(metal_types), len(metal_types))
            cn_precomps (np.ndarray) Precomp Vector (CNi x CN_bulk) NOTE: CE_bulk = 12 by default 
        """
        # precompute values for BCM calc
        n_met = len(self.metal_types)
        precomps = np.ones((n_met, n_met))

        # General precomps
        for i, M1 in enumerate(self.metal_types):
            for j, M2 in enumerate(self.metal_types):
                precomp_bulk = self.ce_bulk[M1]
                precomp_gamma = self.gammas[M1][M2]
                precomps[i, j] = precomp_gamma * precomp_bulk
        self.precomps = precomps
        
        # QUESTION: Expand this functionality to include other CE_bulk values?
        # Coordination number precomp
        if self.CN_Method == 'int':
            self.cn_precomps = np.sqrt(self.cn * 12)[self.a1]
        elif self.CN_Method == 'frac':
            self.cn_precomps = np.sqrt((self.inv_radii * np.vstack(range(15))) * 12)
        
        
