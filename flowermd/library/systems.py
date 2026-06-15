"""Examples for the Systems class."""

import mbuild as mb
import numpy as np
from scipy.spatial.distance import pdist
import unyt as u

from flowermd.base.system import System
from flowermd.utils import (
    get_target_box_mass_density,
    get_target_box_number_density,
)


class SingleChainSystem(System):
    """Builds a vacuum box around a single chain.

    The box lengths are chosen so they are at least as long as the largest particle distance.
    The maximum distance of the chain is calculated using scipy.spatial.distance.pdist().
    This distance multiplied by a buffer defines the box dimensions. The chain is centered in the box.

    Parameters
    ----------
    buffer : float, default 1.05
        A factor to multiply box dimensions. Must be greater than 1 so that the particles are inside the box.

    """

    def __init__(self, molecules, base_units=dict(), buffer=1.05):
        self.buffer = buffer
        super(SingleChainSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        if len(self.all_molecules) > 1:
            raise ValueError(
                "This system class only works for systems contianing a single molecule."
            )
        chain = self.all_molecules[0]
        eucl_dist = pdist(self.all_molecules[0].xyz)
        chain_length = np.max(eucl_dist)
        box = mb.Box(lengths=np.array([chain_length] * 3) * self.buffer)
        comp = mb.Compound()
        comp.add(chain)
        comp.box = box
        chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
        return comp


class mbuildSystem(System):
    """Builds a system using mbuild box and mbuild positions.

    The box lengths and positions are read from the input mbuild compound. This is intended to be used with mbuild intialization methods,
    like translating polymer contiuents within the box, or a random walk cuboid constraint in mbuild 2.0.

    """

    def __init__(self, molecules, base_units=dict()):
        self.box_temp = molecules.box
        super(mbuildSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        chain = self.all_molecules
        comp = mb.Compound()
        comp.add(chain)
        comp.box = self.box_temp
        return comp

class RandomWalk(System):
    """Places molecules in the system using a vectorized random walk algorithm.
    
    Assumes all beads in each molecule are identical type (e.g., all "A" beads).
    All molecules are positioned in a single vectorized computation:
    - Each chain's first bead is placed at a random location within the box
    - Subsequent beads are placed at fixed bond length steps in random directions
    - All molecules are processed simultaneously using numpy broadcasting
    
    Parameters
    ----------
    molecules : Polymer or list of Polymer
        The molecule(s) to place in the system. All molecules must be identical.
    density : float or unyt_quantity
        The target density for the system (g/cm^3 or m^-3).
    base_units : dict, default {}
        Base units for the system.
    seed : int, default 12345
        Random seed for reproducibility.
    unique_molecules : bool, default True
        If True, each molecule in the list is treated as unique.
        If False, the single molecule is replicated.
    **kwargs
        Additional keyword arguments passed to System.
    
    Notes
    -----
    This implementation uses vectorized numpy operations to generate positions
    for all molecules at once, providing significant performance benefits
    compared to per-molecule iteration.
    """
    
    def __init__(
        self,
        molecules,
        density: float,
        base_units=dict(),
        seed=12345,
        unique_molecules=True,
        **kwargs,
    ):
        # Handle density units
        if not isinstance(density, u.array.unyt_quantity):
            self.density = density * u.Unit("g") / u.Unit("cm**3")
            warnings.warn(
                "Units for density were not given, assuming units of g/cm**3."
            )
        else:
            self.density = density
        
        self.seed = seed
        self.unique_molecules = unique_molecules
        self.bond_length =  molecules.bond_lengths['A-A']
        self.n_mols =  molecules.n_mols[0]
        self.lengths =  molecules.lengths[0]
        
        super(RandomWalk, self).__init__(
            molecules=molecules, base_units=base_units, **kwargs
        )
    
    def _build_system(self, **kwargs):
        """Build the system by placing molecules using random walk algorithm.
        
        Uses vectorized computation to generate all positions at once.
        """
        
        # Calculate target box based on density
        mass_density = u.Unit("kg") / u.Unit("m**3")
        number_density = u.Unit("nm**-3")
        
        if self.density.units.dimensions == mass_density.dimensions:
            target_box = get_target_box_mass_density(
                density=self.density, mass=self.mass
            ).to("nm")
        elif self.density.units.dimensions == number_density.dimensions:
            target_box = get_target_box_number_density(
                density=self.density, n_beads=self.n_particles
            ).to("nm")
        else:
            raise ValueError(
                f"Density dimensions of {self.density.units.dimensions} "
                "were given, but only mass density "
                f"({mass_density.dimensions}) and "
                f"number density ({number_density.dimensions}) are supported."
            )
        
        # Validate unique_molecules logic
        if not self.unique_molecules and len(self._molecules) > 1:
            raise ValueError(
                f"unique_molecules kwarg was set to {self.unique_molecules}, "
                "which doesn't match the length of molecules given: "
                f"{len(self._molecules)} molecules"
            )
        
        if len(self._molecules) == 1:
            if not self.unique_molecules and self._molecules[0].n_mols > 1:
                raise ValueError(
                    f"unique_molecules kwarg was set to {self.unique_molecules}, "
                    "which doesn't match the polydisperse system given: "
                    f"{self._molecules[0].n_mols}"
                )
        
        # Extract box lengths (magnitude removes units)
        box_lengths = target_box.to_value('nm')[0]
        
        # Initialize random number generator
        rng = np.random.default_rng(self.seed)
        
        # VECTORIZED: Generate all random walk positions at once
        # Shape: (num_molecules, beads_per_molecule, 3)
        all_positions = self._generate_all_random_walks_vectorized(
            num_molecules=self.n_mols,
            beads_per_molecule=self.lengths,
            bond_length=self.bond_length,
            box_lengths=box_lengths,
            rng=rng,
        )
        
        # Create system compound
        system = mb.Compound()
        
        # Apply positions to all molecules and add to system
        for mol_idx, molecule in enumerate(self.all_molecules): #TODO: fix this
            positions = all_positions[mol_idx]
            molecule.translate_to(positions)
            system.add(molecule)
        
        # Set system box from density calculation
        system.box = mb.box.Box(box_lengths)
        
        # Center system in its box
        center = np.array([
            system.box.Lx / 2,
            system.box.Ly / 2,
            system.box.Lz / 2
        ])
        system.translate_to(center)
        
        return system
    
    def _generate_all_random_walks_vectorized(
        self,
        num_molecules,
        beads_per_molecule,
        bond_length,
        box_lengths,
        rng,
    ):
        """Generate random walk positions for ALL identical molecules at once.
        
        Uses vectorized numpy operations for efficiency. Each molecule has
        identical structure (same bead type throughout), differing only in position.
        
        Parameters
        ----------
        num_molecules : int
            Number of molecules to generate positions for.
        beads_per_molecule : int
            Number of beads in each molecule.
        bond_length : float
            Bond length between consecutive beads (nm).
        box_lengths : np.ndarray
            The box dimensions [Lx, Ly, Lz] in nm.
        rng : np.random.Generator
            Random number generator instance.
        
        Returns
        -------
        np.ndarray
            Array of shape (num_molecules, beads_per_molecule, 3) with all
            positions in nm.
        """
        # Initialize positions array: (num_molecules, beads_per_molecule, 3)
        positions = np.empty((num_molecules, beads_per_molecule, 3))
        
        # STEP 1: First bead for all molecules - random positions
        # Shape: (num_molecules, 3)
        positions[:, 0, :] = rng.uniform(0, box_lengths, size=(num_molecules, 3))
        
        # STEP 2: Subsequent beads - vectorized random walk
        # All beads are identical type, so bond_length is constant
        for i in range(1, beads_per_molecule):
            # Generate random directions for ALL molecules at once
            # theta: shape (num_molecules,)
            # phi: shape (num_molecules,)
            theta = rng.uniform(0, 2 * np.pi, size=num_molecules)
            phi = np.arccos(rng.uniform(-1, 1, size=num_molecules))
            
            # Vectorized direction calculation
            # Result shape: (num_molecules, 3)
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ]).T  # Transpose to (num_molecules, 3)
            
            # Apply to all molecules: each gets bond_length step in random direction
            # Broadcasting: (num_molecules, 3) + (num_molecules, 3)
            positions[:, i, :] = positions[:, i - 1, :] + bond_length * direction
        
        # STEP 3: Apply periodic boundary conditions
        # Individual beads can wrap across boundaries
        positions %= box_lengths
        print(positions)
        
        return positions

