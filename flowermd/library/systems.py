"""Examples for the Systems class."""

import mbuild as mb
import numpy as np
from scipy.spatial.distance import pdist

from flowermd.base.system import System


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
    """
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
        if not isinstance(density, u.array.unyt_quantity):
            self.density = density * u.Unit("g") / u.Unit("cm**3")
            warnings.warn(
                "Units for density were not given, assuming units of g/cm**3."
            )
        else:
            self.density = density
        self.packing_expand_factor = packing_expand_factor
        self.edge = edge
        self.overlap = overlap
        self.seed = seed
        self.unique_molecules = unique_molecules
        self.fix_orientation = fix_orientation
        super(Pack, self).__init__(
            molecules=molecules, base_units=base_units, **kwargs
        )

    def _build_system(self, **kwargs):
        mass_density = u.Unit("kg") / u.Unit("m**3")
        number_density = u.Unit("m**-3")
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
        if not self.unique_molecules and len(self._molecules) > 1:
            raise ValueError(
                f"unique_molecules kwarg was set to {self.unique_molecules}"
                ", which doesn't match the length of molecules given: "
                f"{len(self._molecules)} molecules"
            )
        if len(self._molecules) == 1:
            if not self.unique_molecules and len(self._molecules[0].n_mols) > 1:
                raise ValueError(
                    f"unique_molecules kwarg was set to {self.unique_molecules}"
                    ", which doesn't match the polydisperse system given: "
                    f"{self._molecules[0].n_mols}"
                )
        compound = self.all_molecules
        n_compounds = [1 for i in self.all_molecules]
        if not self.unique_molecules:
            compound = self.all_molecules[0]
            n_compounds = len(self.all_molecules)

        rng = np.random.default_rng(self.seed)

                N = #get N from self
                L = self.box_lengths[0]
            
                positions = np.empty((N, 3))
                starts = rng.uniform(0, L, size=(num_pol, 3))
            
                thetas = rng.uniform(0,2*np.pi,size=(num_pol,num_mon-1))
                phis = np.arccos(rng.uniform(-1,1,size=(num_pol,num_mon-1)))
                x = np.sin(phis)*np.cos(thetas)
                y = np.sin(phis)*np.sin(thetas)
                z = np.cos(phis)
            
                deltas = np.stack([x,y,z],axis=2) * bond_length
                displacements = np.cumsum(deltas, axis=1)
            
                positions_view = positions.reshape(num_pol, num_mon, 3)
                positions_view[:, 0, :] = starts
                positions_view[:, 1:, :] = starts[:, None, :] + displacements
            
                #pbc
                positions %= L
                positions -= L/2
            
                indices = np.arange(N).reshape(num_pol, num_mon)
                bonds = np.column_stack([
                    indices[:, :-1].ravel(),
                    indices[:, 1:].ravel()
                ])
            
                frame = gsd.hoomd.Frame()
                frame.particles.types = ['A']
                frame.particles.N = N
                frame.particles.position = positions
                frame.bonds.N = len(bonds)
                frame.bonds.group = bonds
                frame.bonds.types = ['b']
                frame.configuration.box = [L, L, L, 0, 0, 0]
            
                return frame
        return system


class PhantomWalk(System):
    """
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
        if not isinstance(density, u.array.unyt_quantity):
            self.density = density * u.Unit("g") / u.Unit("cm**3")
            warnings.warn(
                "Units for density were not given, assuming units of g/cm**3."
            )
        else:
            self.density = density
        self.packing_expand_factor = packing_expand_factor
        self.edge = edge
        self.overlap = overlap
        self.seed = seed
        self.unique_molecules = unique_molecules
        self.fix_orientation = fix_orientation
        super(Pack, self).__init__(
            molecules=molecules, base_units=base_units, **kwargs
        )

    def _build_system(self, **kwargs):
        #call RandomWalk
        system = RandomWalk(
            **kwargs
        )
        #parameterizes DPD FF
        system.apply_forcefield(r_cut=1.15, force_field=Bead_Spring_DPD())
        #run DPD simulation, calls DPD simulation class
        dpd_sim = Simulation.from_system(system=)
        #update system positions from DPD output
        
        return system

