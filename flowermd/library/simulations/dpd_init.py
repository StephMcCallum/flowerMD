"""DPD energy relaxation simulation class."""

import hoomd
from utils.dpd_utils import simulation_energy_end

from flowermd.base.simulation import Simulation


class DPDInit(Simulation):
    """ """

    def __init__(
        self,
        initial_state,
        forcefield,
        tensile_axis,
        fix_ratio=0.20,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        A=A,
        r=min_pair_dist,
        r_cut=r_cut,
        num_pol=num_pol,
        num_mon=num_mon,
        density=density,
    ):
        super(DPDInit, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
        )

        while not simulation_energy_end(
            A=A,
            r=min_pair_dist,
            r_cut=self.r_cut,
            num_pol=num_pol,
            num_mon=num_mon,
            density=density,
            log_file_name=self.log_file_name,
        ):
            self.run_NVE(n_steps=sim_steps_incr, kT=1.0, tau_kt=0.01)
            for writer in self.operations.writers:
                if hasattr(writer, "flush"):
                    writer.flush()
