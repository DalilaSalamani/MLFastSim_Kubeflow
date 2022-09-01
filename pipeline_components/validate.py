from typing import NamedTuple


def validate(saved_gen: str, N_CELLS_Z: int, N_CELLS_R: int, N_CELLS_PHI: int,
             save_dir: str, max_energy: int, checkpoint_dir: str, init_dir: str,
             gen_dir: str, visual_dir: str, original_dim: int, valid_dir: str) -> NamedTuple('Variable_Details',
                                                                                             [('validate_data', str)]):
    SIZE_R = 2.325
    SIZE_Z = 3.4
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    save_dir = visual_dir
    plt.rcParams.update({"font.size": 22})

    def load_showers(init_dir, geo, energy_particle, angle_particle):
        dir_geo = init_dir + geo + "/"
        f_name = f"{geo}_angle_{angle_particle}.h5"
        f_name = dir_geo + f_name
        # read the HDF5 file
        h5 = h5py.File(f_name, "r")
        energies = np.array(h5[f"{energy_particle}"])
        return energies

    def prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry, bins, y_log_scale,
                                     hist_weight):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), clear=True, sharex="all")
        if hist_weight:
            axes[0].hist(np.arange(N_CELLS_Z), weights=g4, label="FullSim", bins=bins, alpha=0.4)
            axes[0].hist(np.arange(N_CELLS_Z), weights=vae, label="MLSim", bins=bins, alpha=0.4)
        else:
            axes[0].hist(g4, label="FullSim", bins=bins, alpha=0.4)
            axes[0].hist(vae, label="FullSim", bins=bins, alpha=0.4)
        if y_log_scale:
            axes[0].set_yscale("log")
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel("Energy [Mev]")
        axes[0].set_title(f" $e^-$ , {energy_particle} [GeV], {angle_particle}$^{{\circ}}$, {geometry} ")
        axes[1].plot(np.array(vae) / np.array(g4), "-o")
        axes[1].set_ylabel("MLSim/FullSim")
        axes[1].axhline(y=1, color="black")
        return fig, axes

    def longitudinal_profile(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True,
                             hist_weight=True):
        fig, axes = prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry,
                                                 np.linspace(0, N_CELLS_Z, N_CELLS_Z), y_log_scale, hist_weight)
        axes[0].set_xlabel("Layer index")
        axes[1].set_xlabel("Layer index")
        plt.savefig(f"{save_dir}LongProf_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    # lateral_profile function plots the lateral first moment comparing full and fast simulation data of a single geometry,
    # energy and angle of primary particles
    def lateral_profile(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True,
                        hist_weight=True):
        fig, axes = prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry,
                                                 np.linspace(0, N_CELLS_R - 1, N_CELLS_R), y_log_scale, hist_weight)
        axes[0].set_xlabel("r index")
        axes[1].set_xlabel("r index")
        plt.savefig(f"{save_dir}LatProf_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    # longitudinal_first_moment function plots the longitudinal profile comparing full and fast simulation data of a single
    # geometry, energy and angle of primary particles
    def longitudinal_first_moment(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True,
                                  hist_weight=False):
        fig, axes = prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry,
                                                 np.linspace(0, 0.4 * N_CELLS_Z * SIZE_Z, 128), y_log_scale,
                                                 hist_weight)
        axes[0].set_xlabel("$<\lambda>$ (mm)")
        axes[1].set_xlabel("$<\lambda>$ (mm)")
        plt.savefig(f"{save_dir}Long_First_Moment_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    # lateral_first_moment function plots the lateral first moment comparing full and fast simulation data of a single
    # geometry, energy and angle of primary particles
    def lateral_first_moment(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True,
                             hist_weight=False):
        fig, axes = prepare_customizable_profile(g4, vae, energy_particle, angle_particle, geometry,
                                                 np.linspace(0, 0.75 * N_CELLS_R * SIZE_R, 128), y_log_scale,
                                                 hist_weight)
        axes[0].set_xlabel("$<r>$ (mm)")
        axes[1].set_xlabel("$<r>$ (mm)")
        plt.savefig(f"{save_dir}LatFirstMoment_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    def e_tot(g4, vae, energy_particle, angle_particle, geometry, save_dir, y_log_scale=True):
        plt.figure(figsize=(12, 8))
        bins = np.linspace(np.min(g4), np.max(vae), 50)
        plt.hist(g4, histtype="step", label="FullSim", bins=bins, color="black")
        plt.hist(vae, histtype="step", label="MLSim", bins=bins, color="red")
        plt.legend()
        if y_log_scale:
            plt.yscale("log")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("# events")
        plt.savefig(f"{save_dir}E_tot_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    # Energy per layer distribution comparing full and fast simulation data of a single geometry, energy and angle of
    # primary particles
    def energy_layer(g4, vae, energy_particle, angle_particle, geometry, save_dir):
        fig, ax = plt.subplots(5, 9, figsize=(20, 20))
        cpt = 0
        for i in range(5):
            for j in range(9):
                g4_l = np.array([np.sum(i) for i in g4[:, :, :, i, j]])
                vae_l = np.array([np.sum(i) for i in vae[:, :, :, i, j]])
                bins = np.linspace(0, np.max(g4_l), 15)
                n_g4, bins_g4, _ = ax[i][j].hist(g4_l, histtype="step", label="FullSim", bins=bins, color="black")
                n_vae, bins_vae, _ = ax[i][j].hist(vae_l, histtype="step", label="FastSim", bins=bins, color="red")
                ax[i][j].set_title("Layer %s" % cpt, fontsize=12)
                cpt += 1
        plt.savefig(f"{save_dir}E_Layer_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    def cell_energy(g4, vae, energy_particle, angle_particle, geometry, save_dir):
        def log_energy(events, colour, label):
            all_log_en = []
            for ev in range(len(events)):
                energies = events[ev]
                for en in energies:
                    if en > 0:
                        all_log_en.append(np.log10(en))
                    else:
                        all_log_en.append(0)
            return plt.hist(all_log_en, bins=np.linspace(-4, 1, 1000), facecolor=colour, histtype="step", label=label)

        plt.figure(figsize=(12, 8))
        log_energy(g4, "b", "FullSim")
        log_energy(vae, "r", "FastSim")
        plt.xlabel("log10(E//MeV)")
        plt.ylim(bottom=1)
        plt.yscale("log")
        plt.ylim(bottom=1)
        plt.ylabel("# entries")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{save_dir}Cell_E_Dist_Log_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.png")
        plt.show()

    energy_particle = 1
    angle_particle = 50
    geometry = "SciPb"
    # 1. Full simulation data loading
    # Load energy of showers from a single geometry, energy and angle
    e_layer_g4 = load_showers(init_dir, geometry, energy_particle, angle_particle)
    valid_dir = valid_dir
    # 2. Fast simulation data loading, scaling to original energy range & reshaping
    vae_energies = np.loadtxt(
        f"{gen_dir}VAE_Generated_Geo_{geometry}_E_{energy_particle}_Angle_{angle_particle}.txt") * (
                               energy_particle * 1000)
    # Reshape the events into 3D
    e_layer_vae = vae_energies.reshape(len(vae_energies), N_CELLS_R, N_CELLS_PHI, N_CELLS_Z)
    # 3. Plot observables
    lp_g4 = []
    lp_vae = []
    tp_g4 = []
    tp_vae = []
    for i in range(N_CELLS_Z):
        lp_g4.append(np.sum(np.array([np.sum(i) for i in e_layer_g4[:, :, :, i]])))
        lp_vae.append(np.sum(np.array([np.sum(i) for i in e_layer_vae[:, :, :, i]])))
    for i in range(N_CELLS_R):
        tp_g4.append(np.sum(np.array([np.sum(i) for i in e_layer_g4[:, i, :, :]])))
        tp_vae.append(np.sum(np.array([np.sum(i) for i in e_layer_vae[:, i, :, :]])))
    longitudinal_profile(lp_g4, lp_vae, energy_particle, angle_particle, geometry, valid_dir)
    #     lateral_profile(tp_g4, tp_vae, energy_particle, angle_particle, geometry, valid_dir)
    g4 = e_layer_g4.reshape(len(e_layer_g4), 40500)
    vae = e_layer_vae.reshape(len(e_layer_vae), 40500)
    sum_g4 = np.array([np.sum(i) for i in g4])
    sum_vae = np.array([np.sum(i) for i in vae])
    e_tot(sum_g4, sum_vae, energy_particle, angle_particle, geometry, valid_dir)
    cell_energy(g4, vae, energy_particle, angle_particle, geometry, valid_dir)
    energy_layer(e_layer_g4.reshape(len(e_layer_g4), 18, 50, 5, 9), e_layer_vae.reshape(len(e_layer_vae), 18, 50, 5, 9),
                 energy_particle, angle_particle, geometry, valid_dir)
    z_ids = np.arange(N_CELLS_Z)
    r_ids = np.arange(N_CELLS_R)
    fml_g4 = []
    fml_vae = []
    fmt_g4 = []
    fmt_vae = []
    for s_id in range(len(e_layer_g4)):
        e_g4 = [np.sum(e_layer_g4[s_id, :, :, i]) for i in range(N_CELLS_Z)]
        fml_g4.append(np.sum([z_ids[i] * SIZE_Z * e_g4[i] for i in range(N_CELLS_Z)]) / sum_g4[s_id])
        e_vae = [np.sum(e_layer_vae[s_id, :, :, i]) for i in range(N_CELLS_Z)]
        fml_vae.append(np.sum([z_ids[i] * SIZE_Z * e_vae[i] for i in range(N_CELLS_Z)]) / sum_vae[s_id])
        e_g4 = [np.sum(e_layer_g4[s_id, i, :, :]) for i in range(N_CELLS_R)]
        fmt_g4.append(np.sum([r_ids[i] * SIZE_R * e_g4[i] for i in range(N_CELLS_R)]) / sum_g4[s_id])
        e_vae = [np.sum(e_layer_vae[s_id, i, :, :]) for i in range(N_CELLS_R)]
        fmt_vae.append(np.sum([r_ids[i] * SIZE_R * e_vae[i] for i in range(N_CELLS_R)]) / sum_vae[s_id])
    longitudinal_first_moment(fml_g4, fml_vae, energy_particle, angle_particle, geometry, valid_dir)
    lateral_first_moment(fmt_g4, fmt_vae, energy_particle, angle_particle, geometry, valid_dir)