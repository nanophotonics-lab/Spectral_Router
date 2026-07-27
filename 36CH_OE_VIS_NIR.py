# ==========================================================================
# Evaluation script for the VIS--NIR 36-channel (6x6) inverse-designed
# spectral router.
#
# Loads the optimized binary TiO2/SiO2 design (final_design.txt) and evaluates:
#   (1) Per-channel routing-efficiency spectra T_ii(lambda), defined as the
#       z-directed Poynting flux through the target sub-pixel region divided
#       by the incident power (normalization run below).
#       -> ./ucell/OE_abs.png, ./ucell/OE_abs_data.csv
#       The per-channel routing efficiency is T_ii evaluated at the target
#       wavelength lambda_i (linear interpolation of the CSV columns), and
#       the average routing efficiency is (1/N) * sum_i T_ii(lambda_i).
#   (2) Focal-plane intensity (Sz) maps at each target wavelength, with each
#       channel map normalized independently for visualization.
#       -> ./E2_plot/  (montage panels)
#
# Target wavelengths: 36 values equally spaced between 404 and 996 nm
# (channel spacing ~16.9 nm), i.e. np.linspace(0.404, 0.996, 36) um,
# assigned to the 6x6 sub-pixel array in row-major order from shorter to
# longer wavelengths.
#
# Conditions: sub-pixel size P = 0.60 um, device thickness h = 2.0 um,
# FDTD resolution 50 pixels/um, normal incidence, periodic boundaries in
# x,y and PML in z. The evaluation band spans 400--1000 nm.
# ==========================================================================
# ## 1. Simulation Environment

import sys
import meep as mp
import cmath
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import scipy
import meep.adjoint as mpa
import os
from mpi4py import MPI
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import csv
import os



# CHOOSE #######################

OE        = True
INTENSITY = True

################################



if True: # Shut-up Malfoy ##############################################################################################
    mp.verbosity(1)


if True: # VIS-NIR 36-channel wavelength setting ############################################################################################################

    # Target wavelengths:
    #   Keep the 36 target wavelengths used by the optimization code as-is.
    #   If the design targets 404--996 nm, do not change the two values below.
    #
    # Evaluation spectrum:
    #   OE_abs/OE/RTL plots and CSV are computed over the full 400--1000 nm range.
    NUM_CHANNELS = 36

    WL_TARGET_MIN_UM = 0.404
    WL_TARGET_MAX_UM = 0.996

    WL_EVAL_MIN_UM = 0.400
    WL_EVAL_MAX_UM = 1.000
    N_FREQ_EVAL = 1000

    wavelengths_list = np.linspace(WL_TARGET_MIN_UM, WL_TARGET_MAX_UM, NUM_CHANNELS)
    frequency_list = 1.0 / wavelengths_list

    # Evaluation frequency range for add_flux / GaussianSource
    eval_freq_min = 1.0 / WL_EVAL_MAX_UM
    eval_freq_max = 1.0 / WL_EVAL_MIN_UM
    freq_center = 0.5 * (eval_freq_min + eval_freq_max)
    freq_span = eval_freq_max - eval_freq_min
    source_fwidth = 1.2 * freq_span

    OUTPUT_TAG = (
        f"VIS_NIR_target_{int(WL_TARGET_MIN_UM * 1000)}_{int(WL_TARGET_MAX_UM * 1000)}nm"
        f"_eval_{int(WL_EVAL_MIN_UM * 1000)}_{int(WL_EVAL_MAX_UM * 1000)}nm"
    )
    OE_DIR = f"./ucell"
    INTENSITY_DIR = f"./E2_plot"

    # Also check the legacy output folder name as a candidate.
    OPT_SAVE_DIR_CANDIDATES = [
        f"./36ch_hyper_TiO2_VIS_NIR_{int(WL_TARGET_MIN_UM * 1000)}_{int(WL_TARGET_MAX_UM * 1000)}nm",
        f"./36ch_hyper_TiO2_VIS_NIR_{int(WL_EVAL_MIN_UM * 1000)}_{int(WL_EVAL_MAX_UM * 1000)}nm",
        f"./36ch_hyper_TiO2_{int(WL_TARGET_MIN_UM * 1000)}_{int(WL_TARGET_MAX_UM * 1000)}nm",
    ]

    if NUM_CHANNELS != 36:
        raise ValueError("This verification script currently assumes exactly 36 channels.")

    if mp.am_master():
        print("===================================================")
        print("OE Start: VIS-NIR 36-channel verification")
        print(f"Target wavelength range = {WL_TARGET_MIN_UM * 1000:.1f}--{WL_TARGET_MAX_UM * 1000:.1f} nm")
        print(f"Evaluation range        = {WL_EVAL_MIN_UM * 1000:.1f}--{WL_EVAL_MAX_UM * 1000:.1f} nm")
        print(f"Channel spacing         = {(wavelengths_list[1] - wavelengths_list[0]) * 1000:.2f} nm")

        for i, wl in enumerate(wavelengths_list):
            print(f"Cell {i:02d} = {wl * 1000:.2f} nm")

        print("===================================================")


def make_channel_band_edges(wavelengths, wl_min, wl_max):
    """Generate band-mask edges around each target wavelength."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    edges = np.empty(len(wavelengths) + 1)
    edges[0] = wl_min
    edges[-1] = wl_max
    edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
    return edges

def _interpolate_rgb(c0, c1, t):
    c0 = np.asarray(mcolors.to_rgb(c0), dtype=float)
    c1 = np.asarray(mcolors.to_rgb(c1), dtype=float)
    return (1.0 - t) * c0 + t * c1


def _visible_wavelength_rgb(wl_nm, gamma=0.8):
    """Approximate visible-wavelength RGB for 380--700 nm."""
    wl = float(wl_nm)

    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wl <= 700:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0

    # Edge attenuation: keeps violet/deep-red ends from becoming overly bright.
    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 420 <= wl <= 645:
        factor = 1.0
    elif 645 < wl <= 700:
        factor = 0.3 + 0.7 * (700 - wl) / (700 - 645)
    else:
        factor = 0.0

    rgb = np.array([r, g, b]) * factor
    rgb = np.clip(rgb, 0.0, 1.0) ** gamma
    return rgb


def _nir_false_color_rgb(wl_nm, wl_start_nm=700.0, wl_end_nm=1000.0):
    """False-color map for NIR wavelengths, since NIR is not visible.

    700 nm starts from deep red and gradually moves through brown/purple
    toward dark gray at 1000 nm. This is only for plotting distinction.
    """
    t = np.clip((float(wl_nm) - wl_start_nm) / (wl_end_nm - wl_start_nm), 0.0, 1.0)
    anchors = [
        (0.00, "#A00000"),  # just beyond red edge
        (0.35, "#7A2E00"),  # dark amber/brown
        (0.70, "#542788"),  # purple false color
        (1.00, "#404040"),  # long-NIR dark gray
    ]
    for (t0, c0), (t1, c1) in zip(anchors[:-1], anchors[1:]):
        if t0 <= t <= t1:
            return _interpolate_rgb(c0, c1, (t - t0) / (t1 - t0))
    return np.asarray(mcolors.to_rgb(anchors[-1][1]), dtype=float)


def _blend_with_white(hex_color, amount=0.68):
    rgb = np.asarray(mcolors.to_rgb(hex_color), dtype=float)
    pastel = (1.0 - amount) * rgb + amount * np.ones(3)
    return mcolors.to_hex(np.clip(pastel, 0.0, 1.0))


def make_vis_nir_colors(wavelengths_um):
    """Return line/fill colors matched to 400--1000 nm VIS--NIR channels.

    Visible channels use approximate physical colors. NIR channels use false
    colors because they are outside the human-visible range.
    """
    line_colors = []
    fill_colors = []

    for wl_um in wavelengths_um:
        wl_nm = float(wl_um) * 1000.0
        if wl_nm <= 700.0:
            rgb = _visible_wavelength_rgb(wl_nm)
        else:
            rgb = _nir_false_color_rgb(wl_nm)

        line_hex = mcolors.to_hex(np.clip(rgb, 0.0, 1.0))
        line_colors.append(line_hex)
        fill_colors.append(_blend_with_white(line_hex, amount=0.70))

    return line_colors, fill_colors


if True: # Parameters ##################################################################################################
    
    Air  = mp.Medium(index=1.0)
    SiO2 = mp.Medium(index=1.45)
    SiN  = mp.Medium(index=2.10)
    TiO2 = mp.Medium(index=2.65)
    Si   = mp.Medium(index=4.0)

    # Resolution
    resolution = 50
    
    design_region_x = 3.6
    design_region_y = 3.6
    design_region_z = 2.0

    DTI_size  = 0.12
    PD_size   = 0.48
    PD_height = 0.80

    pml_2_src = 0.20
    src_2_geo = 0.20
    mon_2_pml = 0.40

    Lpml       = 0.40
    pml_layers = [mp.PML(thickness = Lpml, direction = mp.Z)]

    Sx = round(design_region_x, 3)
    Sy = round(design_region_y, 3)
    Sz = round(Lpml + pml_2_src + src_2_geo + design_region_z + PD_height, 3) # PD_height includes the bottom PML
    cell_size = mp.Vector3(Sx, Sy, Sz)

    # Design Region
    design_region_resolution = int(50)
    Nx = int(design_region_resolution * design_region_x) + 1
    Ny = int(design_region_resolution * design_region_y) + 1
    Nz = int(design_region_resolution * design_region_z) + 1

    # Source
    source_center = mp.Vector3(0, 0, round(Sz/2 - Lpml - pml_2_src, 3))
    source_size   = mp.Vector3(Sx, Sy, 0)

    # VIS-NIR broadband source. Evaluation range is set by freq_center/freq_span above.
    src       = mp.GaussianSource(frequency=freq_center, fwidth=source_fwidth, is_integrated=True)
    source    = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),
                 mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]

if True: # Simulation Environment

    structure_path_candidates = ["final_design.txt"]
    structure_path_candidates += [
        os.path.join(d, "final_design.txt") for d in OPT_SAVE_DIR_CANDIDATES
    ]

    STRUCTURE_FILE = None
    for candidate in structure_path_candidates:
        if os.path.exists(candidate):
            STRUCTURE_FILE = candidate
            break

    if STRUCTURE_FILE is None:
        raise FileNotFoundError(
            "final_design.txt not found. Put it in the run directory or one of:\n"
            + "\n".join(structure_path_candidates[1:])
        )

    if mp.am_master():
        print(f"[INFO] Load structure from: {STRUCTURE_FILE}")

    structure_weight = np.loadtxt(STRUCTURE_FILE)
    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), SiO2, TiO2, grid_type="U_MEAN", do_averaging=False)
    design_variables.update_weights(structure_weight.reshape(Nx, Ny, Nz))
    design_region = mpa.DesignRegion(design_variables,
                                    volume=mp.Volume(
                                        center=mp.Vector3(0, 0, round(Sz/2 - Lpml -  pml_2_src - src_2_geo - design_region_z/2, 3)),
                                        size=mp.Vector3(design_region_x, design_region_y, design_region_z),
                                        )
                                    )

    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, -Sz/2 + PD_height/2), size=mp.Vector3(Sx, Sy, PD_height), material=TiO2
        ),
    ]

    geometry.append(mp.Block(center=design_region.center,
                            size=design_region.size,
                            material=design_variables))

    
    # Simulation
    sim = mp.Simulation(
        cell_size=cell_size, 
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        default_material=Air,
        resolution=resolution,
        k_point = mp.Vector3(0,0,0), # Bloch B.C.
        eps_averaging= True, 
        extra_materials=[SiO2, SiN, TiO2, Si],
    )


    wavelengths = np.linspace(WL_EVAL_MIN_UM, WL_EVAL_MAX_UM, N_FREQ_EVAL)
    frequencies = np.sort(1.0 / wavelengths)
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[],
        objective_arguments=[],
        design_regions=design_region,
        frequencies=frequencies,
        decay_by=1e-3,
    )


if OE:
    # Save directory
    if mp.am_master():
        
        print("------------------------------------------------------")
        print("Compute OE...")
        print("------------------------------------------------------")
        
        os.makedirs(OE_DIR, exist_ok=True)


        opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)),
                source_parameters={'alpha':1}, monitor_parameters={'alpha':1},
                )
        plt.xlabel("Width (μm)")
        plt.ylabel("Height (μm)")
        plt.savefig(f"{OE_DIR}/Design_XZ.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
        

        opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, np.inf, 0), center = (0,0,Sz/2-Lpml-pml_2_src-src_2_geo)))
        plt.xlabel("X (μm)")
        plt.ylabel("Y (μm)")
        plt.savefig(f"{OE_DIR}/Design_XY.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
        w3d_bin = (structure_weight.reshape(Nx, Ny, Nz) > 0.5).astype(int)

        _SIO2 = np.array([200, 200, 200]) / 255      # light gray
        _TIO2 = np.array([ 90,  90,  90]) / 255      # dark gray

        def _face_colors(slice2d, brightness):
            s = slice2d[:-1, :-1]
            rgb = np.where(s[..., None] == 1, _TIO2, _SIO2) * brightness
            return np.concatenate([rgb, np.ones((*s.shape, 1))], axis=-1)

        _x = np.linspace(0, design_region_x, Nx)
        _y = np.linspace(0, design_region_y, Ny)
        _z = np.linspace(0, design_region_z, Nz)

        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection='3d')

        # top face (illumination side = z index Nz-1)
        _X, _Y = np.meshgrid(_x, _y, indexing='ij')
        ax.plot_surface(_X, _Y, np.full_like(_X, design_region_z),
                        facecolors=_face_colors(w3d_bin[:, :, -1], 1.00),
                        rstride=1, cstride=1, shade=False)

        # side face x = xmax (YZ boundary slice)
        _Yf, _Zf = np.meshgrid(_y, _z, indexing='ij')
        ax.plot_surface(np.full_like(_Yf, design_region_x), _Yf, _Zf,
                        facecolors=_face_colors(w3d_bin[-1, :, :], 0.85),
                        rstride=1, cstride=1, shade=False)

        # side face y = 0 (XZ boundary slice)
        _Xf, _Zf = np.meshgrid(_x, _z, indexing='ij')
        ax.plot_surface(_Xf, np.zeros_like(_Xf), _Zf,
                        facecolors=_face_colors(w3d_bin[:, 0, :], 0.70),
                        rstride=1, cstride=1, shade=False)

        ax.set_box_aspect((design_region_x, design_region_y, design_region_z))
        ax.set_axis_off()
        ax.view_init(elev=18, azim=-55)
        plt.tight_layout(pad=0)
        plt.savefig("./ucell/Design_3D.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[3D] Design_3D.png saved")
        # ==========================================================================
        
        
        
if INTENSITY:
    # Save directory
    if mp.am_master():
        
        print("------------------------------------------------------")
        print("Intensity Plot...")
        print("------------------------------------------------------")
        
        os.makedirs(INTENSITY_DIR, exist_ok=True)
    


if True: # Angle ############################################################################################################################################################
    
    t = 0
    phi_deg   = 0
    theta_deg = 0


if OE: # OE Plot ###################################################################################################################################################################

    phi = math.radians(phi_deg)
    phi_str = f"{int(phi_deg)}deg"

    theta = math.radians(theta_deg)
    theta_str = f"{int(theta_deg)}deg"


    if mp.am_master(): # Save directory
        print("------------------------------------------------------")
        print("Compute OE...")
        print("------------------------------------------------------")
        save_dir_str = OE_DIR
        os.makedirs(save_dir_str, exist_ok=True)

    
    k = mp.Vector3(0, 0, 0) 

    # NORMALIZATION ###################################################################################################################################################################################
    geometry_1 = [
        mp.Block(
            center=mp.Vector3(0, 0, 0), size=mp.Vector3(Sx, Sy, Sz), material=Air
        )
    ]

    opt.sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry_1,
        sources=source,
        default_material=Air,
        resolution=resolution,
        k_point=k
    )
    
    # Target frequency range for VIS-NIR OE spectrum
    fcen  = freq_center
    df    = freq_span
    nfreq = N_FREQ_EVAL

    src    = mp.GaussianSource(frequency=fcen, fwidth=source_fwidth, is_integrated=True) 
    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),
              mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
    opt.sim.change_sources(source)


    # reflection moniter
    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, round(Sz/2 - Lpml - 1/resolution, 3)),
                            size=mp.Vector3(Sx, Sy, 0),) 
    refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

    # transmission moiniter
    tran_t = mp.FluxRegion(center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution, 3)),
                            size=mp.Vector3(Sx, Sy, 0),)
    tran_total = opt.sim.add_flux(fcen, df, nfreq, tran_t)
    
    # pt
    pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution, 3))

    
    # Run Sim_NORM
    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-7))

    #Save data
    straight_refl_data = opt.sim.get_flux_data(refl)
    total_flux = mp.get_fluxes(tran_total)
    flux_freqs = mp.get_flux_freqs(tran_total)
    opt.sim.reset_meep()

    # Forward Simulation ##############################################################################################################################
    opt.sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        default_material=Air,
        resolution=resolution,
        k_point = k,
        eps_averaging=True,
        extra_materials=[SiO2, SiN, TiO2, Si],
    )


    # Compute reflected flux
    refl = opt.sim.add_flux(fcen, df, nfreq, refl_fr)

    # Compute transmitted flux
    tran = opt.sim.add_flux(fcen, df, nfreq, tran_t)

    # Subtract reflected flux
    opt.sim.load_minus_flux_data(refl, straight_refl_data)
    
    # Compute flux entering the pixel
    tran_p = mp.FluxRegion(center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution, 3)),
                            size=mp.Vector3(Sx, Sy, 0),)
    tran_pixel = opt.sim.add_flux(fcen, df, nfreq, tran_p)

    #Compute flux for each pixel
    monitor_size = mp.Vector3(PD_size, PD_size, 0)
    
    FluxRegion_00 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 5/12, +design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_01 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 3/12, +design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_02 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/12, +design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_03 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/12, +design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_04 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 3/12, +design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_05 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 5/12, +design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_06 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 5/12, +design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_07 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 3/12, +design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_08 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/12, +design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_09 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/12, +design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_10 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 3/12, +design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_11 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 5/12, +design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_12 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 5/12, +design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_13 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 3/12, +design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_14 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/12, +design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_15 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/12, +design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_16 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 3/12, +design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_17 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 5/12, +design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_18 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 5/12, -design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_19 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 3/12, -design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_20 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/12, -design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_21 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/12, -design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_22 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 3/12, -design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_23 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 5/12, -design_region_y * 1/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_24 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 5/12, -design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_25 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 3/12, -design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_26 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/12, -design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_27 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/12, -design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_28 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 3/12, -design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_29 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 5/12, -design_region_y * 3/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_30 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 5/12, -design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_31 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 3/12, -design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_32 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/12, -design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_33 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/12, -design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_34 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 3/12, -design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_35 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 5/12, -design_region_y * 5/12, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    
    #Compute flux for each pixel
    
    
    tran_00 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_00)
    tran_01 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_01)
    tran_02 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_02)
    tran_03 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_03)
    tran_04 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_04)
    tran_05 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_05)
    
    tran_06 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_06)
    tran_07 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_07)
    tran_08 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_08)
    tran_09 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_09)
    tran_10 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_10)
    tran_11 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_11)
    
    tran_12 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_12)
    tran_13 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_13)
    tran_14 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_14)
    tran_15 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_15)
    tran_16 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_16)
    tran_17 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_17)
    
    tran_18 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_18)
    tran_19 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_19)
    tran_20 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_20)
    tran_21 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_21)
    tran_22 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_22)
    tran_23 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_23)
    
    tran_24 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_24)
    tran_25 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_25)
    tran_26 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_26)
    tran_27 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_27)
    tran_28 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_28)
    tran_29 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_29)
    
    tran_30 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_30)
    tran_31 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_31)
    tran_32 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_32)
    tran_33 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_33)
    tran_34 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_34)
    tran_35 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_35)
    

    # Run Simulation
    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-7))

    # Save data
    refl_flux = mp.get_fluxes(refl)
    tran_flux = mp.get_fluxes(tran)
    tran_flux_p = mp.get_fluxes(tran_pixel)
    
    fluxes_00 = mp.get_fluxes(tran_00)
    fluxes_01 = mp.get_fluxes(tran_01)
    fluxes_02 = mp.get_fluxes(tran_02)
    fluxes_03 = mp.get_fluxes(tran_03)
    fluxes_04 = mp.get_fluxes(tran_04)
    fluxes_05 = mp.get_fluxes(tran_05)
    fluxes_06 = mp.get_fluxes(tran_06)
    fluxes_07 = mp.get_fluxes(tran_07)
    fluxes_08 = mp.get_fluxes(tran_08)
    fluxes_09 = mp.get_fluxes(tran_09)
    fluxes_10 = mp.get_fluxes(tran_10)
    fluxes_11 = mp.get_fluxes(tran_11)
    fluxes_12 = mp.get_fluxes(tran_12)
    fluxes_13 = mp.get_fluxes(tran_13)
    fluxes_14 = mp.get_fluxes(tran_14)
    fluxes_15 = mp.get_fluxes(tran_15)
    fluxes_16 = mp.get_fluxes(tran_16)
    fluxes_17 = mp.get_fluxes(tran_17)
    fluxes_18 = mp.get_fluxes(tran_18)
    fluxes_19 = mp.get_fluxes(tran_19)
    fluxes_20 = mp.get_fluxes(tran_20)
    fluxes_21 = mp.get_fluxes(tran_21)
    fluxes_22 = mp.get_fluxes(tran_22)
    fluxes_23 = mp.get_fluxes(tran_23)
    fluxes_24 = mp.get_fluxes(tran_24)
    fluxes_25 = mp.get_fluxes(tran_25)
    fluxes_26 = mp.get_fluxes(tran_26)
    fluxes_27 = mp.get_fluxes(tran_27)
    fluxes_28 = mp.get_fluxes(tran_28)
    fluxes_29 = mp.get_fluxes(tran_29)
    fluxes_30 = mp.get_fluxes(tran_30)
    fluxes_31 = mp.get_fluxes(tran_31)
    fluxes_32 = mp.get_fluxes(tran_32)
    fluxes_33 = mp.get_fluxes(tran_33)
    fluxes_34 = mp.get_fluxes(tran_34)
    fluxes_35 = mp.get_fluxes(tran_35)

    fluxes_list = [
        fluxes_00, fluxes_01, fluxes_02, fluxes_03, fluxes_04, fluxes_05,
        fluxes_06, fluxes_07, fluxes_08, fluxes_09, fluxes_10, fluxes_11,
        fluxes_12, fluxes_13, fluxes_14, fluxes_15, fluxes_16, fluxes_17,
        fluxes_18, fluxes_19, fluxes_20, fluxes_21, fluxes_22, fluxes_23,
        fluxes_24, fluxes_25, fluxes_26, fluxes_27, fluxes_28, fluxes_29,
        fluxes_30, fluxes_31, fluxes_32, fluxes_33, fluxes_34, fluxes_35,
    ]
    
    # Efficiency relative to total flux (Abs OE)
    # Meep flux frequencies may be in increasing-frequency order,
    # so sort by increasing wavelength (400 -> 1000 nm) before saving/plotting.
    wl = 1.0 / np.asarray(flux_freqs)
    wl_sort_idx = np.argsort(wl)
    wl = wl[wl_sort_idx]

    total_flux_arr = np.asarray(total_flux)[wl_sort_idx]
    tran_flux_arr = np.asarray(tran_flux)[wl_sort_idx]
    tran_flux_p_arr = np.asarray(tran_flux_p)[wl_sort_idx]
    refl_flux_arr = np.asarray(refl_flux)[wl_sort_idx]
    fluxes_arr_list = [np.asarray(f)[wl_sort_idx] for f in fluxes_list]

    T_list = [fluxes_arr_list[i] / total_flux_arr for i in range(36)]

    ######################################################################################################
    line_colors, fill_colors = make_vis_nir_colors(wavelengths_list)

    # fill_colors is generated by make_vis_nir_colors().

    ######################################################################################################
    
    # Band edges: auto-set the interval around each target wavelength
    band_edges = make_channel_band_edges(wavelengths_list, WL_EVAL_MIN_UM, WL_EVAL_MAX_UM)


    # Masks
    band_masks = []
    for i in range(36):
        mask = (wl >= band_edges[i]) & (wl <= band_edges[i+1])
        band_masks.append(mask)
    
    ##############################################################################################################
    
    # -------------------------------
    # Tb / OE_abs Plot
    # -------------------------------
    if mp.am_master():
        plt.figure(dpi=150)

        # light lines
        for i in range(36):
            plt.plot(wl, T_list[i], color=line_colors[i], alpha=0.4)

        # axis
        plt.axis([WL_EVAL_MIN_UM, WL_EVAL_MAX_UM, 0, 1])
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Efficiency")


        # Fill bands (light shading)
        for i in range(36):
            x = [band_edges[i], band_edges[i], band_edges[i+1], band_edges[i+1]]
            y = [-0.03, 1.03, 1.03, -0.03]
            plt.fill(x, y, color=fill_colors[i], alpha=0.4)

        # Bold lines (each band)
        for i in range(36):
            plt.plot(wl[band_masks[i]], T_list[i][band_masks[i]], color=line_colors[i], alpha=1)

        # Save
        plt.savefig(f"{save_dir_str}/OE_abs.png")
        plt.cla()
        plt.clf()
        plt.close()
        
        
        # CSV
        csv_path = f"{save_dir_str}/OE_abs_data.csv"

        # CSV header
        header = "wl," + ",".join([f"T{i:02d}" for i in range(36)])

        with open(csv_path, "w") as f:
            f.write(header + "\n")

            # For each wavelength index
            for idx in range(len(wl)):
                row = [f"{wl[idx]:.8f}"]  # wl value with good precision

                # Append T00 ~ T35
                for i in range(36):
                    row.append(f"{T_list[i][idx]:.8f}")

                # Write row
                f.write(",".join(row) + "\n")



if INTENSITY:

    
    phi = math.radians(phi_deg)
    phi_str = f"{int(phi_deg)}deg"

    
    theta = math.radians(theta_deg)
    theta_str = f"{int(theta_deg)}deg"


    if mp.am_master(): # Save directory
        print("------------------------------------------------------")
        print("Intensity Plot...")
        print("------------------------------------------------------")
        save_dir_str = INTENSITY_DIR
        os.makedirs(save_dir_str, exist_ok=True)
    
    
    frequency_list
        

    fcen   = freq_center
    fwidth = source_fwidth
    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True)

    source_center = mp.Vector3(0, 0, round(Sz/2 - Lpml - pml_2_src, 3))
    source_size   = mp.Vector3(Sx, Sy, 0)
    
    def pw_amp(k,x0):
        def _pw_amp(x):
            return cmath.exp(1j*2*math.pi*k.dot(x+x0))
        return _pw_amp


    def wavevector(frequency, theta_src, phi_src):
        kx = math.sin(theta_src) * math.cos(phi_src) * frequency
        ky = math.sin(theta_src) * math.sin(phi_src) * frequency
        kz = math.cos(theta_src) * frequency
        return mp.Vector3(kx, ky, kz)

    k = mp.Vector3(0,0,0)

    sources = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amp_func=pw_amp(k, source_center))]
    
    line_colors, _ = make_vis_nir_colors(wavelengths_list)
    
    # cmap creation
    cmap_dict = {}

    for i in range(36):
        cmap_dict[f"B{i:02d}"] = mcolors.LinearSegmentedColormap.from_list(
            f"B{i:02d}_cmap",
            ["black", line_colors[i], "white"]
        )

    # -------------------------------
    # Plotting
    # -------------------------------

    # Forward Simulation
    sim_RGB = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=Air,
        resolution=resolution,
        k_point = k,
        eps_averaging=True,
        extra_materials=[SiO2, SiN, TiO2, Si],
    )
    
    Ex_fields, Ey_fields, Hx_fields, Hy_fields = [], [], [], []

    mon_center = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution, 3))
    mon_size   = mp.Vector3(Sx, Sy, 0)
    for freq in frequency_list:
        Ex_fields.append(sim_RGB.add_dft_fields([mp.Ex], freq, 0, 1,
                                            center=mon_center, size=mon_size,
                                            yee_grid=False))
        Ey_fields.append(sim_RGB.add_dft_fields([mp.Ey], freq, 0, 1,
                                            center=mon_center, size=mon_size,
                                            yee_grid=False))
        Hx_fields.append(sim_RGB.add_dft_fields([mp.Hx], freq, 0, 1,
                                            center=mon_center, size=mon_size,
                                            yee_grid=False))
        Hy_fields.append(sim_RGB.add_dft_fields([mp.Hy], freq, 0, 1,
                                            center=mon_center, size=mon_size,
                                            yee_grid=False))
            
    # Run simulation
    sim_RGB.run(until_after_sources=mp.stop_when_dft_decayed(1e-7))
    
    
    # Extract field
    Sz_list = []
    for Ex_f, Ey_f, Hx_f, Hy_f in zip(Ex_fields, Ey_fields, Hx_fields, Hy_fields):
        Ex = sim_RGB.get_dft_array(Ex_f, mp.Ex, 0)
        Ey = sim_RGB.get_dft_array(Ey_f, mp.Ey, 0)
        Hx = sim_RGB.get_dft_array(Hx_f, mp.Hx, 0)
        Hy = sim_RGB.get_dft_array(Hy_f, mp.Hy, 0)

        Sz = 0.5 * np.real(-Ex * np.conj(Hy) + Ey * np.conj(Hx))
        Sz_list.append(Sz)

    k_grid = int(round(len(Sz_list) ** 0.5))   # 3/4/5/6 auto
    P_sub  = Sx / k_grid                        # sub-pixel size (um)

    def target_box(i, lw):
        """Channel i's target sub-pixel region (white outline)."""
        cx = (i %  k_grid - (k_grid - 1) / 2) * P_sub
        cy = ((k_grid - 1) / 2 - i // k_grid) * P_sub
        m = P_sub*0.02
        return Rectangle((cx - P_sub/2 + m, cy - P_sub/2 + m), P_sub - 2*m, P_sub - 2*m,
                        edgecolor='white', facecolor='none',
                        linewidth=lw, zorder=5)


    # Plot
    if mp.am_master():
        for i, Sz in enumerate(Sz_list):

            plt.figure(figsize=(6,6))
            im = plt.imshow(
                Sz.transpose(),
                origin='lower',
                cmap=cmap_dict[f"B{i:02d}"],
                interpolation='spline36',
                extent=[-Sx/2, Sx/2, -Sy/2, Sy/2]
            )
            
            plt.gca().add_patch(target_box(i, lw=1.2))

            # Create colorbar
            cbar = plt.colorbar(im)

            # Compute min/max
            vmin = Sz.min()
            vmax = Sz.max()

            # Set ticks ONLY at min/max
            cbar.set_ticks([vmin, vmax])

            # Set LaTeX labels
            cbar.set_ticklabels([
                r"$Sz_{\min}$",
                r"$Sz_{\max}$"
            ])

            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')
            plt.tight_layout()

            plt.savefig(f"{save_dir_str}/Sz_XY_band{i:02d}.png", dpi=300)
            plt.close()
            
            csv_path = f"{save_dir_str}/Sz_XY_band{i:02d}.csv"
            np.savetxt(csv_path, Sz, delimiter=",")

    # -----------------------------------------
    # Save 36-band intensity as one 6x6 figure
    # -----------------------------------------
    if mp.am_master():

        fig, axes = plt.subplots(6, 6, figsize=(12, 12))
        for i, Sz in enumerate(Sz_list):
            ax = axes[i // 6, i % 6]

            im = ax.imshow(
                Sz.transpose(),
                origin='lower',
                cmap=cmap_dict[f"B{i:02d}"],
                interpolation='spline36',
                extent=[-Sx/2, Sx/2, -Sy/2, Sy/2]
            )
            
            ax.add_patch(target_box(i, lw=0.8))

            ax.set_title(f"B{i:02d}: {wavelengths_list[i]*1000:.0f} nm", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        plt.savefig(f"{save_dir_str}/Sz_XY_all_6x6.png", dpi=300)
        plt.close()
