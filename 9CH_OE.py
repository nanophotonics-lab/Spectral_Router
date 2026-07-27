# ==========================================================================
# Evaluation script for the 9-channel (3x3) inverse-designed spectral router.
#
# Loads the optimized binary TiO2/SiO2 design (final_design.txt) and evaluates:
#   (1) Per-channel routing-efficiency spectra T_ii(lambda), defined as the
#       z-directed Poynting flux through the target sub-pixel region divided
#       by the incident power (normalization run below).
#       -> ./ucell/OE_abs.png, ./ucell/OE_abs_data.csv
#       The paper's per-channel routing efficiency is T_ii evaluated at the
#       target wavelength lambda_i (linear interpolation of the CSV columns),
#       and the average routing efficiency is (1/N) * sum_i T_ii(lambda_i).
#   (2) Focal-plane intensity (Sz) maps at each target wavelength, with each
#       channel map normalized independently for visualization.
#       -> ./E2_plot/  (montage panels)
#
# Target wavelengths: 430--670 nm (nominal spacing 30 nm),
# assigned to the 3x3 sub-pixel array in row-major order.
#
# Sub-pixel geometry (config-dependent; see configs.csv). The k x k super-pixel
# has lateral size design_region = k * P. Each sub-pixel has pitch P, split into
# a photodiode aperture PD and a deep-trench-isolation gap DTI = P - PD:
#     P (nm):   120   160   200   240   280   340   500   600
#     PD (um):  0.08  0.12  0.16  0.20  0.24  0.30  0.40  0.48
#     DTI (um): 0.04  0.04  0.04  0.04  0.04  0.04  0.10  0.12
# Set design_region_x/y, PD_size, DTI_size (and the target wavelengths and
# device height h) to match the config folder being evaluated.
#
# Conditions follow the paper unless noted: device thickness
# h = 2.0 um, FDTD resolution 50 pixels/um, normal incidence,
# periodic boundaries in x,y and PML in z.
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


if True: # Target wavelengths ###########################################################################################################################
    
    wavelengths_0 = 0.430
    wavelengths_1 = 0.460
    wavelengths_2 = 0.490
    
    wavelengths_3 = 0.520
    wavelengths_4 = 0.550
    wavelengths_5 = 0.580
    
    wavelengths_6 = 0.610
    wavelengths_7 = 0.640
    wavelengths_8 = 0.670
    
    wavelengths_list = np.array([
        wavelengths_0,
        wavelengths_1,
        wavelengths_2,
        wavelengths_3,
        wavelengths_4,
        wavelengths_5,
        wavelengths_6,
        wavelengths_7,
        wavelengths_8,
    ])
    
    if mp.am_master():
        print("===================================================")
        print("OE Start")

        for i, wl in enumerate(wavelengths_list):
            print(f"Cell {i:02d} = {wl * 1000:.2f} nm")

        print("===================================================")
    
    frequency_list = 1 / wavelengths_list


if True: # Parameters ##################################################################################################
    
    Air  = mp.Medium(index=1.0)
    SiO2 = mp.Medium(index=1.45)
    SiN  = mp.Medium(index=2.10)
    TiO2 = mp.Medium(index=2.65)
    Si   = mp.Medium(index=1.45)

    # Resolution
    resolution = 50
    
    design_region_x = 1.5
    design_region_y = 1.5
    design_region_z = 2.0

    DTI_size  = 0.10
    PD_size   = 0.40
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

    frequency = 1/0.55
    width     = 2
    fwidth    = frequency * width
    src       = mp.GaussianSource(frequency=frequency, fwidth=fwidth, is_integrated=True)
    source    = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center),
                mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]

if True: # Simulation Environment

    structure_weight = np.loadtxt('final_design.txt')
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


    wavelengths = np.linspace(0.400, 0.700, 61)
    frequencies = 1/wavelengths
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
        
        os.makedirs("./ucell", exist_ok=True)


        opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, 0, np.inf), center = (0,0,0)),
                source_parameters={'alpha':1}, monitor_parameters={'alpha':1},
                )
        plt.xlabel("Width (μm)")
        plt.ylabel("Height (μm)")
        plt.savefig("./ucell/Design_XZ.png", bbox_inches='tight')
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
        

        opt.plot2D(False, output_plane = mp.Volume(size = (np.inf, np.inf, 0), center = (0,0,Sz/2-Lpml-pml_2_src-src_2_geo)))
        plt.xlabel("X (μm)")
        plt.ylabel("Y (μm)")
        plt.savefig("./ucell/Design_XY.png", bbox_inches='tight')
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
        
        os.makedirs("./E2_plot", exist_ok=True)
    


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
        save_dir_str = f"./ucell"
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
    
    # Target freq.
    fcen  = 1/0.5
    df    = fcen * 1.5
    nfreq = 1000

    src    = mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True) 
    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)
                ,mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]
    opt.sim.change_sources(source)



    # transmission moiniter
    tran_t = mp.FluxRegion(center=mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution, 3)),
                            size=mp.Vector3(Sx, Sy, 0),)
    tran_total = opt.sim.add_flux(fcen, df, nfreq, tran_t)
    
    # pt
    pt = mp.Vector3(0, 0, round(-Sz/2 + Lpml + mon_2_pml - 1/resolution, 3))

    
    # Normalization run (no design structure): records the incident-power
    # spectrum total_flux, used as the denominator of the per-channel
    # routing efficiency T_ii(lambda).
    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-7))

    #Save data
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
        k_point = k
    )


    #Compute flux for each pixel
    monitor_size = mp.Vector3(PD_size, PD_size, 0)
    
    FluxRegion_00 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/3, +design_region_y * 1/3, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_01 = mp.FluxRegion(center=mp.Vector3(0, +design_region_y * 1/3, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_02 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/3, +design_region_y * 1/3, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_03 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/3, 0, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_04 = mp.FluxRegion(center=mp.Vector3(0, 0, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_05 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/3, 0, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    FluxRegion_06 = mp.FluxRegion(center=mp.Vector3(-design_region_x * 1/3, -design_region_y * 1/3, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_07 = mp.FluxRegion(center=mp.Vector3(0, -design_region_y * 1/3, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    FluxRegion_08 = mp.FluxRegion(center=mp.Vector3(+design_region_x * 1/3, -design_region_y * 1/3, round(-Sz/2 + PD_height - 1/resolution, 3)), size=monitor_size)
    
    
    tran_00 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_00)
    tran_01 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_01)
    tran_02 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_02)
    
    tran_03 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_03)
    tran_04 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_04)
    tran_05 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_05)
    
    tran_06 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_06)
    tran_07 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_07)
    tran_08 = opt.sim.add_flux(fcen, df, nfreq, FluxRegion_08)

    # Run Simulation
    opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-7))

    # Save data
    
    fluxes_00 = mp.get_fluxes(tran_00)
    fluxes_01 = mp.get_fluxes(tran_01)
    fluxes_02 = mp.get_fluxes(tran_02)
    fluxes_03 = mp.get_fluxes(tran_03)
    fluxes_04 = mp.get_fluxes(tran_04)
    fluxes_05 = mp.get_fluxes(tran_05)
    fluxes_06 = mp.get_fluxes(tran_06)
    fluxes_07 = mp.get_fluxes(tran_07)
    fluxes_08 = mp.get_fluxes(tran_08)

    fluxes_list = [
        fluxes_00, fluxes_01, fluxes_02, fluxes_03,
        fluxes_04, fluxes_05, fluxes_06, fluxes_07,
        fluxes_08,
    ]
    
    # Per-channel routing-efficiency spectra:
    #   T_ii(lambda) = flux through target sub-pixel i / incident power.
    # This is the quantity reported as "routing efficiency" in the paper;
    # the average routing efficiency is the mean of T_ii(lambda_i) over i.
    wl     = []
    T_list = [[] for _ in range(9)]

    for d in range(nfreq):
        wl = np.append(wl, 1/flux_freqs[d])
        
        for i in range(9): # 00~08
            T_list[i] = np.append(T_list[i], fluxes_list[i][d] / total_flux[d])

    ######################################################################################################
    line_colors = [
        "#5A33CC",  # 0 (430) purple-blue
        "#364ADE",  # 1 (460) deep blue
        "#0086FF",  # 2 (490) blue-cyan

        "#00A3FF",  # 3 (520) cyan
        "#00C084",  # 4 (550) teal-green 
        "#00A66A",  # 5 (580) green

        "#FF7A00",  # 6 (610) deep orange
        "#FF4500",  # 7 (640) red-orange
        "#E02020"   # 8 (670) red
    ]

    fill_colors = [
        "#B3A6F2",  # 0 (430)
        "#A2B1FF",  # 1 (460)
        "#A7D9FF",  # 2 (490)

        "#B8E4FF",  # 3 (520)
        "#BFF3DD",  # 4 (550)
        "#BDF0D6",  # 5 (580)

        "#FFD1A3",  # 6 (610)
        "#FFC2B0",  # 7 (640)
        "#FFB3B3"   # 8 (670)
    ]
    ######################################################################################################
    
    # Band edges
    band_edges = [0.400, 0.450, 0.475,
                0.505, 0.535, 0.565,
                0.595, 0.625, 0.655, 0.700]

    # Masks
    band_masks = []
    for i in range(9):
        mask = (wl >= band_edges[i]) & (wl <= band_edges[i+1])
        band_masks.append(mask)
    
    ##############################################################################################################
    
    # -------------------------------
    # Tb / OE_abs Plot
    # -------------------------------
    if mp.am_master():
        plt.figure(dpi=150)

        # light lines
        for i in range(9):
            plt.plot(wl, T_list[i], color=line_colors[i], alpha=0.4)

        # axis
        plt.axis([0.400, 0.700, 0, 1])
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Efficiency")


        # Fill bands (light shading)
        for i in range(9):
            x = [band_edges[i], band_edges[i], band_edges[i+1], band_edges[i+1]]
            y = [-0.03, 1.03, 1.03, -0.03]
            plt.fill(x, y, color=fill_colors[i], alpha=0.4)

        # Bold lines (each band)
        for i in range(9):
            plt.plot(wl[band_masks[i]], T_list[i][band_masks[i]], color=line_colors[i], alpha=1)

        # Save
        plt.savefig(f"{save_dir_str}/OE_abs.png")
        plt.cla()
        plt.clf()
        plt.close()
        
        
        # CSV: per-channel spectra (columns T00..T08); the paper's
        # efficiencies are obtained by interpolating each column at its
        # target wavelength.
        csv_path = f"{save_dir_str}/OE_abs_data.csv"

        # CSV header
        header = "wl," + ",".join([f"T{i:02d}" for i in range(9)])

        with open(csv_path, "w") as f:
            f.write(header + "\n")

            # For each wavelength index
            for idx in range(len(wl)):
                row = [f"{wl[idx]:.8f}"]  # wl value with good precision

                # Append T00 ~ T8
                for i in range(9):
                    row.append(f"{T_list[i][idx]:.8f}")

                # Write row
                f.write(",".join(row) + "\n")



# Focal-plane intensity (Sz) maps at each target wavelength.
# Each channel map is normalized independently for visualization
# (focal-plane Sz montage panels).
if INTENSITY:

    
    phi = math.radians(phi_deg)
    phi_str = f"{int(phi_deg)}deg"

    
    theta = math.radians(theta_deg)
    theta_str = f"{int(theta_deg)}deg"


    if mp.am_master(): # Save directory
        print("------------------------------------------------------")
        print("Intensity Plot...")
        print("------------------------------------------------------")
        save_dir_str = f"./E2_plot"
        os.makedirs(save_dir_str, exist_ok=True)
    
    
    frequency_list
        

    fcen   = 1/0.5
    fwidth = fcen * 2
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

    sources = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amp_func=pw_amp(k, source_center)),
                mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amp_func=pw_amp(k, source_center))]
    
    line_colors = [
        "#5A33CC",  # 0 (430) purple-blue
        "#364ADE",  # 1 (460) deep blue
        "#0086FF",  # 2 (490) blue-cyan

        "#00A3FF",  # 3 (520) cyan
        "#00C084",  # 4 (550) teal-green (green center)
        "#00A66A",  # 5 (580) green

        "#FF7A00",  # 6 (610) deep orange
        "#FF4500",  # 7 (640) red-orange
        "#E02020"   # 8 (670) red
    ]
    
    # cmap creation
    cmap_dict = {}

    for i in range(9):
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
        k_point = k
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

    # -----------------------------------------
    # Save 9-band intensity as one 3x3 figure
    # -----------------------------------------
    if mp.am_master():

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for i, Sz in enumerate(Sz_list):
            ax = axes[i // 3, i % 3]

            im = ax.imshow(
                Sz.transpose(),
                origin='lower',
                cmap=cmap_dict[f"B{i:02d}"],
                interpolation='spline36',
                extent=[-Sx/2, Sx/2, -Sy/2, Sy/2]
            )
            
            ax.add_patch(target_box(i, lw=0.8))

            ax.set_title(f"Band {i:02d}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f"{save_dir_str}/Sz_XY_all_3x3.png", dpi=300)
        plt.close()
