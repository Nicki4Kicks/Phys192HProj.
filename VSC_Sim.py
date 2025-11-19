import numpy as np
import matplotlib.pyplot as plt

# -------------------- Physical constants --------------------
SPEED_OF_LIGHT_CM_S = 2.99792458e10   # cm/s
TWO_PI = 2.0 * np.pi

# -------------------- Background (water) ---------
N_WATER = 1.33
EPS_WATER = N_WATER**2  # dielectric function epsilon = n^2

# -------------------- Polymers (need confirmed numbers for lines and center) ----------------
POLYMERS = {
    "PET": {
        "eps_inf": 2.46,   # n = 1.57
        "lines":   [{"nu_cm": 1725.0, "gamma_cm": 14.0, "S": 0.90}],  # C=O
        "density": 1.38,
        "center":  1725.0,
        "window":  200.0,
    },
    "PE": {
        "eps_inf": 2.28,   # n = 1.51
        "lines":   [{"nu_cm": 1465.0, "gamma_cm": 16.0, "S": 0.80}],  # CH2
        "density": 0.95,
        "center":  1465.0,
        "window":  200.0,
    },
    "PP": {
        "eps_inf": 2.22,   # n = 1.49
        "lines": [
            {"nu_cm": 1455.0, "gamma_cm": 16.0, "S": 0.60},
            {"nu_cm": 1375.0, "gamma_cm": 16.0, "S": 0.60},
        ],
        "density": 0.90,
        "center":  1455.0,
        "window":  220.0,
    },
    "PS": {
        "eps_inf": 2.53,   # n = 1.59
        "lines": [
            {"nu_cm": 1601.0, "gamma_cm": 12.0, "S": 0.55},
            {"nu_cm": 1492.0, "gamma_cm": 14.0, "S": 0.65},
        ],
        "density": 1.05,
        "center":  1490.0,
        "window":  260.0,
    },
    "PVC": {
        "eps_inf": 2.37,   # n = 1.54
        "lines": [
            {"nu_cm": 1430.0, "gamma_cm": 16.0, "S": 0.60},
            {"nu_cm": 1250.0, "gamma_cm": 18.0, "S": 0.40},
        ],
        "density": 1.40,
        "center":  1430.0,
        "window":  240.0,
    },
}

# -------- Select polymer to simulate ----------
SELECTED_POLYMER = "PET"   # change as needed

# -------------------- Cavity / simulation settings ----------
MIRROR_REFLECTIVITY = 0.95          # power reflectivity R
N_SPECTRAL_POINTS = 2001            # samples across spectral window

# Concentrations (mg/L)
CONC_MG = [15, 20, 25, 35, 50]  # add more values as needed

# Optional losses (disabled)
WATER_ALPHA_CM = 0.0                # baseline water absorption (1/cm)
PARTICLE_RADIUS_UM = 1.0            # for Rayleigh-like scattering
SCATTER_COEFF = 0.0                 # 0 => disable scattering

# -------------------- VSC / cavity model parameters ---------
N_CAV = 1.33          # effective refractive index in the cavity (water)
MODE_INDEX = 1        # longitudinal mode number m

G_CM_REF = 40.0       # cm^-1 at reference concentration
CONC_VSC_REF = 50.0   # concentration taken as "strong coupling" reference
CONC_VSC_THRESH = 8.5 # threshold concentration for onset of VSC (same units as CONC_MG)

N_L_POINTS = 300      # points per thickness sweep
L_RANGE_FACTOR = 0.6  # how wide to sweep around crossing (fractional)
L_SPECTRA_OFFSET = 0.10  # +-10% around crossing for spectra

# -------------------- Helper functions ----------------------
def wavenumber_to_omega(nu_cm):
    """Convert wavenumber (cm^-1) to angular frequency (rad/s)."""
    return TWO_PI * SPEED_OF_LIGHT_CM_S * nu_cm


def eps_water():
    """Complex dielectric function of water (constant here)."""
    return complex(EPS_WATER, 0.0)


def eps_polymer(omega, eps_inf, lines):
    """Polymer dielectric function via sum of Lorentz oscillators."""
    eps = complex(eps_inf, 0.0)
    for ln in lines:
        w0 = wavenumber_to_omega(ln["nu_cm"])      # resonance (rad/s)
        gam = wavenumber_to_omega(ln["gamma_cm"])  # damping (rad/s)
        S = ln["S"]                                # oscillator strength
        eps += S * (w0**2) / ((w0**2 - omega**2) - 1j * gam * omega)
    return eps


def eps_effective_MG(omega, vol_frac, eps_inf, lines):
    """Maxwell-Garnett effective permittivity (spherical inclusions)."""
    eps_host = eps_water()
    eps_incl = eps_polymer(omega, eps_inf, lines)
    num = eps_incl + 2*eps_host + 2*vol_frac*(eps_incl - eps_host)
    den = eps_incl + 2*eps_host -   vol_frac*(eps_incl - eps_host)
    return eps_host * (num / den)


def mgL_to_volfrac(conc_mgL, density_g_cm3):
    """Convert mg/L concentration to volume fraction (f)."""
    return (conc_mgL * 1e-6) / density_g_cm3   # 1 mg/L = 1e-6 g/cm^3


def airy_transmission(omega, length_um, reflectivity_R, vol_frac,
                      eps_inf, lines,
                      particle_radius_um=1.0,
                      scatter_coeff=0.0,
                      water_alpha_cm=0.0):
    """Airy FP transmission with optional bulk/scatter loss."""
    R = np.clip(reflectivity_R, 1e-6, 0.999999)
    F = 4.0 * R / (1.0 - R)**2           # finesse factor
    L_cm = length_um * 1e-4              # um -> cm

    # Effective medium index inside cavity
    eps_eff = eps_effective_MG(omega, vol_frac, eps_inf, lines)
    n_eff = np.sqrt(eps_eff)             # complex index

    # Wavelength in cm from angular frequency
    lam_cm = TWO_PI * SPEED_OF_LIGHT_CM_S / omega
    # Round-trip phase
    phase = TWO_PI * n_eff * L_cm / lam_cm

    # Optional phenomenological losses
    r_cm = particle_radius_um * 1e-4
    alpha_scat = scatter_coeff * vol_frac * (TWO_PI * r_cm / lam_cm)**4
    alpha_total = water_alpha_cm + alpha_scat
    path_loss = np.exp(-alpha_total * 2.0 * L_cm)

    # Airy formula (power transmission); take real part for plotting
    return np.real(path_loss / (1.0 + F * (np.sin(phase)**2)))


def wavenumber_axis(center_cm, window_cm, n_points):
    """Create a linear wavenumber axis centered on the band."""
    half = window_cm / 2.0
    return np.linspace(center_cm - half, center_cm + half, int(n_points))


def cavity_wavenumber(L_um, m, n_cav):
    """
    Bare cavity resonance wavenumber v_c(L) in cm^-1.

    FP condition: m * lambda / (2 n) = L
      => lambda = 2 n L / m
      => lambda = 1 / lambda = m / (2 n L)
    """
    L_cm = L_um * 1e-4
    return m / (2.0 * n_cav * L_cm)


def polariton_branches(nu_c, nu_v, g_cm):
    """
    Coupled oscillator (cavity + vibration) polariton branches.

    v+- = (v_c + v_v)/2 +- 0.5 * sqrt( (v_c - v_v)**2 + 4 g**2 )
    """
    delta = nu_c - nu_v
    root = np.sqrt(delta**2 + 4.0 * g_cm**2)
    nu_plus = 0.5 * (nu_c + nu_v + root)
    nu_minus = 0.5 * (nu_c + nu_v - root)
    return nu_plus, nu_minus


def crossing_thickness_um(nu_v_cm, m, n_cav):
    """
    Solve v_c(L_cross) = v_v for L_cross.

    v_c = m / (2 n L_cm) = v_v  =>  L_cm = m / (2 n v_v)
    Convert to um: L_um = L_cm * 1e4
    """
    L_cm = m / (2.0 * n_cav * nu_v_cm)
    return L_cm * 1e4


def g_from_conc_linear_threshold(conc):
    """
    Concentration-dependent coupling strength g(conc) with a threshold.
    - For conc <= CONC_VSC_THRESH: g = 0 (no Rabi splitting).
    - For conc >  CONC_VSC_THRESH: g grows linearly with conc and
      equals G_CM_REF at conc = CONC_VSC_REF (based on paper).
    """
    if conc <= CONC_VSC_THRESH:
        return 0.0
    if CONC_VSC_REF <= CONC_VSC_THRESH:
        # avoid divide by zero, treat as no coupling
        return 0.0
    slope = G_CM_REF / (CONC_VSC_REF - CONC_VSC_THRESH)
    return slope * (conc - CONC_VSC_THRESH)

# -------------------- Main: VSC anticrossing + spectra ------

def main():
    # ---------- Pick the polymer configuration ----------------
    polymer_name = SELECTED_POLYMER
    cfg = POLYMERS[polymer_name]
    nu_v = cfg["center"]  # vibrational center (cm^-1)

    # ---------- Anticrossing: compute thickness range ----------
    L_cross_um = crossing_thickness_um(nu_v, MODE_INDEX, N_CAV)
    L_min_um = L_cross_um * (1.0 - L_RANGE_FACTOR)
    L_max_um = L_cross_um * (1.0 + L_RANGE_FACTOR)
    L_um = np.linspace(L_min_um, L_max_um, N_L_POINTS)

    # Bare cavity resonance vs thickness
    nu_c = cavity_wavenumber(L_um, MODE_INDEX, N_CAV)

    # Choose a concentration to represent the VSC anticrossing
    conc_for_vsc = max(CONC_MG)
    g_eff = g_from_conc_linear_threshold(conc_for_vsc)

    # Polariton branches (if g_eff = 0, this just gives a crossing)
    nu_plus, nu_minus = polariton_branches(nu_c, nu_v, g_eff)

    # ---------- Plot 1: anticrossing for this polymer ----------
    plt.figure(figsize=(7.0, 4.8))

    plt.plot(L_um, nu_plus, label="Upper polariton v+")
    plt.plot(L_um, nu_minus, label="Lower polariton v-")
    plt.plot(L_um, nu_c, "--", label="Bare cavity v_c")
    plt.axhline(nu_v, linestyle=":", label="Bare vibration v_v")

    plt.xlabel("Cavity thickness L (um)")
    plt.ylabel("Wavenumber (cm$^{-1}$)")
    plt.title(
        f"{polymer_name} VSC anticrossing\n"
        f"ν_v ≈ {nu_v:.0f} cm$^{{-1}}$, "
        f"L_cross ≈ {L_cross_um:.2f} um, "
        f"conc_for_VSC = {conc_for_vsc:g}, "
        f"2g_eff ≈ {2*g_eff:.1f} cm$^{{-1}}$"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---------- Plot 2: transmission spectra near anticrossing ----------
    # Thicknesses slightly below, at, and above crossing
    L_list = [
        L_cross_um * (1.0 - L_SPECTRA_OFFSET),
        L_cross_um,
        L_cross_um * (1.0 + L_SPECTRA_OFFSET),
    ]

    # Wavenumber axis around the vibrational band
    nu_cm = wavenumber_axis(cfg["center"], cfg["window"], N_SPECTRAL_POINTS)
    omega = wavenumber_to_omega(nu_cm)

    # Storage for final combined deltaT plot
    combined_dT_data = []   # each entry: (nu_cm, dT, conc, L_spec)

    # Loop over all concentrations
    for conc in CONC_MG:
        vol_frac = mgL_to_volfrac(conc, cfg["density"])

        # Compute baseline water-only spectra (vol_frac = 0)
        T_baseline_list = []
        T_poly_list = []
        for L_spec in L_list:
            T0 = airy_transmission(
                omega,
                L_spec,
                MIRROR_REFLECTIVITY,
                0.0,               # no polymer: baseline water
                cfg["eps_inf"],
                cfg["lines"],
                particle_radius_um=PARTICLE_RADIUS_UM,
                scatter_coeff=SCATTER_COEFF,
                water_alpha_cm=WATER_ALPHA_CM,
            )
            T_baseline_list.append(T0)

            T = airy_transmission(
                omega,
                L_spec,
                MIRROR_REFLECTIVITY,
                vol_frac,
                cfg["eps_inf"],
                cfg["lines"],
                particle_radius_um=PARTICLE_RADIUS_UM,
                scatter_coeff=SCATTER_COEFF,
                water_alpha_cm=WATER_ALPHA_CM,
            )
            T_poly_list.append(T)

        # ---- Plot absolute spectra: baseline vs polymer (VSC) ----
        plt.figure(figsize=(7.0, 4.8))
        for (L_spec, T0, T) in zip(L_list, T_baseline_list, T_poly_list):
            # baseline water (thin dashed)
            plt.plot(
                nu_cm,
                T0,
                linestyle="--",
                alpha=0.6,
                label=f"Water, L={L_spec:.2f} um"
            )
            # with polymer (VSC / absorption)
            plt.plot(
                nu_cm,
                T,
                label=f"Polymer conc={conc:g} mg/L, L={L_spec:.2f} um"
            )

        plt.gca().invert_xaxis()  # IR convention: high -> low wavenumber
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Transmission")
        plt.title(
            f"{polymer_name}: FP spectra near VSC "
            f"(conc = {conc:g} mg/L)"
        )
        plt.axvline(nu_v, linestyle=":", color="k",
                    label="Vibrational center v_v")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        # ---- Store deltaT curves (do NOT plot yet) ----
        for (L_spec, T0, T) in zip(L_list, T_baseline_list, T_poly_list):
            dT = T - T0
            combined_dT_data.append((nu_cm, dT, conc, L_spec))

    # ---------- FINAL PLOT: All concentrations deltaT together ----------
    plt.figure(figsize=(7.0, 4.8))

    for (nu_cm, dT, conc, L_spec) in combined_dT_data:
        plt.plot(
            nu_cm,
            dT,
            label=f"deltaT, conc={conc:g} mg/L, L={L_spec:.2f} um"
        )

    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("deltaTransmission (polymer - water)")
    plt.title(f"{polymer_name}: All deltaT curves for all concentrations")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.axvline(nu_v, linestyle=":", color="k",
                label="Vibrational center v_v")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
