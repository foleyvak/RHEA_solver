import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from rhea_thermodynamics_transport_coefficients import PengRobinsonModel, HighPressureTransportCoeficients
from matplotlib.colorbar import Colorbar 
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import pandas as pd

plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 18, family='serif' )
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb}')

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

save_curve = True
load_curve = True

if not load_curve:
    # Constants
    thermodynamics = PengRobinsonModel(
        molecular_weight=0.04401, 
        acentric_factor=0.22394, 
        critical_temperature=304.1282, 
        critical_pressure=7377270.6, 
        critical_molar_volume=0.0000941189, 
        NASA_coefficients=[ 4.6365111000000000000000,
                            0.0027414569000000000000,
                            -0.0000009958975900000000,
                            0.0000000001603866600000,
                            -0.0000000000000091619857,
                            -49024.904000000000000000,
                            -1.9348955000000000000000,
                            2.3568130000000000000000,
                            0.0089841299000000000000,
                            -0.0000071220632000000000,
                            0.0000000024573008000000,
                            -0.0000000000001428854800,
                            -48371.971000000000000000,
                            9.9009035000000000000000,
                            -47328.105000000000000000] 
                            )
    R = 8.31446261815324        # Universal gas constant (J/mol·K)

    # Substance properties (Carbon Dioxide - CO2)
    Tc = 304.1282       # Critical temperature (K)
    Pc = 7377270.6      # Critical pressure (Pa)
    rho_c, _ = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(-1.0, -1.0, P=Pc, T=Tc)
    vc = 1.0/rho_c
    # vc = 0.0021386      # Critical specific volume (m3/kg)
    M  = 0.04401        # Molar mass (kg/mol)
    omega = 0.22394     # Acentric factor (-)

    # Peng-Robinson EOS Parameters
    a = 0.457236 * (R**2 * Tc**2) / Pc
    b = 0.077796 * (R * Tc) / Pc

    # Temperature-dependent alpha
    def alpha(T):
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2.0
        if omega > 0.49:
            kappa = 0.379642 + 1.48503*omega - 0.164423*omega**2.0 + 0.016666*omega**3.0 
        Tr = T / Tc
        return (1 + kappa * (1 - np.sqrt(Tr)))**2

    # Solve for Z-factor
    def cubic_Z(A, B):
        try:
            A, B = float(A), float(B)  # Ensure A and B are scalars
            coeffs = [1.0, -(1.0 - B), (A - 3.0 * B**2 - 2.0 * B), -(A * B - B**2 - B**3)]
            roots = np.roots(coeffs)
            real_roots = np.real(roots[np.isreal(roots)])
            return real_roots if len(real_roots) > 0 else []
        except Exception as e:
            print(f"Error in cubic_Z: {e}, A={A}, B={B}")
            return []

    # Fugacity coefficient
    def fugacity_coeff(Z, A, B):
        term1 = Z - 1 - np.log(Z - B)
        term2 = A / (2 * np.sqrt(2) * B)
        term3 = np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
        return np.exp(term1 - term2 * term3)

    # Solve for saturation pressure
    T_range = np.linspace( 0.925*Tc, 0.99995*Tc, 150000 )       # Temperature range (K)
    P_sat   = []
    v_L_sat = []
    v_V_sat = []

    def objective(P, T):
        A = a * alpha(T) * P / (R**2 * T**2)
        B = b * P / (R * T)
        
        Z_roots = cubic_Z(A, B)
        Z_L = min(Z_roots)
        Z_V = max(Z_roots)
        
        phi_L = fugacity_coeff(Z_L, A, B)
        phi_V = fugacity_coeff(Z_V, A, B)
        
        return phi_L - phi_V

    P_guess = 5.0e6
    P_old = P_guess
    for T in T_range:
        P_guess = P_old    # Initial guess
        P_sat_T = opt.fsolve(objective, P_guess, args=(T))[0]
        P_old = P_sat_T
        P_sat.append( P_sat_T )
        A = a * alpha(T) * P_sat_T / (R**2 * T**2)
        B = b * P_sat_T / (R * T)
        Z_roots = cubic_Z(A, B)
        Z_L = min(Z_roots)
        Z_V = max(Z_roots)    
        v_L_sat.append( R*T*Z_L/( P_sat_T*M ) )
        v_V_sat.append( R*T*Z_V/( P_sat_T*M ) )

    if save_curve:
        v_combined = np.concatenate((v_L_sat, v_V_sat))  # Combine the two volume arrays
        P_combined = np.concatenate((P_sat, P_sat))  # Corresponding pressures

        # Sort arrays based on volume
        sorted_indices = np.argsort(v_combined)
        v_sorted = v_combined[sorted_indices]
        P_sorted = P_combined[sorted_indices]

        # Create interpolation function
        from scipy.interpolate import interp1d
        interp_func = interp1d(v_sorted, P_sorted, kind='cubic')

        # Generate finer grid
        v_sat = np.linspace(v_sorted.min(), v_sorted.max(), 1000)
        P_sat = interp_func(v_sat)
        np.save('VLE_curve.npy', np.array([v_sat, P_sat]))
else:
    loaded_data = np.load('VLE_curve.npy')
    v_sat, P_sat = loaded_data  # Unpack saved arrays

# Plot VLE Curve
# plt.scatter( ( v_sat ), np.array( P_sat ) )
# plt.show()


# Normalization factors
thermodynamics = PengRobinsonModel(
        molecular_weight=0.04401, 
        acentric_factor=0.22394, 
        critical_temperature=304.1282, 
        critical_pressure=7377270.6, 
        critical_molar_volume=0.0000941189, 
        NASA_coefficients=[ 4.6365111000000000000000,
                            0.0027414569000000000000,
                            -0.0000009958975900000000,
                            0.0000000001603866600000,
                            -0.0000000000000091619857,
                            -49024.904000000000000000,
                            -1.9348955000000000000000,
                            2.3568130000000000000000,
                            0.0089841299000000000000,
                            -0.0000071220632000000000,
                            0.0000000024573008000000,
                            -0.0000000000001428854800,
                            -48371.971000000000000000,
                            9.9009035000000000000000,
                            -47328.105000000000000000] 
                            )
transport_coefficients = HighPressureTransportCoeficients( molecular_weight=0.04401, 
                                                        acentric_factor=0.22394, 
                                                        critical_temperature=304.1282, 
                                                        critical_molar_volume=0.0000941189, 
                                                        NASA_coefficients=[ 4.6365111000000000000000,
                                                                            0.0027414569000000000000,
                                                                            -0.0000009958975900000000,
                                                                            0.0000000001603866600000,
                                                                            -0.0000000000000091619857,
                                                                            -49024.904000000000000000,
                                                                            -1.9348955000000000000000,
                                                                            2.3568130000000000000000,
                                                                            0.0089841299000000000000,
                                                                            -0.0000071220632000000000,
                                                                            0.0000000024573008000000,
                                                                            -0.0000000000001428854800,
                                                                            -48371.971000000000000000,
                                                                            9.9009035000000000000000,
                                                                            -47328.105000000000000000] , 
                                                        dipole_moment=0.0,
                                                        association_factor=0.0)


critical_pressure=7377270.6
critical_temperature=304.1282 
index_critical = np.argmax(P_sat) 
P_c_PR = P_sat[index_critical] 
v_c_PR = v_sat[index_critical]  
print( "Critical pressure using Peng-robinson: ", P_c_PR, ". Standard value: ", 7377270.6)
print( "Critical specific volume using Peng-robinson: ", v_c_PR, ". Standard value: ", 0.0021386)

n_points_pressure = 500
n_points_volume = 400
P_range = np.linspace( 0.6*critical_pressure, 2.5*critical_pressure, n_points_pressure )
v_range = np.linspace( 0.5*v_c_PR, 3.5*v_c_PR, n_points_volume )

T_matrix = np.full((n_points_pressure, n_points_volume), np.nan)  # Fill with NaNs initially
Pr_matrix = np.full((n_points_pressure, n_points_volume), np.nan)  # Fill with NaNs initially
c_p_matrix = np.full((n_points_pressure, n_points_volume), np.nan)  # Fill with NaNs initially
Z_matrix = np.full((n_points_pressure, n_points_volume), np.nan)  # Fill with NaNs initially

for i, P_i in enumerate(P_range):
    for j, v_i in enumerate(v_range):
        if P_i < P_c_PR:
            # Find the two volumes (v_L, v_V) at this pressure from the VLE curve
            idx = np.where(P_sat >= P_i)[0]  # Indices where P_sat is below P_i
            # print( P_i, idx )
            if len(idx) >= 2:
                v_L, v_V = v_sat[idx[0]], v_sat[idx[-1]]  # Saturated liquid & vapor volumes

                if v_L <= v_i <= v_V:
                    continue  # Skip points inside the VLE region

        # Compute variables
        rho_i = 1.0 / v_i
        T_i = thermodynamics.calculateTemperatureFromPressureDensity(P=P_i, rho=rho_i)
        mu = transport_coefficients.calculateDynamicViscosity(P=P_i, T=T_i, rho=rho_i)
        kappa = transport_coefficients.calculateThermalConductivity(P=P_i, T=T_i, rho=rho_i)
        _, c_p = thermodynamics.calculateSpecificHeatCapacities(c_v = -1, c_p=-1,  P=P_i, T=T_i, rho=rho_i)
        Z_i = thermodynamics.calculate_Z(P = P_i, T=T_i, bar_v=0.04401/rho_i)

        T_matrix[i, j] = T_i
        Pr_matrix[i,j] = c_p*mu/kappa
        c_p_matrix[i,j] = c_p
        Z_matrix[i,j] = Z_i

# Pseudo-boiling calculation
P_pseudoboiling = P_range[P_range >= P_c_PR]
v_pseudoboiling = v_range[ np.argmax(c_p_matrix[P_range >= P_c_PR, :], axis=1)]


# Isotherm calculations
subcritical_isotherm_P_r = np.zeros_like(v_range)
critical_isotherm_P_r = np.zeros_like(v_range)
supercritical_isotherm_P_r = np.zeros_like(v_range)
for ii, v_i in enumerate(v_range):
    rho_i = 1/v_i
    # Subcritical isotherm:
    T_i = 0.96*critical_temperature
    P_i = thermodynamics.calculatePressureFromTemperatureDensity(T=T_i, rho=rho_i)
    subcritical_isotherm_P_r[ii] = P_i/P_c_PR
    # Critical isotherm:
    T_i = 1.00*critical_temperature
    P_i = thermodynamics.calculatePressureFromTemperatureDensity(T=T_i, rho=rho_i)
    critical_isotherm_P_r[ii] = P_i/P_c_PR
    # Supercritical isotherm:
    T_i = 1.1*critical_temperature
    P_i = thermodynamics.calculatePressureFromTemperatureDensity(T=T_i, rho=rho_i)
    supercritical_isotherm_P_r[ii] = P_i/P_c_PR

# Mechanical spinoidal calculations
v_spinodal = []
P_spinodal = []
for T_i in np.linspace(0.90*critical_temperature, 0.999*critical_temperature, 100):
    pressure_curve = np.zeros_like(v_range)
    for ii, v_i in enumerate(v_range):
        rho_i = 1/v_i
        P_i = thermodynamics.calculatePressureFromTemperatureDensity(T=T_i, rho=rho_i)
        pressure_curve[ii] = P_i
    
    # Find local maxima and minima (mechanical spinodal points)
    dP_dV = np.gradient(pressure_curve, v_range)  # First derivative
    sign_change = np.where(np.diff(np.sign(dP_dV)))[0]  # Indices where slope changes sign

    if len(sign_change) >= 2:  # Ensure we have at least one min & max
        min_idx, max_idx = sign_change[0], sign_change[-1]
        v_spinodal.append(v_range[min_idx])  # Append min volume
        v_spinodal.append(v_range[max_idx])  # Append max volume
        P_spinodal.append(pressure_curve[min_idx])  # Append min pressure
        P_spinodal.append(pressure_curve[max_idx])  # Append max pressure

sorted_indices = np.argsort(v_spinodal)  # Get indices that would sort v_spinodal
v_spinodal_sorted = np.array(v_spinodal)[sorted_indices]  # Sort v_spinodal
P_spinodal_sorted = np.array(P_spinodal)[sorted_indices]  # Sort P_spinodal accordingly
from scipy.interpolate import interp1d
interp_v = interp1d(v_spinodal_sorted, P_spinodal_sorted, kind='linear', fill_value="extrapolate")
v_spinoidal_interp = np.linspace(np.min(v_spinodal_sorted), np.max(v_spinodal_sorted), 40)
P_spinoidal_interp = interp_v(v_spinoidal_interp)


##################################################
##################### Figure #####################
##################################################
color_isotherms         = "#4daf4a"
color_Z                 = "#984ea3"
color_Pr                = "#797F81"
color_pseudoboiling     = "#e41a1c"
color_mech_spinoidal    = "b"
cmap                    = "plasma_r"
# cmap = "plasma"
fig = plt.figure(1, figsize=(15, 6.5))
gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
ax1 = plt.subplot(gs[0,0])

# Interpolate to fix stair 
interp_func = interp1d(v_pseudoboiling/v_c_PR, P_pseudoboiling/P_c_PR,  kind='nearest', fill_value='extrapolate')
x_pseudoboiling = np.linspace(np.min(v_pseudoboiling/v_c_PR), np.max(v_pseudoboiling/v_c_PR), 7)
y_pseudoboiling = interp_func( x_pseudoboiling )

## EXPORT PSEUDOBOILING LINE
file_name = 'pseudo-boiling_line.csv'
data_file_out = open(file_name, 'wt')

for x, y in zip(x_pseudoboiling,y_pseudoboiling):
    output_string = f"{x:.18e},{y:.18e}\n"
    data_file_out.write(output_string)

data_file_out.close()
## ==============================================

header = "1/rho [m3/kg], P [Pa]"
np.savetxt('pseudoboiling_line.txt', np.c_[x_pseudoboiling, y_pseudoboiling], header=header, delimiter=', ', fmt='%.18e')

# Pseudoboiling/Critical point
ax1.scatter( 1, 1, s=80, marker="p", color="r", label= r"$\textrm{Critical point}$", zorder=12 )
ax1.plot( v_sat/v_c_PR, P_sat/P_c_PR, color="k", linestyle="-", label=r"$\textrm{VLE}$", zorder=11 )

# Lines: Widom line, Pr=1, Z=0.95, ithotherms (subcritical, critical & supercritical), mechanical spinoidal
# ax1.plot( v_pseudoboiling/v_c_PR, P_pseudoboiling/P_c_PR, color=color_pseudoboiling, linestyle="-", label=r"$\textrm{Pseudo-boiling line}$", zorder = 23)
ax1.plot( x_pseudoboiling, y_pseudoboiling, color=color_pseudoboiling, linestyle=(0, (3, 1, 1, 1)), label=r"$\textrm{Pseudo-boiling line}$", zorder = 23)
contour_Pr = ax1.contour(v_range / v_c_PR, P_range / P_c_PR, Pr_matrix, levels=[1], colors=color_Pr, linestyles="dashdot", linewidths=2)
contour_Z = ax1.contour(v_range / v_c_PR, P_range / P_c_PR, Z_matrix, levels=[0.90], colors=color_Z, linestyles=[(0, (3, 5, 1, 5, 1, 5))], linewidths=2)
contour_label_Pr = ax1.clabel(contour_Pr, inline=True, fmt=r"$\textrm{Pr = %.2f}$", fontsize=18, inline_spacing=15, use_clabeltext=True, manual=[(1.6, 2.12)])
contour_label_Z = ax1.clabel(contour_Z, inline=True, fmt=r"$\textrm{Z = %.2f}$", fontsize=18, inline_spacing=15, use_clabeltext=True, manual=[(2.2, 2.13)])
# for label in contour_label_Pr:            # use_clabeltext=True
#     label.set_rotation(-47)               # use_clabeltext=True
# for label in contour_label_Z:             # use_clabeltext=True
#     label.set_rotation(-46)               # use_clabeltext=True
ax1.plot(v_range/v_c_PR, subcritical_isotherm_P_r,      zorder=11, linestyle="--", color=color_isotherms, label=r"$\textrm{Isotherm}$")
ax1.plot(v_range/v_c_PR, critical_isotherm_P_r,         zorder=11, linestyle="--", color=color_isotherms)
ax1.plot(v_range/v_c_PR, supercritical_isotherm_P_r,    zorder=11, linestyle="--", color=color_isotherms)

ax1.text(1.0, 0.70, r"$T/T_c < 1$", 
         color=color_isotherms, fontsize=18, ha="center", va="bottom", rotation=19)
ax1.text(1.6, 0.945, r"$T/T_c = 1$", 
         color=color_isotherms, fontsize=18, ha="center", va="bottom", rotation=-9)
ax1.text(2.0, 1.1, r"$T/T_c > 1$", 
         color=color_isotherms, fontsize=18, ha="center", va="bottom", rotation=-18)

ax1.plot(v_spinoidal_interp/v_c_PR, P_spinoidal_interp/P_c_PR, zorder=10, linestyle='dotted', color=color_mech_spinoidal, label=r"$\textrm{Mechanical spinodal}$")

ax1.text(1.35, 1.8, r"$\textrm{Supercritical fluid}$", color="k", fontsize=18, ha="center", va="center")
ax1.text(1.35, 0.7, r"$\textrm{Two-phase region}$", color="k", fontsize=18, ha="center", va="center")

# Prandtl colormap
# plt.contourf(v_range / v_c_PR, P_range / P_c_PR, Pr_matrix, levels=100, cmap="coolwarm", zorder=0)
ax1.contourf(v_range / v_c_PR, P_range / P_c_PR, np.ones_like(Z_matrix), levels=1, colors=["lightgray"], zorder=0) # Background contour
contour_filled = ax1.contourf(v_range / v_c_PR, P_range / P_c_PR, Z_matrix, levels=100, cmap=cmap, zorder=0)


# Colorbar
cb2_ax = fig.add_axes([
    ax1.get_position().x0, 
    ax1.get_position().y1 + 0.015,  # Move it closer to ax1
    ax1.get_position().width, 
    0.02  # Height of colorbar
])
cb2 = Colorbar(ax = cb2_ax, mappable = contour_filled, orientation = 'horizontal', ticklocation = 'top')
cb2.set_label(r'$Z$', labelpad=5)
tick_values = np.linspace(contour_filled.norm.vmin, contour_filled.norm.vmax, 5)
cb2.set_ticks(tick_values)
cb2.ax.set_xticklabels([rf"${tick:.2f}$" for tick in tick_values])  # Format to 3 decimal places



ax1.legend(frameon=False)
ax1.set_ylim( [P_range[0]/P_c_PR, P_range[-1]/P_c_PR] )
ax1.set_xlim( [0.5, 3.5] )
ax1.set_xlabel(r"$1/(\rho/\rho_c)$")
ax1.set_ylabel(r"$P/P_c$")


plt.show()







