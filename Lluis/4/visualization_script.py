import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

# from rhea_thermodynamics_transport_coefficients import BaseThermodynamicModel
# from rhea_thermodynamics_transport_coefficients import IdealGasModel
# from rhea_thermodynamics_transport_coefficients import PengRobinsonModel
# from rhea_thermodynamics_transport_coefficients import CoolPropModel
# from rhea_thermodynamics_transport_coefficients import BaseTransportCoefficients
# from rhea_thermodynamics_transport_coefficients import ConstantTransportCoefficients
# from rhea_thermodynamics_transport_coefficients import LowPressureGasTransportCoefficients
# from rhea_thermodynamics_transport_coefficients import HighPressureTransportCoeficients
# from rhea_thermodynamics_transport_coefficients import CoolPropTransportCoefficients

########## SET PARAMETERS ############

###################### DATA IMPORT SETTINGS ##########################
output_iter     = 200  # Select imported data iteration

num_grid_x      = 32  # Number of internal grid points in the x-direction
num_grid_y      = 16   # Number of internal grid points in the y-direction
num_grid_z      = 1    # Number of internal grid points in the z-direction

name_file_out   = 'output_data_'  # Name of output data [-]
filename = name_file_out + str(output_iter) + '.csv'
######################################################################

###################### GEOMETRY CONFIGURATION ########################
rt              = 1.0e-3  # Throat radius [m]
rc              = 2.5e-3  # Chamber radius [m]
R1_rt           = 3.0  # Convergent-throat arc ratio as (R1/rt)
R2_R1           = 5.0  # Chamber-convergent arc ratio as (R2/R1)
Rexp_rt         = 2.0  # Expansion arc ratio as (Rexp/rt)
theta           = 15.0  # Convergent segment inclination angle [deg]
alpha           = 5.0  # Conical nozzle half-angle [deg]
L_N             = 30e-3  # Conical section length [m]
L_c             = 3e-3 # Chamber section length [m]

# Stretching factors: x = L*eta + A*( 0.5*L - L*eta )*( 1.0 - eta )*eta, with eta = ( l - 0.5 )/num_grid
# A < 0: stretching at ends; A = 0: uniform; A > 0: stretching at center
A_x = 0.0  # Stretching factor in x-direction
A_y = 0.0  # Stretching factor in y-direction
A_z = 0.0  # Stretching factor in z-direction
######################################################################

##################################### FLUID PROPERTIES ######################################
substance               = 'CO2'
R_specific              = 188.92  # Specific gas constant [J/(kg·K)]
gamma                   = 1.289  # Heat capacity ratio (ideal-gas) [-]
molecular_weight        = 0.04401  # Molecular weight [kg/mol]
acentric_factor         = 0.22394  # Acentric factor [-]
critical_temperature    = 304.1282  # Critical temperature [K]
critical_pressure       = 7377270.6  # Critical pressure [Pa]
critical_molar_volume   = 0.0000941189  # Critical molar volume [m3/mol]
NASA_coefficients       = [4.6365111000000000000000,
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
                          -47328.105000000000000000]  # NASA 7-coefficient polynomial (15 values)
T_0                     = 273.0  # Reference temperature [K]
S_mu                    = 222.0  # Sutherland's dynamic viscosity constant [K]
S_kappa                 = 1800.0  # Sutherland's thermal conductivity constant [K]
dipole_moment           = 0.0  # Dipole moment [D]
association_factor      = 0.0  # Association factor [-]
#############################################################################################

#############################################################################################
x_0 = 0.0
y_0 = 0.0
z_0 = 0.0

L = 1.0
L_y = np.ones(num_grid_x+2) * L
L_y_0 = 1.0 * L
L_y_f = 0.5 * L

L_x = 2.0
L_y_var = 1.0
L_z = 0.01

A_x = 0.0
A_y = 0.0
A_z = 0.0
#############################################################################################

def spatial_discretization( grid):
    physical_grid = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim]) # Physical grid (x, y, z)
    # All points
    for i in range( 0, num_grid_x + 2 ):    
        for j in range( 0, num_grid_y + 2 ):    
            for k in range( 0, num_grid_z + 2 ):
                # Evenly-spaced (dummy variable) cell centroids
                eta_x = ( i - 0.5 )/num_grid_x
                eta_y = ( j - 0.5 )/num_grid_y
                eta_z = ( k - 0.5 )/num_grid_z
                # Unevenly-spaced (stretched) cell centroids
                grid[i][j][k][0] = x_0 + L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x

                ### Define geometry contour
                # Ramp
                L_y[i] = L_y_0 + (L_y_f-L_y_0)/L_x*grid[i][j][k][0] # Example geometry -- a linearly expanding duct
                
                # # Hyperbolic tangent
                # # 1. Extract the current X coordinate for readability
                # x_coord = grid[i][j][k][0]
                # # 2. Define a scaling/sharpness parameter (s)
                # # s = 3.0 is a good default where the transition finishes right at the boundaries.
                # s = 3.5 
                # # 3. The tanh geometry transformation
                # L_y[i] = L_y_0 + (L_y_f - L_y_0) * 0.5 * (1.0 + np.tanh(s * (2.0 * x_coord / L_x - 1.0)))

                grid[i][j][k][1] = y_0 + L_y[i]*eta_y + A_y*( 0.5*L_y[i] - L_y[i]*eta_y )*( 1.0 - eta_y )*eta_y
                grid[i][j][k][2] = z_0 + L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z
                # Adjust (symmetric) boundary cell centroids
                if( grid[i][j][k][0] < x_0 ):
                    eta_x = ( 1.0 - 0.5 )/num_grid_x
                    grid[i][j][k][0] = x_0 - ( L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x )
                if( grid[i][j][k][0] > ( x_0 + L_x ) ):
                    eta_x = ( num_grid_x - 0.5 )/num_grid_x
                    grid[i][j][k][0] = x_0 + 2.0*L_x - ( L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x )
                if( grid[i][j][k][1] < y_0 ):
                    eta_y = ( 1.0 - 0.5 )/num_grid_y
                    grid[i][j][k][1] = y_0 - ( L_y[i]*eta_y + A_y*( 0.5*L_y[i] - L_y[i]*eta_y )*( 1.0 - eta_y )*eta_y )
                if( grid[i][j][k][1] > ( y_0 + L_y[i] ) ):
                    eta_y = ( num_grid_y - 0.5 )/num_grid_y
                    grid[i][j][k][1] = y_0 + 2.0*L_y[i] - ( L_y[i]*eta_y + A_y*( 0.5*L_y[i] - L_y[i]*eta_y )*( 1.0 - eta_y )*eta_y )
                if( grid[i][j][k][2] < z_0 ):
                    eta_z = ( 1.0 - 0.5 )/num_grid_z
                    grid[i][j][k][2] = z_0 - ( L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z )
                if( grid[i][j][k][2] > ( z_0 + L_z ) ):
                    eta_z = ( num_grid_z - 0.5 )/num_grid_z
                    grid[i][j][k][2] = z_0 + 2.0*L_z - ( L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z )


#################### SELECT THERMODYNAMIC AND TRANSPORT COEFFICIENTS MODEL ##################
# thermodynamics = IdealGasModel(R_specific, gamma)
# thermodynamics = PengRobinsonModel(molecular_weight, acentric_factor, critical_temperature, critical_pressure, critical_molar_volume, NASA_coefficients)
# thermodynamics = CoolPropModel(substance)
#############################################################################################
# transport_coefficients = ConstantTransportCoefficients(mu_ref, kappa_ref)
# transport_coefficients = LowPressureGasTransportCoefficients( mu_0, kappa_0, T_0, S_mu, S_kappa)
# transport_coefficients = HighPressureTransportCoefficients(molecular_weight, acentric_factor, critical_temperature, critical_molar_volume, NASA_coefficients, dipole_moment,association_factor)
# transport_coefficients = CoolPropTransportCoefficients(substance)
##############################################################################################

# x_pseudoboiling, y_pseudoboiling = np.loadtxt('pseudoboiling_line.txt', delimiter=',', skiprows=1, unpack=True)
# saturation_curve = np.load('VLE_curve.npy')
# x_saturation, y_saturation = saturation_curve
# x_saturation = x_saturation/critical_specific_volume
# y_saturation = y_saturation/critical_pressure

# def getSaturatedLiquidSpecificVolume(P_i, v_i):
#     v_ord_liquid = np.sort(x_saturation[x_saturation < 1.0])
#     P_ord_liquid = np.sort(y_saturation[x_saturation < 1.0])

#     v_ord_gas = np.sort(x_saturation[x_saturation > 1.0])
#     P_ord_gas = np.sort(y_saturation[x_saturation > 1.0])

#     # Saturated liquid

#     idx = np.searchsorted(v_ord_liquid, v_i)
    
#     if idx == len(v_ord_liquid):
#         v_sl_i = v_ord_liquid[idx-1]
#     elif idx == 0:
#         v_sl_i = v_ord_liquid[idx]
#     else:
#         v_1 = v_ord_liquid[idx-1]; P_1 = P_ord_liquid[idx-1]
#         v_2 = v_ord_liquid[idx]; P_2 = P_ord_liquid[idx]
#         v_sl_i = (P_i-P_1)/(P_2-P_1)*(v_2-v_1)+v_1

#     # Saturated gas

#     idx = np.searchsorted(v_ord_gas, v_i)

#     if idx == len(v_ord_gas):
#         v_sg_i = v_ord_gas[idx-1]
#     elif idx == 0:
#         v_sg_i = v_ord_gas[idx]
#     else:
#         v_1 = v_ord_gas[idx-1]; P_1 = P_ord_gas[idx-1]
#         v_2 = v_ord_gas[idx]; P_2 = P_ord_gas[idx]
#         v_sg_i = (P_i-P_1)/(P_2-P_1)*(v_2-v_1)+v_1


#     return v_sl_i, v_sg_i

# def getPseudoboilingSpecificVolume(P_i, v_i):
#     v_ord = np.sort(x_pseudoboiling)
#     P_ord = np.sort(y_pseudoboiling)

#     idx = np.searchsorted(v_ord, v_i)

#     if idx == len(v_ord):
#         v_cpmax_i = v_ord[idx-1]
#     elif idx == 0:
#         v_cpmax_i = v_ord[idx]
#     else:
#         v_1 = v_ord[idx-1]; P_1 = P_ord[idx-1]
#         v_2 = v_ord[idx]; P_2 = P_ord[idx]
#         v_cpmax_i = (P_i-P_1)/(P_2-P_1)*(v_2-v_1)+v_1

#     return v_cpmax_i

# header_string = '# x [m], y [m], z [m], rho [kg/m3], u [m/s], v [m/s], w [m/s], E [J/kg], s [J/(kg·K)], P [Pa], T [K], sos [m/s]\n'
        
# Load the data - handles headers automatically or assumes none
print('! Importing data')
df = pd.read_csv(filename, skiprows=3,header=None, encoding='latin1')
num_sptl_dim = 3  # Number of spatial dimensions (fixed value)

# Extract columns
x   = df.iloc[:,0]
y   = df.iloc[:,1]
z   = df.iloc[:,2]
rho = df.iloc[:,3]
u   = df.iloc[:,4]
v   = df.iloc[:,5]
w   = df.iloc[:,6]
E   = df.iloc[:,7]
s   = df.iloc[:,8]
P   = df.iloc[:,9]
T   = df.iloc[:,10]
sos = df.iloc[:,11]
Ma = np.sqrt(u**2+v**2+w**2)/sos

### Mesh coordinates ... two positions added for boundary points
physical_plane = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim])  # 3-D positions of the physical plane mesh
computational_plane = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim]) # 3-D positions of the computational plane grid
drc_dx = np.zeros([num_grid_x + 2])

R1 = rt * R1_rt
R2 = R1 * R2_R1
Rexp = rt * Rexp_rt

theta = theta*math.pi/180
alpha = alpha*math.pi/180

x1 = -R1*math.sin(theta)
r1 = rt + R1*(1-math.cos(theta))
r2 = rc + R2 * ( math.cos(theta) - 1 )
x2 = x1 - (r2 - r1) / math.tan(theta)
xc = x2 - R2*math.sin(theta)
xexp = Rexp*math.sin(alpha)
                    
spatial_discretization(physical_plane)
# physical_plane /= 100.0

# Reshape 3D vectors into 3D arrays
x   = np.reshape(x, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
y   = np.reshape(y, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
rho = np.reshape(rho, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
u   = np.reshape(u, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
v   = np.reshape(v, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
w   = np.reshape(w, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
E   = np.reshape(E, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
s   = np.reshape(s, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
P   = np.reshape(P, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
T   = np.reshape(T, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
sos = np.reshape(sos, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
Ma  = np.reshape(Ma, (num_grid_x+2, num_grid_y+2, num_grid_z+2))

# Select middle plane
print('! Extracting and reshaping data structure')

rho_2d = rho[:,:,1]
u_2d = u[:,:,1]
v_2d = v[:,:,1]
w_2d = w[:,:,1]
V_2d = np.sqrt(u_2d**2+v_2d**2+w_2d**2)
E_2d = E[:,:,1]
s_2d = s[:,:,1]
P_2d = P[:,:,1]
T_2d = T[:,:,1]
sos_2d = sos[:,:,1]
Ma_2d = Ma[:,:,1]

contour_x = physical_plane[1:num_grid_x,-2,1,0] # 0.5 * ( physical_plane[1:num_grid_x,-1,1,0] + physical_plane[1:num_grid_x,-2,1,0])
contour_y = physical_plane[1:num_grid_x,-2,1,1] # 0.5 * ( physical_plane[1:num_grid_x,-1,1,1] + physical_plane[1:num_grid_x,-2,1,1])

# cv_2d = np.zeros([num_grid_x+2,num_grid_y+2])
# cp_2d = np.zeros([num_grid_x+2,num_grid_y+2])
# z_2d = np.zeros([num_grid_x+2,num_grid_y+2])
# specVol_2d = np.zeros([num_grid_x+2,num_grid_y+2])
# state_2d = np.zeros([num_grid_x+2,num_grid_y+2])

# print('! Obtaining thermodynamic and state properties')
# for i in range(1,num_grid_x+1):
#     for j in range(1,num_grid_y+1):
#         cv = -1.0; cp = -1.0
#         cv_2d[i][j], cp_2d[i][j] = thermodynamics.calculateSpecificHeatCapacities(cv,cp,P_2d[i][j],T_2d[i][j],rho_2d[i][j])
#         specVol_2d[i][j] = 1/rho_2d[i][j]
#         z_2d[i][j] = (P_2d[i][j]*specVol_2d[i][j])/(R_specific*T_2d[i][j])

#         v_sl_i, v_sg_i = getSaturatedLiquidSpecificVolume(P_2d[i][j]/critical_pressure,specVol_2d[i][j]/critical_specific_volume)
#         v_cpmax_i = getPseudoboilingSpecificVolume(P_2d[i][j]/critical_pressure,specVol_2d[i][j]/critical_specific_volume)

#         if P_2d[i][j]/critical_pressure > 1.0:
#             if specVol_2d[i][j] < v_cpmax_i*critical_specific_volume:
#                 state_2d[i][j] = 1                                                      # SUPERCRITICAL LIQUID-LIKE
#             else:
#                 state_2d[i][j] = 2                                                      # SUPERCRITICAL GAS-LIKE
#         elif P_2d[i][j] < 1.0:
#             if specVol_2d[i][j] < v_sl_i*critical_specific_volume:
#                 state_2d[i][j] = 3                                                      # SUBCRITICAL LIQUID
#             elif specVol_2d[i][j] > v_sl_i*critical_specific_volume and specVol_2d[i][j] < v_sg_i*critical_specific_volume:
#                 state_2d[i][j] = 4                                                      # SUBCRITICAL 2-PHASE
#             else:
#                 state_2d[i][j] = 5                                                      # SUBCRITICAL GAS

# plt.figure()
# plt.scatter(physical_plane[1:num_grid_x,1:num_grid_y,1,0], physical_plane[1:num_grid_x,1:num_grid_y,1,1], c=P_2d[1:num_grid_x,1:num_grid_y]/1e5, cmap='viridis', s=2)
# plt.colorbar()
# plt.show()

# exit()

print('! Generating subplots')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

# --- Add the iteration number as the main figure title ---
fig.suptitle(f'Iteration: {output_iter}', fontsize=14, fontweight='bold')

# --- Top left corner - Surface (Pressure) ---
surf1 = ax1.scatter(physical_plane[1:num_grid_x,1:num_grid_y,1,0], 
                       physical_plane[1:num_grid_x,1:num_grid_y,1,1], 
                       c=P_2d[1:num_grid_x,1:num_grid_y]/1e5, 
                       cmap='viridis', s=2)
ax1.plot(contour_x[0:num_grid_x+1], contour_y[0:num_grid_x+1], linewidth=2.5, color='black')

cbar1 = fig.colorbar(surf1, ax=ax1, format='%.4f')
cbar1.set_label('Pressure [bar]')
ax1.set_ylim(0, None)
ax1.set_xlabel('Axis distance [m]')
ax1.set_ylabel('Radial distance [m]')

# --- Top right corner - Surface (Temperature) ---
surf2 = ax2.scatter(physical_plane[1:num_grid_x,1:num_grid_y,1,0], 
                       physical_plane[1:num_grid_x,1:num_grid_y,1,1], 
                       c=rho_2d[1:num_grid_x,1:num_grid_y], 
                       cmap='viridis', s=2)
ax2.plot(contour_x[0:num_grid_x+1], contour_y[0:num_grid_x+1], linewidth=2.5, color='black')

cbar2 = fig.colorbar(surf2, ax=ax2, format='%.4f')
cbar2.set_label('Temperature [K]')
ax2.set_ylim(0, None)
ax2.set_xlabel('Axis distance [m]')
ax2.set_ylabel('Radial distance [m]')

# --- Bottom left corner - Scatter (Mach) ---
surf3 = ax3.pcolormesh(physical_plane[1:num_grid_x,1:num_grid_y,1,0], 
                       physical_plane[1:num_grid_x,1:num_grid_y,1,1], 
                       Ma_2d[1:num_grid_x,1:num_grid_y], 
                       cmap='viridis', shading='gouraud')
ax3.plot(contour_x[1:num_grid_x], contour_y[1:num_grid_x], linewidth=2.5, color='black')
ax3.set_ylim(0, None)

# Formato para la colorbar de Mach
cbar3 = fig.colorbar(surf3, ax=ax3, format='%.4f')
cbar3.set_label('Mach number')
ax3.set_xlabel('Axis distance [m]')
ax3.set_ylabel('Radial distance [m]')

# --- Bottom right corner - Plot (Pressure-Axis) ---
ax4.plot(physical_plane[1:num_grid_x,1,1,0], P_2d[1:num_grid_x,1]/1.0e5, linewidth=1.5)
ax4.set_xlabel('Axis distance [m]')
ax4.set_ylabel('Pressure [bar]')
ax4.set_ylim(0, None)
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

# Show subplot
plt.tight_layout()
plt.show()

png_filename = f'plot_iter_{output_iter}.png'

# Save figure
# plt.savefig(png_filename, dpi=600, bbox_inches='tight')

# plt.close(fig)

