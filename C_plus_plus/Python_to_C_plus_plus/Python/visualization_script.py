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
output_iter     = 1000  # Select imported data iteration

name_file_out   = 'output_data_'  # Name of output data [-]
filename = name_file_out + str(output_iter) + '.csv'

# filename = 'converged_micronozzle.csv'
######################################################################
# Stretching factors: x = L*eta + A*( 0.5*L - L*eta )*( 1.0 - eta )*eta, with eta = ( l - 0.5 )/num_grid
# A < 0: stretching at ends; A = 0: uniform; A > 0: stretching at center
A_x = 0.0  # Stretching factor in x-direction
A_y = -1.0  # Stretching factor in y-direction
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

#############################################################################################

def spatial_discretization():
    physical_grid = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim]) # Physical grid (x, y, z)
    grid = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim])
    L_y = np.zeros(num_grid_x+2)
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

                # Channel
                # L_y[i] = L_y_0

                # # Parabolic
                # L_y[i] = L_y_0 + (grid[i][j][k][0]-L_x/2)*(grid[i][j][k][0]-L_x/2)

                # # Periodic
                # if ( grid[i][j][k][0] > L_x/2.0 - 0.7*Rc and grid[i][j][k][0] < L_x/2.0 + 0.7*Rc ):
                #     L_y[i] = L_y_0 + np.sqrt( Rc**2.0 - ( L_x/2.0 - grid[i][j][k][0] )**2.0 )
                # else:
                #     L_y[i] = 1.35*L_y_0

                # # Tube with a convergent zone
                # if ( grid[i][j][k][0] > L_x/2.0 and grid[i][j][k][0] < L_x/2.0 + 0.7*Rc):
                #     L_y[i] = L_y_0 + np.sqrt( Rc**2.0 - ( L_x/2.0 - grid[i][j][k][0] )**2.0 )
                # elif grid[i][j][k][0] > L_x/2.0 :
                #     L_y[i] = 1.35*L_y_0  
                # else:
                #     L_y[i] = 1.5*L_y_0  
                
                # # Tube with a convergent zone
                # if ( grid[i][j][k][0] > 1.0*L and grid[i][j][k][0] < 3.0*L):
                #     m = (L_y_f-L_y_0)/(2.0*L)
                #     L_y[i] = (L_y_0 - m*L) + m*grid[i][j][k][0] # Example geometry -- a linearly expanding duct
                # elif grid[i][j][k][0] < 1.0*L :
                #     L_y[i] = L_y_0  
                # else:
                #     L_y[i] = L_y_f              

                # # Ramp
                # L_y[i] = L_y_0 + (L_y_f-L_y_0)/L_x*grid[i][j][k][0] # Example geometry -- a linearly expanding duct
                
                # # Hyperbolic tangent
                # # 1. Extract the current X coordinate for readability
                # x_coord = grid[i][j][k][0]
                # # # 2. Define a scaling/sharpness parameter (s)
                # # # s = 3.0 is a good default where the transition finishes right at the boundaries.
                # s = 3.5 
                # # # 3. The tanh geometry transformation
                # if x_coord < 1.5*L:
                #     L_y[i] = L_y_0
                # else:
                #     L_y[i] = L_y_0 + (L_y_f - L_y_0) * 0.5 * (1.0 + np.tanh(s * (2.0 * (x_coord - 1.5*L) / (L_x - 1.5*L) - 1.0)))

                # Convergent-Divergent nozzle
                # Clean variable for the current x-coordinate
                x = grid[i][j][k][0]
                # print('x: ', x, 'x_c: ', x_c, 'x2: ', x2, 'x1: ', x1, 'x_t: ', x_t, 'x_exp: ', x_exp)

                if x <= x_c:
                    L_y[i] = r_c        
                elif x_c < x <= x2:
                    # Arc 2 (Centered at x_c)
                    L_y[i] = r_c - R2 * (1 - np.sqrt(1 - ((x - x_c) / R2)**2))                    
                elif x2 < x <= x1:
                    # Straight convergent section
                    L_y[i] = r1 - (x - x1) * math.tan(theta_rad)                    
                elif x1 < x <= x_t:
                    # Arc 1 (Centered at x_t)
                    L_y[i] = r_t + R1 * (1 - np.sqrt(1 - ((x - x_t) / R1)**2))                    
                elif x_t < x <= x_exp:
                    # Expansion Arc (Centered at x_t)
                    L_y[i] = r_t + Rexp * (1 - np.sqrt(1 - ((x - x_t) / Rexp)**2))                  
                else:
                    # Straight divergent section
                    L_y[i] = r_exp + (x - x_exp) * math.tan(alpha_rad)

                # x_lim = 0.9*L_con
                # if x < x_lim:
                #     L_y[i] = L_y_0 + (L_y_f - L_y_0) * 0.5 * (1.0 + np.tanh(s * (2.0 * x_coord / L_con - 1.0)))
                # else:
                #     x_coord = x_lim
                #     r1 = L_y_0 + (L_y_f - L_y_0) * 0.5 * (1.0 + np.tanh(s * (2.0 * x_coord / L_con - 1.0)))
                #     L_y[i] = r1 + (x - x_lim) * math.tan(alpha_rad)
    return L_y

def read_file_as_variables(file_name):
    # Temporary storage to handle the file reading safely
    meta = {}
    
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            vals = line.strip().split(',')
            if i == 1:
                meta.update({'time': float(vals[0]), 'time_iter': int(vals[1]), 'num_x': int(vals[2]), 'num_y': int(vals[3]), 'num_z': int(vals[4])})
            elif i == 3:
                meta.update({'x_0': float(vals[0]), 'y_0': float(vals[1]), 'z_0': float(vals[2]), 'L': float(vals[3]), 'L_x': float(vals[4]), 'L_y_0': float(vals[5]), 'L_y_f': float(vals[6]), 'L_z': float(vals[7]), 'Rc': float(vals[8])})
            elif i == 5:
                meta.update({'r_t': float(vals[0]), 'r_c': float(vals[1]), 'R1_rt': float(vals[2]), 'R2_R1': float(vals[3]), 'Rexp_rt': float(vals[4]), 'theta': float(vals[5]), 'alpha': float(vals[6]), 'L_N': float(vals[7]), 'L_c': float(vals[8])})
            elif i > 5:
                break

    # Unpack into individual variables
    time      = meta['time']
    time_iter = meta['time_iter']
    num_grid_x = meta['num_x']
    num_grid_y = meta['num_y']
    num_grid_z = meta['num_z']
    x_0       = meta['x_0']
    y_0       = meta['y_0']
    z_0       = meta['z_0']
    L         = meta['L']
    L_x       = meta['L_x']
    L_y_0     = meta['L_y_0']
    L_y_f     = meta['L_y_f']
    L_z       = meta['L_z']
    Rc        = meta['Rc']
    r_t       = meta['r_t']
    r_c       = meta['r_c']
    R1_rt     = meta['R1_rt']
    R2_R1     = meta['R2_R1'] # Ensure this matches your data source
    Rexp_rt   = meta['Rexp_rt']
    theta     = meta['theta']
    alpha     = meta['alpha']
    L_N       = meta['L_N']
    L_c       = meta['L_c']
    
    return time, time_iter, num_grid_x, num_grid_y, num_grid_z, x_0, y_0, z_0, L, L_x, L_y_0, L_y_f, L_z, Rc, r_t, r_c, R1_rt, R2_R1, Rexp_rt, theta, alpha, L_N, L_c

# Usage
time, time_iter, num_grid_x, num_grid_y, num_grid_z, x_0, y_0, z_0, L, L_x, L_y_0, L_y_f, L_z, Rc, r_t, r_c, R1_rt, R2_R1, Rexp_rt, theta, alpha, L_N, L_c = read_file_as_variables(filename)

R1 = r_t * R1_rt # Convergent-Throat arc radius
R2 = R1 * R2_R1 # Chamber-Convergent arc radius
Rexp = r_t * Rexp_rt # Expansion arc radius
theta_rad = theta * math.pi / 180 # Deg -> rad
alpha_rad = alpha * math.pi / 180 # Deg -> rad

# Segment points (Fixed geometry relationships)
x_c = L_c
r2 = r_c - R2 * (1 - math.cos(theta_rad))
r1 = r_t + R1 * (1 - math.cos(theta_rad))

x2 = x_c + R2 * math.sin(theta_rad)
# Distance between x2 and x1 must account for the change in radius (r2 to r1)
x1 = x2 + (r2 - r1) / math.tan(theta_rad) 

# Throat location is shifted by R1 * sin(theta), not just R1
x_t = x1 + R1 * math.sin(theta_rad) 
x_exp = x_t + Rexp * math.sin(alpha_rad)
r_exp = r_t + Rexp * (1 - math.cos(alpha_rad))

# Load the data - handles headers automatically or assumes none
print('! Importing data')
df = pd.read_csv(filename, skiprows=7,header=None, encoding='latin1')
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

                    
L_y = spatial_discretization()

# Reshape 3D vectors into 3D arrays
x   = np.reshape(x, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
y   = np.reshape(y, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
z   = np.reshape(z, (num_grid_x+2, num_grid_y+2, num_grid_z+2))
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

x_2d = x[:,:,1]
y_2d = y[:,:,1]
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

for i in range(0,num_grid_x+2):
    for j in range(0,num_grid_y+2):
        for k in range(0,num_grid_z+2):
            physical_plane[i][j][k][0] = x[i][j][k]
            physical_plane[i][j][k][1] = y[i][j][k]
            physical_plane[i][j][k][2] = z[i][j][k]

            physical_plane[i][j][k][1] *= L_y[i] 

dz = abs(physical_plane[1, 1, 0, 2] - physical_plane[1, 1, 1, 2])

mass_flow = np.zeros(num_grid_x)

for i in range(1, num_grid_x+1):
    for j in range(1, num_grid_y+1):
        # --- INLET (i=0 e i=1) ---
        y_low_i = 0.5 * (physical_plane[i, j-1, 1, 1] + physical_plane[i, j, 1, 1])
        y_up_i  = 0.5 * (physical_plane[i, j+1, 1, 1] + physical_plane[i, j, 1, 1])
        dy_i = y_up_i - y_low_i

        y_low_p = 0.5 * (physical_plane[i+1, j-1, 1, 1] + physical_plane[i+1, j, 1, 1])
        y_up_p  = 0.5 * (physical_plane[i+1, j+1, 1, 1] + physical_plane[i+1, j, 1, 1])
        dy_p = y_up_p - y_low_p 

        y_low_m = 0.5 * (physical_plane[i-1, j-1, 1, 1] + physical_plane[i-1, j, 1, 1])
        y_up_m  = 0.5 * (physical_plane[i-1, j+1, 1, 1] + physical_plane[i-1, j, 1, 1])
        dy_m = y_up_m - y_low_m

        dy_1 = 0.5 * ((y_up_i + y_up_m) - (y_low_i + y_low_m))
        dy_2 = 0.5 * ((y_up_i + y_up_p) - (y_low_i + y_low_p))

        area_1 = dz*dy_1
        area_2 = dz*dy_2

        rho_surface_in = 0.5 * (rho_2d[i-1, j] + rho_2d[i, j])
        rho_surface_out = 0.5 * (rho_2d[i+1,j] + rho_2d[i,j])
        u_surface_in   = 0.5 * (u_2d[i-1, j] + u_2d[i, j])
        u_surface_out = 0.5 * (u_2d[i+1,j] + u_2d[i,j])

        cell_m_in = rho_surface_in * u_surface_in * area_1
        cell_m_out = rho_surface_out * u_surface_out * area_2

        print('At cell i = ' + str(i) + ', j = ' + str(j) + ', m_dot_in = ' + str(cell_m_in) + ' kg/s & m_dot_out = ' + str(cell_m_out) + ' kg/s | Difference: ' + str((cell_m_in - cell_m_out)/max(cell_m_in,cell_m_out)))

        # mass_flow[i-1] += rho_surface_in * u_surface_in * dy_in * dz

# char_massflow = mass_flow[0]
# mass_flow /= char_massflow

# plt.plot(physical_plane[1:num_grid_x+1,1,1,0],mass_flow)
# plt.xlabel('x [m]')
# plt.ylabel('mass flow [kg/s]')
# plt.grid()


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
# plt.scatter(physical_plane[1:num_grid_x,1:num_grid_y,1,0], physical_plane[1:num_grid_x,1:num_grid_y,1,1], c=P_2d[1:num_grid_x,1:num_grid_y]/1e5, cmap='viridis', s=5)
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
                       cmap='viridis', s=5)
ax1.plot(contour_x[0:num_grid_x+1], contour_y[0:num_grid_x+1], linewidth=2.5, color='black')

cbar1 = fig.colorbar(surf1, ax=ax1, format='%.4f')
cbar1.set_label('Pressure [bar]')
ax1.set_ylim(0, None)
ax1.set_xlabel('Axis distance [m]')
ax1.set_ylabel('Radial distance [m]')

# --- Top right corner - Surface (Temperature) ---
surf2 = ax2.scatter(physical_plane[1:num_grid_x,1:num_grid_y,1,0], 
                       physical_plane[1:num_grid_x,1:num_grid_y,1,1], 
                       c=T_2d[1:num_grid_x,1:num_grid_y], 
                       cmap='viridis', s=5)
ax2.plot(contour_x[0:num_grid_x+1], contour_y[0:num_grid_x+1], linewidth=2.5, color='black')

cbar2 = fig.colorbar(surf2, ax=ax2, format='%.4f')
cbar2.set_label('Temperature [K]')
ax2.set_ylim(0, None)
ax2.set_xlabel('Axis distance [m]')
ax2.set_ylabel('Radial distance [m]')

# --- Bottom left corner - Velocity Vector Field ---
# 1. Calcular la magnitud de la velocidad para el mapa de colores
vel_mag = np.sqrt(u_2d**2 + v_2d**2)

# 2. CALCULAR EL ESCALADO DINÁMICO SEGÚN LA VELOCIDAD MÁXIMA
max_vel = np.max(vel_mag)

# Medimos el ancho total del dominio en metros para que sea independiente de la malla
x_min = np.min(physical_plane[1:num_grid_x, 1:num_grid_y, 1, 0])
x_max = np.max(physical_plane[1:num_grid_x, 1:num_grid_y, 1, 0])
domain_width = x_max - x_min

# Queremos que la flecha más rápida mida el 4% (0.04) del ancho del dominio. 
# Puedes ajustar este 0.04 (más grande = flechas más largas; más chico = flechas más cortas)
desired_max_arrow_length = 0.05 * domain_width 

# El scale correcto para 'scale_units=xy' es: velocidad_maxima / longitud_deseada
dynamic_scale = max_vel / desired_max_arrow_length

# 3. Generar el campo vectorial usando quiver en ax3

surf3 = ax3.quiver(physical_plane[1:num_grid_x+1, 1:num_grid_y+1, 1, 0], 
                   physical_plane[1:num_grid_x+1, 1:num_grid_y+1, 1, 1], 
                   u_2d[1:num_grid_x+1, 1:num_grid_y+1], 
                   v_2d[1:num_grid_x+1, 1:num_grid_y+1], 
                   vel_mag[1:num_grid_x+1, 1:num_grid_y+1],
                   cmap='viridis', 
                   scale=dynamic_scale,  # Escala adaptada automáticamente a la Vmax
                   pivot='mid',          # Centra la flecha en el punto de la malla
                   angles='xy',          # Dirección física real alineada con los ejes
                   scale_units='xy',
                   linewidth=1.5)     # Escala proporcional al plano físico

# Contorno negro original
# ax3.plot(contour_x[0:num_grid_x+1], contour_y[0:num_grid_x+1], linewidth=2.5, color='black')

# Configuración de la barra de colores y etiquetas
cbar3 = fig.colorbar(surf3, ax=ax3, format='%.2f')
cbar3.set_label('Velocity Magnitude [m/s]')
ax3.set_ylim(0, None)
ax3.set_xlabel('Axis distance [m]')
ax3.set_ylabel('Radial distance [m]')

# --- Gráfico en ax4 (Presión en el eje izquierdo) ---
color_p = 'tab:blue'  # Puedes elegir el color que prefieras
ax4.plot(physical_plane[1:num_grid_x, 1, 1, 0], P_2d[1:num_grid_x, 1] / 1.0e5, 
         linewidth=1.5, color=color_p, label='Pressure')

ax4.set_xlabel('Axis distance [m]')
ax4.set_ylabel('Pressure [bar]', color=color_p)
ax4.set_ylim(0, None)
ax4.tick_params(axis='y', labelcolor=color_p)  # Colorea los números del eje Y izquierdo
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

# --- Crear el eje gemelo para el Número de Mach (Eje derecho) ---
ax4_Ma = ax4.twinx()

color_ma = 'tab:orange'  # Color contrastante para Mach
ax4_Ma.plot(physical_plane[1:num_grid_x, 1, 1, 0], Ma_2d[1:num_grid_x, 1], 
            linewidth=1.5, color=color_ma, linestyle='--', label='Mach Number')

ax4_Ma.set_ylabel('Mach Number [-]', color=color_ma)
# Si quieres que el número de Mach también empiece en 0, descomenta la siguiente línea:
# ax4_Ma.set_ylim(0, None) 
ax4_Ma.tick_params(axis='y', labelcolor=color_ma)  # Colorea los números del eje Y derecho

# --- Opcional: Combinar ambas leyendas en ax4 ---
# Esto evita que tengas que usar plt.legend() y que las cajas se superpongan
lines4, labels4 = ax4.get_legend_handles_labels()
lines4_Ma, labels4_Ma = ax4_Ma.get_legend_handles_labels()
ax4.legend(lines4 + lines4_Ma, labels4 + labels4_Ma, loc='upper right')

# Show subplot
plt.tight_layout()
# plt.show()

# png_filename = f'plot_iter_{output_iter}.png'

# Save figure
# plt.savefig(png_filename, dpi=600, bbox_inches='tight')

# plt.close(fig)

plt.figure()
plt.plot(contour_x[0:num_grid_x+1], contour_y[0:num_grid_x+1], linewidth=2.5, color='black')
plt.quiver(physical_plane[1:num_grid_x, 1:num_grid_y, 1, 0], 
                   physical_plane[1:num_grid_x, 1:num_grid_y, 1, 1], 
                   u_2d[1:num_grid_x, 1:num_grid_y], 
                   v_2d[1:num_grid_x, 1:num_grid_y], 
                   vel_mag[1:num_grid_x, 1:num_grid_y],
                   cmap='viridis', 
                   scale=dynamic_scale,  # Escala adaptada automáticamente a la Vmax
                   pivot='mid',          # Centra la flecha en el punto de la malla
                   angles='xy',          # Dirección física real alineada con los ejes
                   scale_units='xy',
                   linewidth=1.5)     # Escala proporcional al plano físico
plt.colorbar()
plt.show()

