# Geometry @1
# rt = 1mm
# rc = 3mm
# theta = 30
# R1_rt = 1.5
# R2_R1 = 1.5
# Rexp1_rt = 5
# Ln = 1.9mm
# alpha = 3.5

#### Numeric schemes
# ECKEP_flux
# HLLC_flux
# KGP_flux
# HES_flux

# Primitive variables:
#   - Density rho
#   - Velocities u, v, w
#   - Specific total energy E
#     ... E = e + ke
#     ... is the sum of internal energy e
#     ... and kinetic energy ke = (u*u + v*v + w*w)/2
# Conserved variables:
#   - Mass rho
#   - Momentum rho*u, rho*v, rho*w
#   - Total energy rho*E
# Thermodynamic state:
#   - Pressure P
#   - Temperature T
#   - Speed of sound sos
# Thermophysical properties:
#   - Specific gas constant R_specific
#   - Ratio of heat capacities gamma
#   - Dynamic viscosity mu
#   - Thermal conductivity kappa

########## PHYTON MODULES ##########
import os, sys
import numpy as np
import math
from numba import njit
import matplotlib
matplotlib.use('Agg')   # headless: no blocking windows
import matplotlib.pyplot as plt

# Resolve nozzlegeometry and rhea_thermodynamics_transport_coefficients from the parent
# directory so this legacy solver runs from inside ./comparison/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # self-contained: use this folder's module copies

########## LOAD ADDITIONAL CLASSES ##########
from rhea_thermodynamics_transport_coefficients import BaseThermodynamicModel
from rhea_thermodynamics_transport_coefficients import IdealGasModel
from rhea_thermodynamics_transport_coefficients import PengRobinsonModel
from rhea_thermodynamics_transport_coefficients import CoolPropModel
from rhea_thermodynamics_transport_coefficients import BaseTransportCoefficients
from rhea_thermodynamics_transport_coefficients import ConstantTransportCoefficients
from rhea_thermodynamics_transport_coefficients import LowPressureGasTransportCoefficients
from rhea_thermodynamics_transport_coefficients import HighPressureTransportCoeficients
from rhea_thermodynamics_transport_coefficients import CoolPropTransportCoefficients

import nozzlegeometry as nz_cont

########## SET PARAMETERS ############

######################## RESTART SETTINGS ############################
use_restart     = False  # Use restart [-] (fresh start for this 150->130 case)
name_restart    = 'output_data/output_data_symmetric_nozzle/output_data_3600.csv'  # Name of restart data [-]
######################################################################

###################### DATA OUTPUT SETTINGS ##########################
name_file_out   = 'output_data'  # Name of output data [-] (writes output_data_<iter>.csv in this folder)
output_iter     = 500  # Output data every given number of iterations
######################################################################

######################## INITIAL CONDITIONS ##########################
u_inlet         = 10.0  # Subsonic inlet velocity in the x-direction [m/s]
v_inlet         = 0.0  # Subsonic inlet velocity in the y-direction [m/s]
w_inlet         = 0.0  # Subsonic inlet velocity in the z-direction [m/s]
T_ref           = 400.0  # Subsonic inlet temperature [K]
P_ref           = 150.0e5  # Subsonic inlet pressure [Pa]
P_exit          = 130.0e5  # Subsonic outlet exit pressure [Pa] (single-phase window; weak shock near throat)
# mu_ref          = 0.0 # Reference dynamic viscosity [Pa·s]
# kappa_ref       = 0.0 # Reference thermal conductivity [W/mK]

mu_ref = 2.0e-5
kappa_ref = 0.025
######################################################################

###################### GEOMETRY CONFIGURATION ########################
rt              = 1.0e-3  # Throat radius [m]
rc              = 2.5e-3  # Chamber radius [m]
R1_rt           = 5.0  # Convergent-throat arc ratio as (R1/rt)
R2_R1           = 0.5   # Chamber-convergent arc ratio as (R2/R1) - matches the 4-zone nozzle
Rexp_rt         = 2.0  # Expansion arc ratio as (Rexp/rt)
theta           = 15.0  # Convergent segment inclination angle [deg]
alpha           = 20.0  # Conical nozzle half-angle [deg] - pronounced divergence (4-zone nozzle)
L_N             = 30e-3  # Conical section length [m]
L_c             = 5e-3 # Chamber section length [m]

# Stretching factors: x = L*eta + A*( 0.5*L - L*eta )*( 1.0 - eta )*eta, with eta = ( l - 0.5 )/num_grid
# A < 0: stretching at ends; A = 0: uniform; A > 0: stretching at center
A_x = 0.0  # Stretching factor in x-direction
A_y = 0.0  # Stretching factor in y-direction
A_z = 0.0  # Stretching factor in z-direction
######################################################################

######################### SOLVER SETTINGS ##########################
num_grid_x      = 50  # Number of internal grid points in the x-direction
num_grid_y      = 30  # Number of internal grid points in the y-direction
num_grid_z      = 1  # Number of internal grid points in the z-direction

CFL             = 0.1  # CFL coefficient
initial_time    = 0.0 # Initial time [s]
final_time      = 1.0 # Final time [s]

max_num_time_iter   = 100000  # Maximum number of time iterations
max_subr_iter       = 100  # Maximum number of iterations for subsonic inlet subroutine

transport_pressure_scheme           = False  # Select transporting pressure instead of total energy
artificial_compressibility_method   = False  # Activate artificial compressibility method
epsilon_acm                         = 0.01  # Relative error of artificial compressibility method ... it has to be small
#############################################################################################

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

### Nozzle geometry
R1 = rt * R1_rt # Convergent-Throat arc radius
R2 = R1 * R2_R1 # Chamber-Convergent arc radius
Rexp = rt * Rexp_rt # Expansion arc radius

theta = theta*math.pi/180 # Deg -> rad
alpha = alpha*math.pi/180 # Deg -> rad

# Segment points
x1 = -R1*math.sin(theta)
r1 = rt + R1*(1-math.cos(theta))
r2 = rc + R2 * ( math.cos(theta) - 1 )
x2 = x1 - (r2 - r1) / math.tan(theta)
xc = x2 - R2*math.sin(theta)
xexp = Rexp*math.sin(alpha)

# Coordinates origin
x_0 = xc   # Domain origin in x-direction [m] - chamber-arc start of the nozzle
y_0 = 0.0  # Domain origin in y-direction [m]
z_0 = 0.0  # Domain origin in z-direction [m]

# Geometry size
L = 1.0  # Cavity size [m]
L_x = abs(xc) + L_N   # Domain length: chamber-arc start to nozzle exit
L_z = 0.01 * L  # Size of domain in z-direction

##### THERMODYNAMICS AND TRANSPORT MODELS #####
thermodynamics = PengRobinsonModel(molecular_weight, acentric_factor, critical_temperature, critical_pressure, critical_molar_volume, NASA_coefficients)
# thermodynamics = PengRobinsonModel(molecular_weight, acentric_factor, critical_temperature, critical_pressure, critical_molar_volume, NASA_coefficients)
# thermodynamics = CoolPropModel(substance)
transport_coefficients = HighPressureTransportCoeficients(molecular_weight, acentric_factor, critical_temperature, critical_molar_volume, NASA_coefficients, dipole_moment, association_factor)
# transport_coefficients = LowPressureGasTransportCoefficients( mu_0, kappa_0, T_0, S_mu, S_kappa)
# transport_coefficients = HighPressureTransportCoefficients(molecular_weight, acentric_factor, critical_temperature, critical_molar_volume, NASA_coefficients, dipole_moment,association_factor)
# transport_coefficients = CoolPropTransportCoefficients(substance)

### Fixed parameters
num_sptl_dim = 3  # Number of spatial dimensions (fixed value)
rk_order = 3  # Time-integration Runge-Kutta order (fixed value)
epsilon = 1.0e-10  # Small epsilon number (fixed value)
rel_tol = 1.0e-3  # Subroutine error tolerance (fixed value)

### Auxiliar parameters
time = 0.0
time_iter = 0
delta_t = 0.0  # Initialize time step
P_thermo = 0.0  # Initialize thermodynamic pressure for artificial compressibility method
alpha_acm = 1.0  # Initialize speedup factor of artificial compressibility method

########## ALLOCATE MEMORY ##########

### Mesh coordinates ... two positions added for boundary points
physical_plane = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim])  # 3-D positions of the physical plane mesh
computational_plane = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, num_sptl_dim]) # 3-D positions of the computational plane grid
drc_dx = np.zeros([num_grid_x + 2])
drc_dx_dx = np.zeros([num_grid_x + 2])

### Primitive, conserved and thermodynamic variables ... two positions added for boundary points
rho_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhof
u_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of u
v_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of v
w_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of w
E_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of E
s_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of s
rhou_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhou
rhov_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhov
rhow_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhow
rhoE_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhoE
P_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of P
T_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of T
sos_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of sos

### Transport coefficients ... two positions added for boundary points
mu_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of mu
kappa_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of kappa

### Time integration variables ... two positions added for boundary points
rho_0_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D old field of rho
rhou_0_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D old field of rhou
rhov_0_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D old field of rhov
rhow_0_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D old field of rhow
rhoE_0_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D old field of rhoE
P_0_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D old field of P

### Time integration fluxes ... two positions added for boundary points
rho_rk_fluxes = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, rk_order])  # 3-D Runge-Kutta fluxes of rho
rhou_rk_fluxes = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, rk_order])  # 3-D Runge-Kutta fluxes of rhou
rhov_rk_fluxes = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, rk_order])  # 3-D Runge-Kutta fluxes of rhov
rhow_rk_fluxes = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, rk_order])  # 3-D Runge-Kutta fluxes of rhow
rhoE_rk_fluxes = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, rk_order])  # 3-D Runge-Kutta fluxes of rhoE
P_rk_fluxes = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2, rk_order])  # 3-D Runge-Kutta fluxes of P

### Inviscid fluxes ... two positions added for boundary points
rho_inv_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D inviscid fluxes of rho
rhou_inv_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D inviscid fluxes of rhou
rhov_inv_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D inviscid fluxes of rhov
rhow_inv_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D inviscid fluxes of rhow
rhoE_inv_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D inviscid fluxes of rhoE

### Viscous fluxes ... two positions added for boundary points
rhou_vis_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D viscous fluxes of rhou
rhov_vis_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D viscous fluxes of rhov
rhow_vis_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D viscous fluxes of rhow
rhoE_vis_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D viscous fluxes of rhoE
work_vis_rhoe_flux = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D viscous fluxes of work rhoe

### Body forces ... two positions added for boundary points
f_rhou_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhou
f_rhov_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhov
f_rhow_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhow
f_rhoE_field = np.zeros([num_grid_x + 2, num_grid_y + 2, num_grid_z + 2])  # 3-D field of rhoE

rho_ref = -1.0; E_ref = -1.0
rho_ref, E_ref = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_ref,E_ref,P_ref,T_ref)
H_ref = E_ref + P_ref/rho_ref

########## DEFINITIONS ##########

### Initialize u, v, w, P and T variables
def initialize_uvwPT(u, v, w, P, T, grid):
    # sos_init = thermodynamics.calculateSoundSpeed(P_ref,T_ref,rho_ref)
    # P_init = np.linspace(P_ref,P_exit,num_grid_x+2)
    # T_init = np.linspace(T_ref,0.85*T_ref,num_grid_x+2)
    # u_init = np.linspace(u_inlet,u_inlet-5.0,num_grid_x+2)
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                # u[i][j][k] = u_init[i]
                # v[i][j][k] = v_inlet
                # w[i][j][k] = w_inlet
                # P[i][j][k] = P_init[i]
                # T[i][j][k] = T_init[i]

                rc = 0.5 * (grid[i][-1][k][1] + grid[i][-2][k][1])

                # u[i][j][k] = - (u_inlet/(rc)**2)*grid[i][j][k][1]**2 + u_inlet   # Parabolic velocity profile in the y-direction
                r_normalized = min(grid[i][j][k][1] / rc, 1.0)
                u[i][j][k] = u_inlet * (1.0 - r_normalized)**(1.0 / 7.0)
                v[i][j][k] = v_inlet
                w[i][j][k] = w_inlet
                P[i][j][k] = P_ref
                T[i][j][k] = T_ref
# print( u )
    # print( v )
    # print( w )
    # print( P )
    # print( T )


### Read file u, v, w, P and T variables
def read_file_uvwPT(u, v, w, P, T, grid):
    # Read 3-D data:
    # x, y, z, rho, u, v, w, E, s, P, T, sos

    # Open input file
    file_name = str(name_restart)
    data_file_in = open(file_name, 'r')

    # Read lines
    line_counter = 0
    while line := data_file_in.readline():
        # Read metadata ( time, time_iter )
        if (line_counter == 1):
            time = float( line.split(',')[0])
            time_iter = int( line.split(',')[1])
        # Read data ( x, y, z, u, v, w, P, T, sos )
        elif (line_counter > 2):
            grid_index = line_counter - 3
            k = grid_index % (num_grid_z + 2)
            j = (grid_index // (num_grid_z + 2)) % (num_grid_y + 2)
            i = grid_index // ((num_grid_y + 2) * (num_grid_z + 2))
            grid[i][j][k][0] = float(line.split(',')[0])
            grid[i][j][k][1] = float(line.split(',')[1])
            grid[i][j][k][2] = float(line.split(',')[2])
            # rho[i][j][k]     = float(line.split( ',' )[3])
            u[i][j][k] = float(line.split(',')[4])
            v[i][j][k] = float(line.split(',')[5])
            w[i][j][k] = float(line.split(',')[6])
            # E[i][j][k]       = float(line.split( ',' )[7])
            # s[i][j][k]       = float(line.split( ',' )[8])
            P[i][j][k] = float(line.split(',')[9])
            T[i][j][k] = float(line.split(',')[10])
            # sos[i][j][k]     = float(line.split( ',' )[11])
        # Update line counter
        line_counter += 1

    # Close output file
    data_file_in.close()

    return (time, time_iter)


### Update boundaries
# @njit
def update_boundaries(sos, rho, rhou, rhov, rhow, rhoE, u, v, w, P, T, grid):
    # General form: w_g*phi_g + w_in*phi_in = phi_b
    # phi_g is ghost cell value
    # phi_in is inner cell value
    # phi_b is boundary value/flux
    # w_g is ghost cell weight
    # w_in is inner cell weight

    # Boundary conditions:
    # West:  Subsonic inlet
    # East:  Subsonic outlet // Supersonic outlet
    # South: Dirichlet & Neumann
    # North: Dirichlet & Neumann
    # Back:  Periodic
    # Front: Periodic

    # Outlet subsonic-supersonic switch mode
    # Option 1: Switch between outlet type once the min(u_field/sos_field) >= 1
    # Option 2: Individual switching for each grid point (not very fan of this one)
    # Option 3: Average boundary Mach number avg(u_field/sos_field) >= 1

    # Source code changes:
    # 1) Added subsonic 
    # 2) Added subsonic outflow
    # 3) Added supersonic outflow
    # 4) Changed South BC to zero gradient (Axis of symmetry)

    # West boundary points (SUBSONIC INFLOW)
    ##### CHARACTERISTIC BOUNDARY CONDITIONS DISABLED -- REPLACED WITH BASIC INFLOW
    i = 0
    for j in range(1, num_grid_y + 1):
        for k in range(1, num_grid_z + 1): # Original line (for k in range(1, num_grid_z + 1):)
            wg_g = 1.0 - (x_0 - grid[i][j][k][0]) / (grid[i + 1][j][k][0] - grid[i][j][k][0])
            wg_in = 1.0 - (grid[i + 1][j][k][0] - x_0) / (grid[i + 1][j][k][0] - grid[i][j][k][0])
            P_in = P[i + 1][j][k]
            T_in = T[i + 1][j][k]
            u_in = u[i + 1][j][k]
            v_in = v[i + 1][j][k]
            w_in = w[i + 1][j][k]

            rhou_in = rhou[i+1][j][k]
            rhov_in = rhov[i+1][j][k]
            rhow_in = rhow[i+1][j][k]
            # Ghost primitive variables
            rc = 0.5 * (grid[i][-1][k][1] + grid[i][-2][k][1])
            u_inflow = - (u_inlet/(rc)**2)*grid[i][j][k][1]**2 + u_inlet
            u_g = (u_inflow-wg_in*u_in)/wg_g
            v_g = (v_inlet-wg_in*v_in)/wg_g 
            w_g = (w_inlet-wg_in*w_in)/wg_g

            # Subsonic inlet
            Delta_g = grid[i + 1][j][k][0] - grid[i][j][k][0]
            Delta_in_in = grid[i + 2][j][k][0] - grid[i + 1][j][k][0]
            rho_in = rho[i + 1][j][k]
            sos_in = sos[i + 1][j][k]
            drho_dx_in_in = (rho[i + 2][j][k] - rho[i + 1][j][k]) / Delta_in_in
            du_dx_in_in = (u[i + 2][j][k] - u[i + 1][j][k]) / Delta_in_in
            dP_dx_in_in = (P[i + 2][j][k] - P[i + 1][j][k]) / Delta_in_in
            L_1_lambda_1_in_in = dP_dx_in_in - rho_in * sos_in * du_dx_in_in
            rho_g = rho[i + 1][j][k]
            
            ### Option 1: Fix Temperature -- recalculate pressure
            # T_g = ( T_ref - wg_in*T_in )/wg_g
            # P_g = thermodynamics.calculatePressureFromTemperatureDensity(T_g,rho_g)

            ### Option 2: Fix pressure -- recalculate temperature
            # P_g = (P_ref - wg_in * P_in) / wg_g
            # T_g = thermodynamics.calculateTemperatureFromPressureDensity(P_g, rho_g) 

            ### Option 3: Test
            # P_g = P_ref
            # T_g = T_ref
            # e_g = -1.0; rho_g = -1.0
            # rho_g, e_g = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_g,e_g,P_g,T_g)

            ### NSCBC subsonic inflow (LODI / outgoing-acoustic characteristic):
            # Impose the velocity profile (u_g,v_g,w_g above) and the static temperature
            # T_ref. The pressure is set by the OUTGOING (u-c) acoustic characteristic
            # carried out of the domain from the interior (the invariant P - rho*c*u is
            # continuous across the inlet face), and the density follows from (P_g, T_ref).
            # Pinning T keeps the inlet internal energy bounded -> no thermal runaway,
            # while the outgoing-characteristic pressure is non-reflecting.
            T_g = T_ref
            P_g = P_in + rho_in * sos_in * (u_g - u_in)
            rho_g, _e_tmp = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(-1.0, -1.0, P_g, T_g)

            #### CHARACTERISTIC BOUNDARY CONDITION
            # rel_error = 1.0
            
            # for ite in range(1, max_subr_iter + 1):
            #     if rel_error >= rel_tol:
            #         drho_dx_g = (rho_in - rho_g) / Delta_g
            #         du_dx_g = (u_in - u_g) / Delta_g
            #         dP_dx_g = (P_in - P_g) / Delta_g
            #         L_2_lambda_2_g = sos_in ** 2 * drho_dx_g - dP_dx_g
            #         L_5_lambda_5_g = dP_dx_g + rho_in * sos_in * du_dx_g
            #         dQ_1_dx_in = 1.0 / (sos_in ** 2) * (L_2_lambda_2_g + 0.5 * (L_5_lambda_5_g + L_1_lambda_1_in_in))
            #         rho_g_old = rho_g
            #         rho_g = rho_in - Delta_g * dQ_1_dx_in

            #         # P_g = thermodynamics.calculatePressureFromTemperatureDensity(T_g,rho_g) # Uncomment to recalculate pressure
            #         T_g = thermodynamics.calculateTemperatureFromPressureDensity(P_g, rho_g)  # Uncomment to recalculate temperature

            #         rel_error = abs((rho_g - rho_g_old) / rho_g_old)
            #         # print(rho_g, rho_g_old)

            #     else:
            #         break
            
            # #### Kill solver execution when subroutine does not converge -- prints output output data
            # if rel_error > rel_tol:
            #     print("Solver stopped due to inlet subroutine not converging -- i = " + str(i) + ", j = " + str(j) + ", k = " + str(k))
            #     in_par = [wg_in, wg_g, P_in, P[i+1][j][k], T_in, u_in, u[i+1][j][k], v_in, v[i+1][j][k], w_in, w[i+1][j][k], Delta_in_in, Delta_g, rho_in, rho[i+1][j][k], sos_in]
            #     print(in_par)
            #     file_name = "inlet_subroutine_stop" + '.csv'
            #     data_file_out = open(file_name, 'wt')
            #     output_string = str(wg_in) + ',' + str(wg_in) + ',' + str(P_in) + ',' + str(P[i+1][j][k]) + ',' + str(T_in) + ',' + str(u_in) + ',' + str(u[i+1][j][k]) + ',' + str(v_in) + ',' + str(v[i+1][j][k]) + ',' + str(w_in) + ',' + str(w[i+1][j][k]) + ',' + str(Delta_in_in) + ',' + str(Delta_g) + ',' + str(rho_in) + ',' + str(rho[i+1][j][k]) + ',' + str(sos_in) + ',' + str(i) + ',' + str(j) + ',' + str(k)
            #     data_file_out.write(output_string)
            #     data_file_out.close()

            #     data_output(time, time_iter, rho_field, u_field, v_field, w_field, E_field, s_field, P_field, T_field,
            #         sos_field, computational_plane)

            #     exit()

            # Specific internal energy and density
            e_g = thermodynamics.calculateInternalEnergyFromPressureTemperatureDensity(P_g, T_g, rho_g)
            ke_g = 0.5 * (u_g ** 2.0 + v_g ** 2.0 + w_g ** 2.0)  # Specific kinetic energy
            E_g = e_g + ke_g  # Specific total energy
            rho[i][j][k] = rho_g
            rhou[i][j][k] = rho_g * u_g
            rhov[i][j][k] = rho_g * v_g
            rhow[i][j][k] = rho_g * w_g
            rhoE[i][j][k] = rho_g * E_g

    # East boundary points (SUBSONIC OUTFLOW // SUPERSONIC OUTFLOW)
    i = num_grid_x + 1

    # Obtain velocity module at last inner axis point
    u_axis = 0.5 * (u[-2, 0, 1] + u[-2, 1, 1])
    v_axis = 0.5 * (v[-2, 0, 1] + v[-2, 1, 1])
    w_axis = 0.5 * (w[-2, 0, 1] + w[-2, 1, 1])
    sos_axis = 0.5 * (sos[-2, 0, 1] + sos[-2, 1, 1])
    V_axis = np.sqrt(u_axis ** 2 + v_axis ** 2 + w_axis ** 2)

    # Ma_in = V_axis / sos_axis # Uncomment to activate outflow boundary switch
    # Ma_in = 1.0             # Uncomment to activate supersonic outflow only
    Ma_in = 0.1             # Uncomment to activate subsonic outflow only (back pressure P_exit)

    ##### SUBSONIC OUTFLOW
    if Ma_in < 0.99: 
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                # Subsonic outflow
                P_in = P[i - 1][j][k]
                T_in = T[i - 1][j][k]
                u_in = rhou[i - 1][j][k] / rho[i - 1][j][k]
                v_in = rhov[i - 1][j][k] / rho[i - 1][j][k]
                w_in = rhow[i - 1][j][k] / rho[i - 1][j][k]

                # Subsonic outlet
                Delta_g = grid[i - 1][j][k][0] - grid[i][j][k][0]
                Delta_in = grid[i - 2][j][k][0] - grid[i - 1][j][k][0]
                rho_in = rho[i - 1][j][k]
                sos_in = sos[i - 1][j][k]
                Ma_in = u_in / sos_in
                K_in = 0.25 * sos_in * (1.0 - Ma_in ** 2) / L_x
                drho_dx_in = (rho[i - 2][j][k] - rho[i - 1][j][k]) / Delta_in
                du_dx_in = (rhou[i - 2][j][k] / rho[i - 2][j][k] - rhou[i - 1][j][k] / rho[i - 1][j][k]) / Delta_in
                dv_dx_in = (rhov[i - 2][j][k] / rho[i - 2][j][k] - rhov[i - 1][j][k] / rho[i - 1][j][k]) / Delta_in
                dw_dx_in = (rhow[i - 2][j][k] / rho[i - 2][j][k] - rhow[i - 1][j][k] / rho[i - 1][j][k]) / Delta_in
                dP_dx_in = (P[i - 2][j][k] - P[i - 1][j][k]) / Delta_in
                lambda_1_in = u_in - sos_in
                L_1_lambda_1_in = K_in * (P_in - P_exit)/ lambda_1_in
                L_2_lambda_2_in = sos_in ** 2 * drho_dx_in - dP_dx_in
                L_3_lambda_3_in = dv_dx_in
                L_4_lambda_4_in = dw_dx_in
                L_5_lambda_5_in = dP_dx_in + rho_in * sos_in * du_dx_in
                dQ_1_dx_in = 1.0 / (sos_in ** 2) * (L_2_lambda_2_in + 0.5 * (L_5_lambda_5_in + L_1_lambda_1_in))
                dQ_2_dx_in = 1.0 / (2 * rho_in * sos_in) * (L_5_lambda_5_in - L_1_lambda_1_in)
                dQ_3_dx_in = L_3_lambda_3_in
                dQ_4_dx_in = L_4_lambda_4_in
                dQ_5_dx_in = 0.5 * (L_5_lambda_5_in + L_1_lambda_1_in)
                rho_g = rho_in - Delta_g * dQ_1_dx_in
                u_g = u_in - Delta_g * dQ_2_dx_in
                v_g = v_in - Delta_g * dQ_3_dx_in
                w_g = w_in - Delta_g * dQ_4_dx_in
                P_g = P_in - Delta_g * dQ_5_dx_in
                T_g = T_in
                T_g = thermodynamics.calculateTemperatureFromPressureDensityWithInitialGuess(T_g, P_g, rho_g)

                # Specific internal energy and density
                rho_g = -1.0
                e_g = -1.0
                rho_g, e_g = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_g, e_g, P_g, T_g)
                ke_g = 0.5 * (u_g ** 2.0 + v_g ** 2.0 + w_g ** 2.0)  # Specific kinetic energy
                E_g = e_g + ke_g  # Specific total energy
                rho[i][j][k] = rho_g
                rhou[i][j][k] = rho_g * u_g
                rhov[i][j][k] = rho_g * v_g
                rhow[i][j][k] = rho_g * w_g
                rhoE[i][j][k] = rho_g * E_g

    ##### SUPERSONIC OUTFLOW
    else: 
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                P_in = P[i - 1][j][k]
                T_in = T[i - 1][j][k]
                u_in = rhou[i - 1][j][k] / rho[i - 1][j][k]
                v_in = rhov[i - 1][j][k] / rho[i - 1][j][k]
                w_in = rhow[i - 1][j][k] / rho[i - 1][j][k]

                # Supersonic outflow
                u_g = u_in
                v_g = v_in
                w_g = w_in
                P_g = P_in
                T_g = T_in

                # Specific internal energy and density
                rho_g = -1.0
                e_g = -1.0
                rho_g, e_g = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_g, e_g,
                                                                                                  P_g, T_g)
                ke_g = 0.5 * (u_g ** 2.0 + v_g ** 2.0 + w_g ** 2.0)  # Specific kinetic energy
                E_g = e_g + ke_g  # Specific total energy
                rho[i][j][k] = rho_g
                rhou[i][j][k] = rho_g * u_g
                rhov[i][j][k] = rho_g * v_g
                rhow[i][j][k] = rho_g * w_g
                rhoE[i][j][k] = rho_g * E_g

    # South boundary points (Axis of symmetry -> Gradient zero)
    j = 0
    for i in range(1, num_grid_x + 1):  # Original line: (for i in range(1, num_grid_x + 1):)
        for k in range(1, num_grid_z + 1):  # Original line: (for k in range(1, num_grid_z + 1):)
            wg_g = 1.0 - (y_0 - grid[i][j][k][1]) / (grid[i][j + 1][k][1] - grid[i][j][k][1])
            wg_in = 1.0 - (grid[i][j + 1][k][1] - y_0) / (grid[i][j + 1][k][1] - grid[i][j][k][1])
            P_in = P[i][j + 1][k]
            T_in = T[i][j + 1][k]
            u_in = rhou[i][j + 1][k] / rho[i][j + 1][k]
            v_in = rhov[i][j + 1][k] / rho[i][j + 1][k]
            w_in = rhow[i][j + 1][k] / rho[i][j + 1][k]
            P_g = P_in  # Neumann
            T_g = T_in  # Neumann
            u_g = u_in  # Neumann
            v_g = (0.0 - wg_in*v_in)/wg_g  # Neumann
            w_g = w_in  # Neumann

            # Specific internal energy and density
            rho_g = -1.0
            e_g = -1.0
            rho_g, e_g = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_g, e_g, P_g, T_g)
            ke_g = 0.5 * (u_g ** 2.0 + v_g ** 2.0 + w_g ** 2.0)  # Specific kinetic energy
            E_g = e_g + ke_g  # Specific total energy
            rho[i][j][k] = rho_g
            rhou[i][j][k] = rho_g * u_g
            rhov[i][j][k] = rho_g * v_g
            rhow[i][j][k] = rho_g * w_g
            rhoE[i][j][k] = rho_g * E_g

    # North boundary points
    j = num_grid_y + 1   # Ghost cell index
    for i in range(1, num_grid_x + 1): # Original line: (for i in range(1, num_grid_x + 1):)
        for k in range(1, num_grid_z + 1): # Original line: (for k in range(1, num_grid_z + 1):)
            eta_i = 0.5 * (grid[i][-1][k][1] + grid[i][-2][k][1])
            wg_g = 1.0 - (grid[i][j][k][1] - (y_0 + eta_i)) / (grid[i][j][k][1] - grid[i][j - 1][k][1])
            wg_in = 1.0 - ((y_0 + eta_i) - grid[i][j - 1][k][1]) / (grid[i][j][k][1] - grid[i][j - 1][k][1])
            delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0])
            delta_y = (grid[i][j][k][1] - grid[i][j-1][k][1])
            P_in = P[i][j - 1][k]   # Pressure in the interior cell
            T_in = T[i][j - 1][k]
            u_in = rhou[i][j - 1][k] / rho[i][j - 1][k]
            v_in = rhov[i][j - 1][k] / rho[i][j - 1][k]
            w_in = rhow[i][j - 1][k] / rho[i][j - 1][k]

            ##### SLIP-FREE CONDITION #####
            n_x = -drc_dx[i] / np.sqrt(drc_dx[i]**2+1)
            n_y = 1.0 / np.sqrt(drc_dx[i]**2+1)

            # u_g = ((u_in - u_in * n_x) - wg_in * u_in) / wg_g
            # v_g = ((v_in - v_in * n_y) - wg_in * u_in) / wg_g
            # w_g = (0.0 - wg_in * w_in) / wg_g

            ##### NO-SLIP CONDITION #####
            u_g = (0.0 - wg_in * u_in) / wg_g  # Dirichlet
            v_g = (0.0 - wg_in * v_in) / wg_g # Dirichlet
            w_g = (0.0 - wg_in * w_in) / wg_g  # Dirichlet

            # P_g = P_in  # Neumann - Zero gradient not normal to the surface
            # T_g = T_in  # Neumann - Zero gradient not notmal to the surface

            P_g = P[i][j-1][k] + (P[i+1][j-1][k]-P[i-1][j-1][k])/(2*delta_x)*(delta_y*L_y[i])/(grid[i][j][k][1]*drc_dx[i]-n_y/(n_x+epsilon))
            T_g = T[i][j-1][k] + (T[i+1][j-1][k]-T[i-1][j-1][k])/(2*delta_x)*(delta_y*L_y[i])/(grid[i][j][k][1]*drc_dx[i]-n_y/(n_x+epsilon))

            # Specific internal energy and density
            rho_g = -1.0
            e_g = -1.0
            rho_g, e_g = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_g, e_g, P_g, T_g)
            ke_g = 0.5 * (u_g ** 2.0 + v_g ** 2.0 + w_g ** 2.0)  # Specific kinetic energy
            E_g = e_g + ke_g  # Specific total energy
            rho[i][j][k] = rho_g
            rhou[i][j][k] = rho_g * u_g
            rhov[i][j][k] = rho_g * v_g
            rhow[i][j][k] = rho_g * w_g
            rhoE[i][j][k] = rho_g * E_g

    # Back boundary points
    k = 0
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            rho[i][j][k] = rho[i][j][num_grid_z]
            rhou[i][j][k] = rhou[i][j][num_grid_z]
            rhov[i][j][k] = rhov[i][j][num_grid_z]
            rhow[i][j][k] = rhow[i][j][num_grid_z]
            rhoE[i][j][k] = rhoE[i][j][num_grid_z]

    # Front boundary points
    k = num_grid_z + 1
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            rho[i][j][k] = rho[i][j][1]
            rhou[i][j][k] = rhou[i][j][1]
            rhov[i][j][k] = rhov[i][j][1]
            rhow[i][j][k] = rhow[i][j][1]
            rhoE[i][j][k] = rhoE[i][j][1]

    # Fill x-direction edge boundary points
    for i in range(1, num_grid_x + 1):
        j = 0
        k = 0
        rho[i][j][k] = 0.5 * (rho[i][j + 1][k] + rho[i][j][k + 1])
        rhou[i][j][k] = 0.5 * (rhou[i][j + 1][k] + rhou[i][j][k + 1])
        rhov[i][j][k] = 0.5 * (rhov[i][j + 1][k] + rhov[i][j][k + 1])
        rhow[i][j][k] = 0.5 * (rhow[i][j + 1][k] + rhow[i][j][k + 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i][j + 1][k] + rhoE[i][j][k + 1])
        j = 0
        k = num_grid_z + 1
        rho[i][j][k] = 0.5 * (rho[i][j + 1][k] + rho[i][j][k - 1])
        rhou[i][j][k] = 0.5 * (rhou[i][j + 1][k] + rhou[i][j][k - 1])
        rhov[i][j][k] = 0.5 * (rhov[i][j + 1][k] + rhov[i][j][k - 1])
        rhow[i][j][k] = 0.5 * (rhow[i][j + 1][k] + rhow[i][j][k - 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i][j + 1][k] + rhoE[i][j][k - 1])
        j = num_grid_y + 1
        k = 0
        rho[i][j][k] = 0.5 * (rho[i][j - 1][k] + rho[i][j][k + 1])
        rhou[i][j][k] = 0.5 * (rhou[i][j - 1][k] + rhou[i][j][k + 1])
        rhov[i][j][k] = 0.5 * (rhov[i][j - 1][k] + rhov[i][j][k + 1])
        rhow[i][j][k] = 0.5 * (rhow[i][j - 1][k] + rhow[i][j][k + 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i][j - 1][k] + rhoE[i][j][k + 1])
        j = num_grid_y + 1
        k = num_grid_z + 1
        rho[i][j][k] = 0.5 * (rho[i][j - 1][k] + rho[i][j][k - 1])
        rhou[i][j][k] = 0.5 * (rhou[i][j - 1][k] + rhou[i][j][k - 1])
        rhov[i][j][k] = 0.5 * (rhov[i][j - 1][k] + rhov[i][j][k - 1])
        rhow[i][j][k] = 0.5 * (rhow[i][j - 1][k] + rhow[i][j][k - 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i][j - 1][k] + rhoE[i][j][k - 1])

    # Fill y-direction edge boundary points
    for j in range(1, num_grid_y + 1):
        i = 0
        k = 0
        rho[i][j][k] = 0.5 * (rho[i + 1][j][k] + rho[i][j][k + 1])
        rhou[i][j][k] = 0.5 * (rhou[i + 1][j][k] + rhou[i][j][k + 1])
        rhov[i][j][k] = 0.5 * (rhov[i + 1][j][k] + rhov[i][j][k + 1])
        rhow[i][j][k] = 0.5 * (rhow[i + 1][j][k] + rhow[i][j][k + 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i + 1][j][k] + rhoE[i][j][k + 1])
        i = 0
        k = num_grid_z + 1
        rho[i][j][k] = 0.5 * (rho[i + 1][j][k] + rho[i][j][k - 1])
        rhou[i][j][k] = 0.5 * (rhou[i + 1][j][k] + rhou[i][j][k - 1])
        rhov[i][j][k] = 0.5 * (rhov[i + 1][j][k] + rhov[i][j][k - 1])
        rhow[i][j][k] = 0.5 * (rhow[i + 1][j][k] + rhow[i][j][k - 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i + 1][j][k] + rhoE[i][j][k - 1])
        i = num_grid_x + 1
        k = 0
        rho[i][j][k] = 0.5 * (rho[i - 1][j][k] + rho[i][j][k + 1])
        rhou[i][j][k] = 0.5 * (rhou[i - 1][j][k] + rhou[i][j][k + 1])
        rhov[i][j][k] = 0.5 * (rhov[i - 1][j][k] + rhov[i][j][k + 1])
        rhow[i][j][k] = 0.5 * (rhow[i - 1][j][k] + rhow[i][j][k + 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i - 1][j][k] + rhoE[i][j][k + 1])
        i = num_grid_x + 1
        k = num_grid_z + 1
        rho[i][j][k] = 0.5 * (rho[i - 1][j][k] + rho[i][j][k - 1])
        rhou[i][j][k] = 0.5 * (rhou[i - 1][j][k] + rhou[i][j][k - 1])
        rhov[i][j][k] = 0.5 * (rhov[i - 1][j][k] + rhov[i][j][k - 1])
        rhow[i][j][k] = 0.5 * (rhow[i - 1][j][k] + rhow[i][j][k - 1])
        rhoE[i][j][k] = 0.5 * (rhoE[i - 1][j][k] + rhoE[i][j][k - 1])

    # Fill z-direction edge boundary points
    for k in range(1, num_grid_z + 1):
        i = 0
        j = 0
        rho[i][j][k] = 0.5 * (rho[i + 1][j][k] + rho[i][j + 1][k])
        rhou[i][j][k] = 0.5 * (rhou[i + 1][j][k] + rhou[i][j + 1][k])
        rhov[i][j][k] = 0.5 * (rhov[i + 1][j][k] + rhov[i][j + 1][k])
        rhow[i][j][k] = 0.5 * (rhow[i + 1][j][k] + rhow[i][j + 1][k])
        rhoE[i][j][k] = 0.5 * (rhoE[i + 1][j][k] + rhoE[i][j + 1][k])
        i = 0
        j = num_grid_y + 1
        rho[i][j][k] = 0.5 * (rho[i + 1][j][k] + rho[i][j - 1][k])
        rhou[i][j][k] = 0.5 * (rhou[i + 1][j][k] + rhou[i][j - 1][k])
        rhov[i][j][k] = 0.5 * (rhov[i + 1][j][k] + rhov[i][j - 1][k])
        rhow[i][j][k] = 0.5 * (rhow[i + 1][j][k] + rhow[i][j - 1][k])
        rhoE[i][j][k] = 0.5 * (rhoE[i + 1][j][k] + rhoE[i][j - 1][k])
        i = num_grid_x + 1
        j = 0
        rho[i][j][k] = 0.5 * (rho[i - 1][j][k] + rho[i][j + 1][k])
        rhou[i][j][k] = 0.5 * (rhou[i - 1][j][k] + rhou[i][j + 1][k])
        rhov[i][j][k] = 0.5 * (rhov[i - 1][j][k] + rhov[i][j + 1][k])
        rhow[i][j][k] = 0.5 * (rhow[i - 1][j][k] + rhow[i][j + 1][k])
        rhoE[i][j][k] = 0.5 * (rhoE[i - 1][j][k] + rhoE[i][j + 1][k])
        i = num_grid_x + 1
        j = num_grid_y + 1
        rho[i][j][k] = 0.5 * (rho[i - 1][j][k] + rho[i][j - 1][k])
        rhou[i][j][k] = 0.5 * (rhou[i - 1][j][k] + rhou[i][j - 1][k])
        rhov[i][j][k] = 0.5 * (rhov[i - 1][j][k] + rhov[i][j - 1][k])
        rhow[i][j][k] = 0.5 * (rhow[i - 1][j][k] + rhow[i][j - 1][k])
        rhoE[i][j][k] = 0.5 * (rhoE[i - 1][j][k] + rhoE[i][j - 1][k])

    # Fill corner boundary points
    i = 0
    j = 0
    k = 0
    rho[i][j][k] = (1.0 / 3.0) * (rho[i + 1][j][k] + rho[i][j + 1][k] + rho[i][j][k + 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i + 1][j][k] + rhou[i][j + 1][k] + rhou[i][j][k + 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i + 1][j][k] + rhov[i][j + 1][k] + rhov[i][j][k + 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i + 1][j][k] + rhow[i][j + 1][k] + rhow[i][j][k + 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i + 1][j][k] + rhoE[i][j + 1][k] + rhoE[i][j][k + 1])
    i = num_grid_x + 1
    j = 0
    k = 0
    rho[i][j][k] = (1.0 / 3.0) * (rho[i - 1][j][k] + rho[i][j + 1][k] + rho[i][j][k + 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i - 1][j][k] + rhou[i][j + 1][k] + rhou[i][j][k + 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i - 1][j][k] + rhov[i][j + 1][k] + rhov[i][j][k + 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i - 1][j][k] + rhow[i][j + 1][k] + rhow[i][j][k + 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i - 1][j][k] + rhoE[i][j + 1][k] + rhoE[i][j][k + 1])
    i = 0
    j = num_grid_y + 1
    k = 0
    rho[i][j][k] = (1.0 / 3.0) * (rho[i + 1][j][k] + rho[i][j - 1][k] + rho[i][j][k + 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i + 1][j][k] + rhou[i][j - 1][k] + rhou[i][j][k + 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i + 1][j][k] + rhov[i][j - 1][k] + rhov[i][j][k + 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i + 1][j][k] + rhow[i][j - 1][k] + rhow[i][j][k + 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i + 1][j][k] + rhoE[i][j - 1][k] + rhoE[i][j][k + 1])
    i = num_grid_x + 1
    j = num_grid_y + 1
    k = 0
    rho[i][j][k] = (1.0 / 3.0) * (rho[i - 1][j][k] + rho[i][j - 1][k] + rho[i][j][k + 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i - 1][j][k] + rhou[i][j - 1][k] + rhou[i][j][k + 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i - 1][j][k] + rhov[i][j - 1][k] + rhov[i][j][k + 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i - 1][j][k] + rhow[i][j - 1][k] + rhow[i][j][k + 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i - 1][j][k] + rhoE[i][j - 1][k] + rhoE[i][j][k + 1])
    i = 0
    j = 0
    k = num_grid_z + 1
    rho[i][j][k] = (1.0 / 3.0) * (rho[i + 1][j][k] + rho[i][j + 1][k] + rho[i][j][k - 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i + 1][j][k] + rhou[i][j + 1][k] + rhou[i][j][k - 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i + 1][j][k] + rhov[i][j + 1][k] + rhov[i][j][k - 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i + 1][j][k] + rhow[i][j + 1][k] + rhow[i][j][k - 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i + 1][j][k] + rhoE[i][j + 1][k] + rhoE[i][j][k - 1])
    i = num_grid_x + 1
    j = 0
    k = num_grid_z + 1
    rho[i][j][k] = (1.0 / 3.0) * (rho[i - 1][j][k] + rho[i][j + 1][k] + rho[i][j][k - 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i - 1][j][k] + rhou[i][j + 1][k] + rhou[i][j][k - 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i - 1][j][k] + rhov[i][j + 1][k] + rhov[i][j][k - 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i - 1][j][k] + rhow[i][j + 1][k] + rhow[i][j][k - 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i - 1][j][k] + rhoE[i][j + 1][k] + rhoE[i][j][k - 1])
    i = 0
    j = num_grid_y + 1
    k = num_grid_z + 1
    rho[i][j][k] = (1.0 / 3.0) * (rho[i + 1][j][k] + rho[i][j - 1][k] + rho[i][j][k - 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i + 1][j][k] + rhou[i][j - 1][k] + rhou[i][j][k - 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i + 1][j][k] + rhov[i][j - 1][k] + rhov[i][j][k - 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i + 1][j][k] + rhow[i][j - 1][k] + rhow[i][j][k - 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i + 1][j][k] + rhoE[i][j - 1][k] + rhoE[i][j][k - 1])
    i = num_grid_x + 1
    j = num_grid_y + 1
    k = num_grid_z + 1
    rho[i][j][k] = (1.0 / 3.0) * (rho[i - 1][j][k] + rho[i][j - 1][k] + rho[i][j][k - 1])
    rhou[i][j][k] = (1.0 / 3.0) * (rhou[i - 1][j][k] + rhou[i][j - 1][k] + rhou[i][j][k - 1])
    rhov[i][j][k] = (1.0 / 3.0) * (rhov[i - 1][j][k] + rhov[i][j - 1][k] + rhov[i][j][k - 1])
    rhow[i][j][k] = (1.0 / 3.0) * (rhow[i - 1][j][k] + rhow[i][j - 1][k] + rhow[i][j][k - 1])
    rhoE[i][j][k] = (1.0 / 3.0) * (rhoE[i - 1][j][k] + rhoE[i][j - 1][k] + rhoE[i][j][k - 1])

    # print( rho )
    # print( rhou )
    # print( rhov )
    # print( rhow )
    # print( rhoE )


### Calculate transport coefficients
# @njit
def calculate_transport_coefficients(mu, kappa, P, T, rho):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                mu[i][j][k] = transport_coefficients.calculateDynamicViscosity(P[i][j][k], T[i][j][k], rho[i][j][k])
                kappa[i][j][k] = transport_coefficients.calculateThermalConductivity(P[i][j][k], T[i][j][k],
                                                                                     rho[i][j][k])
                # print( mu )
    # print( kappa )


### Calculate source terms
@njit
def source_terms(f_rhou, f_rhov, f_rhow, f_rhoE, rho, u, v, w, mesh):
    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                f_rhou[i][j][k] = 0.0
                f_rhov[i][j][k] = 0.0
                f_rhow[i][j][k] = 0.0
                f_rhoE[i][j][k] = 0.0
    # print( f_rhou )
    # print( f_rhov )
    # print( f_rhow )
    # print( f_rhoE )


### Output data to file
def data_output(time, time_iter, rho, u, v, w, E, s, P, T, sos, grid):
    # Write 3-D data:
    # x, y, z, rho, u, v, w, E, s, P, T, sos

    # Open output file
    file_name = str(name_file_out) + '_' + str(time_iter) + '.csv'
    data_file_out = open(file_name, 'wt')

    # Header string
    header_string = '# Time [s], Iter[-]\n'
    data_file_out.write(header_string)
    header_string = str(time) + ',' + str(time_iter) + '\n'
    data_file_out.write(header_string)
    header_string = '# x [m], y [m], z [m], rho [kg/m3], u [m/s], v [m/s], w [m/s], E [J/kg], s [J/(kg·K)], P [Pa], T [K], sos [m/s]\n'
    data_file_out.write(header_string)

    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                output_string = str(grid[i][j][k][0]) + ',' + str(grid[i][j][k][1]) + ',' + str(grid[i][j][k][2])
                output_string += ','
                output_string += str(rho[i][j][k])
                output_string += ','
                output_string += str(u[i][j][k]) + ',' + str(v[i][j][k]) + ',' + str(w[i][j][k])
                output_string += ','
                output_string += str(E[i][j][k]) + ',' + str(s[i][j][k])
                output_string += ','
                output_string += str(P[i][j][k]) + ',' + str(T[i][j][k]) + ',' + str(sos[i][j][k])
                output_string += '\n'
                data_file_out.write(output_string)

    # Close output file
    data_file_out.close()


### Calculate time step
# @njit
def time_step(rho, u, v, w, P, T, sos, mu, kappa, grid):
    # Inviscid time step size for explicit schemes:
    # E. F. Toro.
    # Riemann solvers and numerical methods for fluid dynamics.
    # Springer, 2009.

    # Viscous time step size for explicit schemes:
    # E. Turkel, R.C. Swanson, V. N. Vatsa, J.A. White.
    # Multigrid for hypersonic viscous two- and three-dimensional flows.
    # NASA Contractor Report 187603, 1991.

    # Initialize to largest float value
    #     delta_t = float( 'inf' )
    delta_t = 1.0e6

    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                # Calculate specific heat capacities
                c_v = -1.0
                c_p = -1.0
                c_v, c_p = thermodynamics.calculateSpecificHeatCapacities(c_v, c_p, P[i][j][k], T[i][j][k],
                                                                          rho[i][j][k])
                ## Geometric stuff (delta_y is the PHYSICAL spacing = L_y * d(eta); grid stores eta)
                delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0])
                delta_y = L_y[i] * 0.5 * (grid[i][j + 1][k][1] - grid[i][j - 1][k][1])
                delta_z = 0.5 * (grid[i][j][k + 1][2] - grid[i][j][k - 1][2])
                ## x-direction inviscid, viscous and thermal terms
                S_x = abs(u[i][j][k]) + sos[i][j][k]
                delta_t = min(delta_t, (1.0 / S_x) * CFL * delta_x)
                delta_t = min(delta_t, (1.0 / (mu[i][j][k] + epsilon)) * CFL * rho[i][j][k] * (delta_x ** 2.0))
                delta_t = min(delta_t, (1.0 / (kappa[i][j][k] + epsilon)) * CFL * rho[i][j][k] * c_p * (delta_x ** 2.0))
                ## y-direction inviscid, viscous and thermal terms
                S_y = abs(v[i][j][k]) + sos[i][j][k]
                delta_t = min(delta_t, (1.0 / S_y) * CFL * delta_y)
                delta_t = min(delta_t, (1.0 / (mu[i][j][k] + epsilon)) * CFL * rho[i][j][k] * (delta_y ** 2.0))
                delta_t = min(delta_t, (1.0 / (kappa[i][j][k] + epsilon)) * CFL * rho[i][j][k] * c_p * (delta_y ** 2.0))
                ## z-direction inviscid, viscous and thermal terms
                S_z = abs(w[i][j][k]) + sos[i][j][k]
                delta_t = min(delta_t, (1.0 / S_z) * CFL * delta_z)
                delta_t = min(delta_t, (1.0 / (mu[i][j][k] + epsilon)) * CFL * rho[i][j][k] * (delta_z ** 2.0))
                delta_t = min(delta_t, (1.0 / (kappa[i][j][k] + epsilon)) * CFL * rho[i][j][k] * c_p * (delta_z ** 2.0))
    # print( delta_t )

    # Return minimum time step
    return (delta_t)


### Define centroids of spatial discretization
def spatial_discretization(grid):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                # Evenly-spaced (dummy variable) cell centroids
                eta_x = (i - 0.5) / num_grid_x
                eta_y = (j - 0.5) / num_grid_y
                eta_z = (k - 0.5) / num_grid_z

                # Unevenly-spaced (stretched) cell centroids
                grid[i][j][k][0] = x_0 + L_x * eta_x + A_x * (0.5 * L_x - L_x * eta_x) * (1.0 - eta_x) * eta_x

                # Calculate contour: top-wall radius of the four-zone nozzle at this x.
                # The transform stays eta = y/L_y(x); only the contour itself changed.
                x_val = grid[i][j][k][0]
                L_y_var = nz_cont.nozzle_top_contour(x_val, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
                # L_y_var = 1.0 # Uncomment to generate a straight pipe

                # Unevenly-spaced (stretched) cell centroids
                grid[i][j][k][0] = x_0 + L_x * eta_x + A_x * (0.5 * L_x - L_x * eta_x) * (1.0 - eta_x) * eta_x
                grid[i][j][k][1] = y_0 + L_y_var * eta_y + A_y * (0.5 * L_y_var - L_y_var * eta_y) * (1.0 - eta_y) * eta_y
                grid[i][j][k][2] = z_0 + L_z * eta_z + A_z * (0.5 * L_z - L_z * eta_z) * (1.0 - eta_z) * eta_z
                # Adjust (symmetric) boundary cell centroids
                if (grid[i][j][k][0] < x_0):
                    eta_x = (1.0 - 0.5) / num_grid_x
                    grid[i][j][k][0] = x_0 - (L_x * eta_x + A_x * (0.5 * L_x - L_x * eta_x) * (1.0 - eta_x) * eta_x)
                if (grid[i][j][k][0] > (x_0 + L_x)):
                    eta_x = (num_grid_x - 0.5) / num_grid_x
                    grid[i][j][k][0] = x_0 + 2.0 * L_x - (L_x * eta_x + A_x * (0.5 * L_x - L_x * eta_x) * (1.0 - eta_x) * eta_x)
                if (grid[i][j][k][1] < y_0):
                    eta_y = (1.0 - 0.5) / num_grid_y
                    grid[i][j][k][1] = y_0 - (L_y_var * eta_y + A_y * (0.5 * L_y_var - L_y_var * eta_y) * (1.0 - eta_y) * eta_y)
                if (grid[i][j][k][1] > (y_0 + L_y_var)):
                    eta_y = (num_grid_y - 0.5) / num_grid_y
                    grid[i][j][k][1] = y_0 + 2.0 * L_y_var - (L_y_var * eta_y + A_y * (0.5 * L_y_var - L_y_var * eta_y) * (1.0 - eta_y) * eta_y)
                if (grid[i][j][k][2] < z_0):
                    eta_z = (1.0 - 0.5) / num_grid_z
                    grid[i][j][k][2] = z_0 - (L_z * eta_z + A_z * (0.5 * L_z - L_z * eta_z) * (1.0 - eta_z) * eta_z)
                if (grid[i][j][k][2] > (z_0 + L_z)):
                    eta_z = (num_grid_z - 0.5) / num_grid_z
                    grid[i][j][k][2] = z_0 + 2.0 * L_z - (L_z * eta_z + A_z * (0.5 * L_z - L_z * eta_z) * (1.0 - eta_z) * eta_z)
    # print( grid )

### Define computational plane from physical plane
def generate_computationalDomain(physical_grid, computational_grid):
    for i in range(0, num_grid_x+2):
        for j in range(0,num_grid_y+2):
            for k in range(0,num_grid_z+2):
                computational_grid[i][j][k][0] = physical_grid[i][j][k][0]
                computational_grid[i][j][k][1] = physical_grid[i][j][k][1]/(0.5*(physical_grid[i][-1][k][1]+physical_grid[i][-2][k][1]))
                computational_grid[i][j][k][2] = physical_grid[i][j][k][2]

# def compute_normalVector(n, drc_dx):
#     for i in range(0, num_grid_x+2):
#         n[i][0] = drc_dx[i] / np.sqrt(drc_dx[i]**2+1)
#         n[i][1] = - 1/np.sqrt(drc_dx[i]**2+1)


### Contour axial derivative
# def drc_dz_segments(drc_dx, phys_mesh, R1, R2, Rexp, theta, alpha, x1, x2, xc, xexp):
#     for i in range(0, num_grid_x+2):
#         x = phys_mesh[i][1][1][0]

#         if x >= x1 and x < 0.0:
#             # print("x >= x1")
#             # print(x,R1)
#             # print((1-(-x/R1)**2))

#             drc_dx[i] = (x / R1) * (1.0 / np.sqrt(1.0-(x**2/R1**2)))
#         elif x < x1 and x > x2:
#             # print("x1 > x > x2")
#             # print(theta)
#             # print(math.asin(- (x - x2)/R2))

#             drc_dx[i] = -math.tan(theta)
#         elif x <= x2 and x >= xc :
#             # print("x <= x2")
#             # print(x, x2, R2)
#             # print(1 - (- (x - x2)/R2)**2)

#             drc_dx[i] = -(x-xc)/(R2*np.sqrt(1-((x-xc)/R2)**2))
#         elif x > 0.0 and x < xexp:
#             # print("0 < x < xexp")
#             # print(1 - (x / Rexp)**2)

#             drc_dx[i] = x / (Rexp * np.sqrt(1 - (x / Rexp)**2))
#         elif x >= xexp:
#             # print(" x > xexp")
#             # print(alpha)

#             drc_dx[i] = math.tan(alpha)
#         else:

#             drc_dx[i] = 0.0

#     return drc_dx

def drc_dz_segments(drc_dx, phys_mesh, num_grid_x):
    for i in range(0, num_grid_x + 2):
        # Extract the x-coordinate from the mesh
        x = phys_mesh[i][1][1][0]

        if x < -2.5:
            # Left straight pipe (flat)
            drc_dx[i] = 0.0

        elif -2.5 <= x <= 0.0:
            # Convergent section derivative using exponential expression
            # -4 / (e^x + e^-x)^2
            drc_dx[i] = -4.0 / ((math.e**x + math.e**-x) ** 2)

        elif 0.0 < x <= 2.5:
            # Divergent section derivative using exponential expression
            # 4 / (e^x + e^-x)^2
            drc_dx[i] = 4.0 / ((math.e**x + math.e**-x) ** 2)

        else:
            # Right straight pipe (flat)
            drc_dx[i] = 0.0

    return drc_dx
# def drc_dz_dz_segments(drc_dx_dx, phys_mesh, R1, R2, Rexp, theta, alpha, x1, x2, xc, xexp):
#     for i in range(0, num_grid_x+2):
#         x = phys_mesh[i][1][1][0]

#         if x >= x1 and x < 0.0:

#             drc_dx_dx[i] = 1.0 / (R1 * (1 - (x**2/R1**2))**(3/2))
#             print(x, R1)
#             # print("First convergent arc (drc_dx_dx = " + str(drc_dx_dx[i])+ ")" )
#         elif x < x1 and x > x2:

#             drc_dx_dx[i] = 0.0
#             # print("Convergent linear transition (drc_dx_dx = " + str(drc_dx_dx[i])+ ")" )
#         elif x <= x2 and x >= xc :

#             drc_dx_dx[i] = - 1.0 / (R2 * (1 - ((x-xc)**2/R2**2))**(3/2))
#             # print("Second convergent arc (drc_dx_dx = " + str(drc_dx_dx[i])+ ")" )
#         elif x > 0.0 and x < xexp:

#             drc_dx_dx[i] = 1.0 / (Rexp * (1 - (x**2/Rexp**2))**(3/2))
#             # print("Expansion arc (drc_dx_dx = " + str(drc_dx_dx[i])+ ")" )
#         elif x >= xexp:

#             drc_dx_dx[i] = 0.0
#             # print("Conical nozzle (drc_dx_dx = " + str(drc_dx_dx[i])+ ")" )
#         else:

#             drc_dx_dx[i] = 0.0
#             # print("Chamber (drc_dx_dx = " + str(drc_dx_dx[i])+ ")" )

#     return drc_dx_dx

def drc_dz_dz_segments(drc_dx_dx, phys_mesh, num_grid_x):
    for i in range(0, num_grid_x + 2):
        # Extract the x-coordinate from the mesh
        x = phys_mesh[i][1][1][0]

        if x < -2.5:
            # Left straight pipe (no curvature)
            drc_dx_dx[i] = 0.0

        elif -2.5 <= x <= 0.0:
            # Convergent section second derivative
            # 8 * (e^x - e^-x) / (e^x + e^-x)^3
            drc_dx_dx[i] = 8.0 * (math.e**x - math.e**-x) / ((math.e**x + math.e**-x) ** 3)

        elif 0.0 < x <= 2.5:
            # Divergent section second derivative
            # -8 * (e^x - e^-x) / (e^x + e^-x)^3
            drc_dx_dx[i] = -8.0 * (math.e**x - math.e**-x) / ((math.e**x + math.e**-x) ** 3)

        else:
            # Right straight pipe (no curvature)
            drc_dx_dx[i] = 0.0

    return drc_dx_dx

### Initialize thermodynamic variables
def initialize_thermodynamics(rho, E, s, u, v, w, P, T):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                rho_aux = -1.0
                e_aux = -1.0
                rho_aux, e_aux = thermodynamics.calculateDensityInternalEnergyFromPressureTemperature(rho_aux, e_aux, P[i][j][k], T[i][j][k])  # Calculate density and Internal Energy
                rho[i][j][k] = rho_aux
                e = e_aux
                ke = 0.5 * (u[i][j][k] ** 2.0 + v[i][j][k] ** 2.0 + w[i][j][k] ** 2.0)
                E[i][j][k] = e + ke
                s[i][j][k] = thermodynamics.calculateEntropyFromPressureTemperatureDensity(P[i][j][k], T[i][j][k], rho[i][j][k])
    # print( rho )
    # print( E )
    # print( s )


### Calculate speed of sound
# @njit
def calculate_speed_sound(sos, rho, P, P_thermo, T):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                if (artificial_compressibility_method):
                    sos[i][j][k] = (1.0 / (alpha_acm + epsilon)) * thermodynamics.calculateSoundSpeed(P_thermo, T[i][j][k], rho[i][j][k])
                else:
                    sos[i][j][k] = thermodynamics.calculateSoundSpeed(P[i][j][k], T[i][j][k], rho[i][j][k])  # Calculate a speed of sound
    # print( sos )


### Update conserved variables from primitive variables
@njit
def update_conserved(conserved, primitive, rho):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                conserved[i][j][k] = rho[i][j][k] * primitive[i][j][k]
    # print( conserved )


### Update field
@njit
def update_field(field_a, field_b):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                field_a[i][j][k] = field_b[i][j][k]
    # print( field_a )


### Calculate volume-averaged value of a field
@njit
def calculate_volume_averaged_value(field, grid):
    # Initialize quantities
    sum_volume = 0.0
    sum_volume_value = 0.0

    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                ## Geometric stuff
                delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0])
                delta_y = 0.5 * (grid[i][j + 1][k][1] - grid[i][j - 1][k][1])
                delta_z = 0.5 * (grid[i][j][k + 1][2] - grid[i][j][k - 1][2])
                volume = delta_x * delta_y * delta_z
                ## Update quantities
                sum_volume += volume
                sum_volume_value += volume * field[i][j][k]

    # Calculate volume-averaged value
    volume_averaged_value = sum_volume_value / sum_volume

    # Return volume-averaged value
    return (volume_averaged_value)


### Calculate alpha value of artificial compressibility method
@njit
def calculate_alpha_acm(P, P_thermo, grid):
    # Initialize value
    alpha = 1.0e6

    # Define pressure threshold
    P_threshold = 1.0e-5 * P_thermo

    # Internal points: L1-norm
    sum_num = 0.0
    sum_den = 0.0
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                ## Geometric stuff
                delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0])
                delta_y = 0.5 * (grid[i][j + 1][k][1] - grid[i][j - 1][k][1])
                delta_z = 0.5 * (grid[i][j][k + 1][2] - grid[i][j][k - 1][2])
                volume = delta_x * delta_y * delta_z
                ## Update values
                sum_num += volume * abs(P[i][j][k])
                sum_den += volume * (max(abs(P[i][j][k] - P_thermo), P_threshold))
    alpha = np.sqrt(1.0 + epsilon_acm * sum_num / sum_den)

    #    # Internal points: L2-norm
    #    sum_num = 0.0
    #    sum_den = 0.0
    #    for i in range( 1, num_grid_x + 1 ):
    #        for j in range( 1, num_grid_y + 1 ):
    #            for k in range( 1, num_grid_z + 1 ):
    #                ## Geometric stuff
    #                delta_x = 0.5*( grid[i+1][j][k][0] - grid[i-1][j][k][0] )
    #                delta_y = 0.5*( grid[i][j+1][k][1] - grid[i][j-1][k][1] )
    #                delta_z = 0.5*( grid[i][j][k+1][2] - grid[i][j][k-1][2] )
    #                volume  = delta_x*delta_y*delta_z
    #                ## Update values
    #                sum_num += ( volume*P[i][j][k] )**2.0
    #                sum_den += ( volume*( max( abs( P[i][j][k] - P_thermo ), P_threshold ) ) )**2.0
    #    alpha = np.sqrt( 1.0 + epsilon_acm*np.sqrt( sum_num )/np.sqrt( sum_den ) )
    #    # Internal points: infinity-norm
    #    for i in range( 1, num_grid_x + 1 ):
    #        for j in range( 1, num_grid_y + 1 ):
    #            for k in range( 1, num_grid_z + 1 ):
    #                ## Update value
    #                alpha_aux = np.sqrt( 1.0 + ( P[i][j][k]*epsilon_acm )/( max( abs( P[i][j][k] - P_thermo ), P_threshold ) ) )
    #                alpha     = min( alpha, alpha_aux )

    # Return value of alpha
    return (alpha)


### calculate wave speeds
@njit
def waves_speed(rho_L, rho_R, u_L, u_R, P_L, P_R, a_L, a_R):
    # Direct wave speed estimates:
    # B. Einfeldt.
    # On Godunov-type methods for gas dynamics.
    # SIAM Journal on Numerical Analysis, 25, 294-318, 1988.

    hat_u = (u_L * np.sqrt(rho_L) + u_R * np.sqrt(rho_R)) / (np.sqrt(rho_L) + np.sqrt(rho_R))
    hat_a = np.sqrt(
        ((a_L * a_L * np.sqrt(rho_L) + a_R * a_R * np.sqrt(rho_R)) / (np.sqrt(rho_L) + np.sqrt(rho_R))) + 0.5 * ((np.sqrt(rho_L) * np.sqrt(rho_R)) / 
                    ((np.sqrt(rho_L) + np.sqrt(rho_R)) * (np.sqrt(rho_L) + np.sqrt(rho_R)))) * (u_R - u_L) * (u_R - u_L))

    S_L = min(u_L - a_L, hat_u - hat_a)
    S_R = max(u_R + a_R, hat_u + hat_a)

    # Return wave speed estimates
    return (S_L, S_R)


### Calculate HLLC flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
@njit
def HLLC_flux(rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R, a_L, a_R, var_type):
    # HLLC approximate Riemann solver:
    # E. F. Toro.
    # Riemann solvers and numerical methods for fluid dynamics.
    # Springer, 2009.

    F_L = rho_L * u_L
    F_R = rho_R * u_R
    U_L = rho_L
    U_R = rho_R
    if (var_type == 0):
        F_L *= 1.0
        F_R *= 1.0
        U_L *= 1.0
        U_R *= 1.0
    elif (var_type == 1):
        F_L *= u_L
        F_L += P_L
        F_R *= u_R
        F_R += P_R
        U_L *= u_L
        U_R *= u_R
    elif (var_type == 2):
        F_L *= v_L
        F_R *= v_R
        U_L *= v_L
        U_R *= v_R
    elif (var_type == 3):
        F_L *= w_L
        F_R *= w_R
        U_L *= w_L
        U_R *= w_R
    elif (var_type == 4):
        F_L *= E_L
        F_L += u_L * P_L
        F_R *= E_R
        F_R += u_R * P_R
        U_L *= E_L
        U_R *= E_R

    S_L, S_R = waves_speed(rho_L, rho_R, u_L, u_R, P_L, P_R, a_L, a_R)
    S_star = (P_R - P_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
                rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    U_star_L = rho_L * ((S_L - u_L) / (S_L - S_star))
    U_star_R = rho_R * ((S_R - u_R) / (S_R - S_star))
    if (var_type == 0):
        U_star_L *= 1.0
        U_star_R *= 1.0
    elif (var_type == 1):
        U_star_L *= S_star
        U_star_R *= S_star
    elif (var_type == 2):
        U_star_L *= v_L
        U_star_R *= v_R
    elif (var_type == 3):
        U_star_L *= w_L
        U_star_R *= w_R
    elif (var_type == 4):
        U_star_L *= (E_L + (S_star - u_L) * (S_star + P_L / (rho_L * (S_L - u_L))))
        U_star_R *= (E_R + (S_star - u_R) * (S_star + P_R / (rho_R * (S_R - u_R))))
    F_star_L = F_L + S_L * (U_star_L - U_L)
    F_star_R = F_R + S_R * (U_star_R - U_R)
    F = 0.0
    if (0.0 <= S_L):
        F = F_L
    elif ((S_L <= 0.0) and (0.0 <= S_star)):
        F = F_star_L
    elif ((S_star <= 0.0) and (0.0 <= S_R)):
        F = F_star_R
    elif (0.0 >= S_R):
        F = F_R
    # print( F )

    # Return F value
    return (F)


### Calculate KGP flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
@njit
def KGP_flux(rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R, a_L, a_R, var_type):
    # Kennedy, Gruber & Pirozzoli (KGP) scheme:
    # G. Coppola , F. Capuano , S. Pirozzoli, L. de Luca.
    # Numerically stable formulations of convective terms for turbulent compressible flows.
    # Journal of Computational Physics, 382, 86-104, 2019.

    F = (1.0 / 8.0) * (rho_L + rho_R) * (u_L + u_R)
    if (var_type == 0):
        F *= 1.0 + 1.0
    elif (var_type == 1):
        F *= u_L + u_R
        F += (1.0 / 2.0) * (P_L + P_R)
    elif (var_type == 2):
        F *= v_L + v_R
    elif (var_type == 3):
        F *= w_L + w_R
    elif (var_type == 4):
        F *= E_L + P_L / rho_L + E_R + P_R / rho_R

    return (F)


### Calculate ECKEP flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
@njit
def ECKEP_flux(rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R, a_L, a_R, var_type):
    # Entropy conservative and kinetic energy preserving (ECKEP) scheme:
    # K. Bahuguna, R. Kolluru, S.V. R. Rao
    # Structure-preserving schemes conserving entropy and kinetic energy.
    # arXiv:2505.13374, 2025.

    F = (1.0 / 8.0) * (rho_L + rho_R) * (u_L + u_R)
    if (var_type == 0):
        F *= 1.0 + 1.0
    elif (var_type == 1):
        F *= u_L + u_R
        F += (1.0 / 2.0) * (P_L + P_R)
    elif (var_type == 2):
        F *= v_L + v_R
    elif (var_type == 3):
        F *= w_L + w_R
    elif (var_type == 4):
        bar_F_1 = (1.0 / 4.0) * (rho_L + rho_R) * (u_L + u_R)
        bar_F_2u = (1.0 / 2.0) * (bar_F_1 * (u_L + u_R) + (P_L + P_R))
        bar_F_2v = (1.0 / 2.0) * bar_F_1 * (v_L + v_R)
        bar_F_2w = (1.0 / 2.0) * bar_F_1 * (w_L + w_R)
        bar_F_3 = (1.0 / 2.0) * bar_F_1 * (E_L + P_L / rho_L + E_R + P_R / rho_R)
        ds_drho_L = (u_L * u_L + v_L * v_L + w_L * w_L - E_L - P_L / rho_L) / (rho_L * T_L)
        ds_drho_R = (u_R * u_R + v_R * v_R + w_R * w_R - E_R - P_R / rho_R) / (rho_R * T_R)
        ds_drhou_L = (-1.0) * u_L / (rho_L * T_L)
        ds_drhou_R = (-1.0) * u_R / (rho_R * T_R)
        ds_drhov_L = (-1.0) * v_L / (rho_L * T_L)
        ds_drhov_R = (-1.0) * v_R / (rho_R * T_R)
        ds_drhow_L = (-1.0) * w_L / (rho_L * T_L)
        ds_drhow_R = (-1.0) * w_R / (rho_R * T_R)
        ds_drhoE_L = 1.0 / (rho_L * T_L)
        ds_drhoE_R = 1.0 / (rho_R * T_R)
        V_1_L = (-1.0) * (s_L + rho_L * ds_drho_L)
        V_1_R = (-1.0) * (s_R + rho_R * ds_drho_R)
        V_2u_L = (-1.0) * rho_L * ds_drhou_L
        V_2u_R = (-1.0) * rho_R * ds_drhou_R
        V_2v_L = (-1.0) * rho_L * ds_drhov_L
        V_2v_R = (-1.0) * rho_R * ds_drhov_R
        V_2w_L = (-1.0) * rho_L * ds_drhow_L
        V_2w_R = (-1.0) * rho_R * ds_drhow_R
        V_3_L = (-1.0) * rho_L * ds_drhoE_L
        V_3_R = (-1.0) * rho_R * ds_drhoE_R
        deltaU_3 = rho_R*E_R - rho_L*E_L
        deltaV_1 = V_1_R - V_1_L
        deltaV_2u = V_2u_R - V_2u_L
        deltaV_2v = V_2v_R - V_2v_L
        deltaV_2w = V_2w_R - V_2w_L
        deltaV_3 = V_3_R - V_3_L
        psi_L = u_L * P_L / T_L
        psi_R = u_R * P_R / T_R
        deltaPsi = psi_R - psi_L
        alpha_3 = 2.0 * (
                    bar_F_1 * deltaV_1 + bar_F_2u * deltaV_2u + bar_F_2v * deltaV_2v + bar_F_2w * deltaV_2w + bar_F_3 * deltaV_3 - deltaPsi) / (
                              deltaV_3 * deltaU_3 + epsilon)
        F = bar_F_3 - (1.0 / 2.0) * alpha_3 * deltaU_3

    return (F)


### Calculate ECKEP_MOVERS_RH flux ... var_type corresponds to: 0 for rho, 1-3 for rhouvw, 4 for rhoE
@njit
def HES_flux(rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R, a_L, a_R,
             var_type):
    # Method of Optimal Viscosity for Enhanced Resolution of Shocks (MOVERS) scheme:
    # S. Jaisankar, S.V. Raghurama Rao.
    # A central Rankine-Hugoniot solver for hyperbolic conservation laws.
    # Journal of Computational Physics, 228, 770-798, 2009.

    # Calculate eigenvalues
    u = (1.0 / 2.0) * (u_L + u_R)
    a = (1.0 / 2.0) * (a_L + a_R)
    lambda_1 = abs(u - a)
    lambda_2 = abs(u)
    lambda_3 = abs(u + a)
    lambda_min = min(lambda_1, lambda_2, lambda_3)
    lambda_max = max(lambda_1, lambda_2, lambda_3)

    # Conserved variable increment
    deltaU_1 = rho_R - rho_L
    deltaU_2u = rho_R * u_R - rho_L * u_L
    deltaU_2v = rho_R * v_R - rho_L * v_L
    deltaU_2w = rho_R * w_R - rho_L * w_L
    deltaU_3 = rho_R * E_R - rho_L * E_L

    # Flux increment
    deltaF_1 = rho_R * u_R - rho_L * u_L
    deltaF_2u = rho_R * u_R * u_R + P_R - rho_L * u_L * u_L - P_L
    deltaF_3 = rho_R * u_R * E_R + P_R * u_R - rho_L * u_L * E_L - P_L * u_L

    # Wave speed (limited)
    S_1 = deltaF_1 / (deltaU_1 + epsilon)
    if abs(S_1) > lambda_max:
        S_1 = math.copysign(lambda_max, S_1)
    if abs(S_1) < lambda_min:
        S_1 = math.copysign(lambda_min, S_1)
    S_2 = deltaF_2u / (deltaU_2u + epsilon)
    if abs(S_2) > lambda_max:
        S_2 = math.copysign(lambda_max, S_2)
    if abs(S_2) < lambda_min:
        S_2 = math.copysign(lambda_min, S_2)
    S_3 = deltaF_3 / (deltaU_3 + epsilon)
    if abs(S_3) > lambda_max:
        S_3 = math.copysign(lambda_max, S_3)
    if abs(S_3) < lambda_min:
        S_3 = math.copysign(lambda_min, S_3)
    alpha_S = min(abs(S_1), abs(S_2), abs(S_3))

    ### ---------------------------------###
    ### START: SHOCK SENSOR MODIFICATION ###
    ### ---------------------------------###

    # Pressure sensor: shocks have sharp pressure jump
    P = (1.0 / 2.0) * (P_L + P_R)
    sensor_P = min(1.0, 50.0 * abs(P_R - P_L) / (P + epsilon))  # Factor: 10, 50, 100

    ## Compression switch: compressions (shocks) will be negative; expansions (rarefactions) will be positive
    # delta_u = u_R - u_L
    ## Apply sensor: if delta_u > 0 (rarefaction/expansion)
    # if( delta_u > 0.0 ):
    #    alpha_S = 0.0

    alpha_S *= sensor_P

    ### -------------------------------###
    ### END: SHOCK SENSOR MODIFICATION ###
    ### -------------------------------###

    # Prevent sonic/entropy glitch
    if (alpha_S > epsilon):
        theta = 1.0  # !! theta needs to be larger than 0.0 !!
        alpha_S = (alpha_S * alpha_S + theta * theta) / (2.0 * theta)

    F = (1.0 / 8.0) * (rho_L + rho_R) * (u_L + u_R)
    if (var_type == 0):
        F *= 1.0 + 1.0
        F -= (1.0 / 2.0) * alpha_S * deltaU_1
    elif (var_type == 1):
        F *= u_L + u_R
        F += (1.0 / 2.0) * (P_L + P_R)
        F -= (1.0 / 2.0) * alpha_S * deltaU_2u
    elif (var_type == 2):
        F *= v_L + v_R
        F -= (1.0 / 2.0) * alpha_S * deltaU_2v
    elif (var_type == 3):
        F *= w_L + w_R
        F -= (1.0 / 2.0) * alpha_S * deltaU_2w
    elif (var_type == 4):
        bar_F_1 = (1.0 / 4.0) * (rho_L + rho_R) * (u_L + u_R)
        bar_F_2u = (1.0 / 2.0) * (bar_F_1 * (u_L + u_R) + (P_L + P_R))
        bar_F_2v = (1.0 / 2.0) * bar_F_1 * (v_L + v_R)
        bar_F_2w = (1.0 / 2.0) * bar_F_1 * (w_L + w_R)
        bar_F_3 = (1.0 / 2.0) * bar_F_1 * (E_L + P_L / rho_L + E_R + P_R / rho_R)
        ds_drho_L = (u_L * u_L + v_L * v_L + w_L * w_L - E_L - P_L / rho_L) / (rho_L * T_L)
        ds_drho_R = (u_R * u_R + v_R * v_R + w_R * w_R - E_R - P_R / rho_R) / (rho_R * T_R)
        ds_drhou_L = (-1.0) * u_L / (rho_L * T_L)
        ds_drhou_R = (-1.0) * u_R / (rho_R * T_R)
        ds_drhov_L = (-1.0) * v_L / (rho_L * T_L)
        ds_drhov_R = (-1.0) * v_R / (rho_R * T_R)
        ds_drhow_L = (-1.0) * w_L / (rho_L * T_L)
        ds_drhow_R = (-1.0) * w_R / (rho_R * T_R)
        ds_drhoE_L = 1.0 / (rho_L * T_L)
        ds_drhoE_R = 1.0 / (rho_R * T_R)
        V_1_L = (-1.0) * (s_L + rho_L * ds_drho_L)
        V_1_R = (-1.0) * (s_R + rho_R * ds_drho_R)
        V_2u_L = (-1.0) * rho_L * ds_drhou_L
        V_2u_R = (-1.0) * rho_R * ds_drhou_R
        V_2v_L = (-1.0) * rho_L * ds_drhov_L
        V_2v_R = (-1.0) * rho_R * ds_drhov_R
        V_2w_L = (-1.0) * rho_L * ds_drhow_L
        V_2w_R = (-1.0) * rho_R * ds_drhow_R
        V_3_L = (-1.0) * rho_L * ds_drhoE_L
        V_3_R = (-1.0) * rho_R * ds_drhoE_R
        deltaV_1 = V_1_R - V_1_L
        deltaV_2u = V_2u_R - V_2u_L
        deltaV_2v = V_2v_R - V_2v_L
        deltaV_2w = V_2w_R - V_2w_L
        deltaV_3 = V_3_R - V_3_L
        psi_L = u_L * P_L / T_L
        psi_R = u_R * P_R / T_R
        deltaPsi = psi_R - psi_L
        alpha_3 = 2.0 * (
                    bar_F_1 * deltaV_1 + bar_F_2u * deltaV_2u + bar_F_2v * deltaV_2v + bar_F_2w * deltaV_2w + bar_F_3 * deltaV_3 - deltaPsi) / (
                              deltaV_3 * deltaU_3 + epsilon)
        F = bar_F_3 - (1.0 / 2.0) * alpha_3 * deltaU_3
        F -= (1.0 / 2.0) * alpha_S * deltaU_3

    return (F)


### Calculate inviscid fluxes
@njit
def inviscid_fluxes(rho_inv, rhou_inv, rhov_inv, rhow_inv, rhoE_inv, rho, u, v, w, E, s, P, P_thermo, T, sos, grid):
    # Unsplit method for Euler equations:
    # E. F. Toro.
    # Riemann solvers and numerical methods for fluid dynamics.
    # Springer, 2009.

    # (MODIFIED) HLLC flux terms were renamed so multiple directions could be considered at the same time

    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                ## Geometric stuff
                delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0]) # epsilon
                delta_y = 0.5 * (grid[i][j + 1][k][1] - grid[i][j - 1][k][1]) # nu
                delta_z = 0.5 * (grid[i][j][k + 1][2] - grid[i][j][k - 1][2]) # z
                dxi_dx = 1.0
                dxi_dy = 0.0
                dxi_dz = 0.0
                deta_dx = - grid[i][j][k][1]/L_y[i] * drc_dx[i]
                deta_dy = 1.0 / L_y[i]
                deta_dz = 0.0
                dzeta_dx = 0.0
                dzeta_dy = 0.0
                dzeta_dz = 1.0
                module_grad_xi = np.sqrt(dxi_dx**2+dxi_dy**2+dxi_dz**2)
                module_grad_eta = np.sqrt(deta_dx**2+deta_dy**2+deta_dz**2)
                module_grad_zeta = np.sqrt(dzeta_dx**2+dzeta_dy**2+dzeta_dz**2)
                n_xi_x = dxi_dx / module_grad_xi
                n_xi_y = dxi_dy / module_grad_xi
                n_xi_z = dxi_dz / module_grad_xi
                n_eta_x = deta_dx / module_grad_eta
                n_eta_y = deta_dy / module_grad_eta
                n_eta_z = deta_dz / module_grad_eta
                n_zeta_x = dzeta_dx / module_grad_zeta
                n_zeta_y = dzeta_dy / module_grad_zeta
                n_zeta_z = dzeta_dz / module_grad_zeta
                ## x-direction i+1/2
                t1_xi_x = n_xi_y
                t1_xi_y = - n_xi_x
                t1_xi_z = 0.0
                t2_xi_x = n_xi_x*n_xi_z
                t2_xi_y = n_xi_y*n_xi_z
                t2_xi_z = - n_xi_x**2 - n_xi_y**2
                module_t1_xi = np.sqrt(t1_xi_x**2+t1_xi_y**2+t1_xi_z**2)
                module_t2_xi = np.sqrt(t2_xi_x**2+t2_xi_y**2+t2_xi_z**2)
                t1_xi_x /= module_t1_xi
                t1_xi_y /= module_t1_xi
                t1_xi_z /= module_t1_xi
                t2_xi_x /= module_t2_xi
                t2_xi_y /= module_t2_xi
                t2_xi_z /= module_t2_xi
                index_L = i
                index_R = i + 1
                rho_L = rho[index_L][j][k]
                rho_R = rho[index_R][j][k]
                u_L = u[index_L][j][k]
                u_R = u[index_R][j][k]
                v_L = v[index_L][j][k]
                v_R = v[index_R][j][k]
                w_L = w[index_L][j][k]
                w_R = w[index_R][j][k]
                E_L = E[index_L][j][k]
                E_R = E[index_R][j][k]
                s_L = s[index_L][j][k]
                s_R = s[index_R][j][k]
                P_L = P[index_L][j][k]
                P_R = P[index_R][j][k]
                T_L = T[index_L][j][k]
                T_R = T[index_R][j][k]
                a_L = sos[index_L][j][k]
                a_R = sos[index_R][j][k]
                P_rhouvw_L = P_L - P_thermo
                P_rhouvw_R = P_R - P_thermo  # P_Thermo = 0.0 when ACM is deactivated

                V_n_xi_L_p = u_L * n_xi_x + v_L * n_xi_y + w_L * n_xi_z
                V_n_xi_R_p = u_R * n_xi_x + v_R * n_xi_y + w_R * n_xi_z
                V_t1_xi_L_p = u_L * t1_xi_x + v_L * t1_xi_y + w_L * t1_xi_z
                V_t1_xi_R_p = u_R * t1_xi_x + v_R * t1_xi_y + w_R * t1_xi_z
                V_t2_xi_L_p = u_L * t2_xi_x + v_L * t2_xi_y + w_L * t2_xi_z
                V_t2_xi_R_p = u_R * t2_xi_x + v_R * t2_xi_y + w_R * t2_xi_z

                # rho
                var_type = 0
                rho_F_p_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_p, V_n_xi_R_p, V_t1_xi_L_p, V_t1_xi_R_p, V_t2_xi_L_p, V_t2_xi_R_p, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R,
                                     a_L, a_R, var_type)
                # rhou
                var_type = 1
                rho_n_F_p_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_p, V_n_xi_R_p, V_t1_xi_L_p, V_t1_xi_R_p, V_t2_xi_L_p, V_t2_xi_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhov
                var_type = 2
                rho_t1_F_p_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_p, V_n_xi_R_p, V_t1_xi_L_p, V_t1_xi_R_p, V_t2_xi_L_p, V_t2_xi_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhow
                var_type = 3
                rho_t2_F_p_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_p, V_n_xi_R_p, V_t1_xi_L_p, V_t1_xi_R_p, V_t2_xi_L_p, V_t2_xi_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhoE
                var_type = 4
                rhoE_F_p_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_p, V_n_xi_R_p, V_t1_xi_L_p, V_t1_xi_R_p, V_t2_xi_L_p, V_t2_xi_R_p, E_L, E_R, s_L, s_R, P_L, P_R, T_L,
                                      T_R, a_L, a_R, var_type)
                
                rhou_F_p_x = rho_n_F_p_x * n_xi_x + rho_t1_F_p_x * t1_xi_x + rho_t2_F_p_x * t2_xi_x
                rhov_F_p_x = rho_n_F_p_x * n_xi_y + rho_t1_F_p_x * t1_xi_y + rho_t2_F_p_x * t2_xi_y
                rhow_F_p_x = rho_n_F_p_x * n_xi_z + rho_t1_F_p_x * t1_xi_z + rho_t2_F_p_x * t2_xi_z

                ## x-direction i-1/2
                index_L = i - 1
                index_R = i
                rho_L = rho[index_L][j][k]
                rho_R = rho[index_R][j][k]
                u_L = u[index_L][j][k]
                u_R = u[index_R][j][k]
                v_L = v[index_L][j][k]
                v_R = v[index_R][j][k]
                w_L = w[index_L][j][k]
                w_R = w[index_R][j][k]
                E_L = E[index_L][j][k]
                E_R = E[index_R][j][k]
                s_L = s[index_L][j][k]
                s_R = s[index_R][j][k]
                P_L = P[index_L][j][k]
                P_R = P[index_R][j][k]
                T_L = T[index_L][j][k]
                T_R = T[index_R][j][k]
                a_L = sos[index_L][j][k]
                a_R = sos[index_R][j][k]
                P_rhouvw_L = P_L - P_thermo
                P_rhouvw_R = P_R - P_thermo  # P_Thermo = 0.0 when ACM is deactivated

                V_n_xi_L_m = u_L * n_xi_x + v_L * n_xi_y + w_L * n_xi_z
                V_n_xi_R_m = u_R * n_xi_x + v_R * n_xi_y + w_R * n_xi_z
                V_t1_xi_L_m = u_L * t1_xi_x + v_L * t1_xi_y + w_L * t1_xi_z
                V_t1_xi_R_m = u_R * t1_xi_x + v_R * t1_xi_y + w_R * t1_xi_z
                V_t2_xi_L_m = u_L * t2_xi_x + v_L * t2_xi_y + w_L * t2_xi_z
                V_t2_xi_R_m = u_R * t2_xi_x + v_R * t2_xi_y + w_R * t2_xi_z

                # rho
                var_type = 0
                rho_F_m_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_m, V_n_xi_R_m, V_t1_xi_L_m, V_t1_xi_R_m, V_t2_xi_L_m, V_t2_xi_R_m, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R,
                                     a_L, a_R, var_type)
                # rhou
                var_type = 1
                rho_n_F_m_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_m, V_n_xi_R_m, V_t1_xi_L_m, V_t1_xi_R_m, V_t2_xi_L_m, V_t2_xi_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhov
                var_type = 2
                rho_t1_F_m_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_m, V_n_xi_R_m, V_t1_xi_L_m, V_t1_xi_R_m, V_t2_xi_L_m, V_t2_xi_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhow
                var_type = 3
                rho_t2_F_m_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_m, V_n_xi_R_m, V_t1_xi_L_m, V_t1_xi_R_m, V_t2_xi_L_m, V_t2_xi_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhoE
                var_type = 4
                rhoE_F_m_x = HLLC_flux(rho_L, rho_R, V_n_xi_L_m, V_n_xi_R_m, V_t1_xi_L_m, V_t1_xi_R_m, V_t2_xi_L_m, V_t2_xi_R_m, E_L, E_R, s_L, s_R, P_L, P_R, T_L,
                                      T_R, a_L, a_R, var_type)
                

                rhou_F_m_x = rho_n_F_m_x * n_xi_x + rho_t1_F_m_x * t1_xi_x + rho_t2_F_m_x * t2_xi_x
                rhov_F_m_x = rho_n_F_m_x * n_xi_y + rho_t1_F_m_x * t1_xi_y + rho_t2_F_m_x * t2_xi_y
                rhow_F_m_x = rho_n_F_m_x * n_xi_z + rho_t1_F_m_x * t1_xi_z + rho_t2_F_m_x * t2_xi_z

                ## y-direction j+1/2  (face-consistent eta metric -> free-stream preserving)
                eta_face = 0.5 * (grid[i][j][k][1] + grid[i][j + 1][k][1])
                deta_dx = - eta_face / L_y[i] * drc_dx[i]
                deta_dy = 1.0 / L_y[i]
                module_grad_eta = np.sqrt(deta_dx**2 + deta_dy**2)
                mge_p = module_grad_eta
                n_eta_x = deta_dx / module_grad_eta
                n_eta_y = deta_dy / module_grad_eta
                n_eta_z = 0.0
                t1_eta_x = n_eta_y
                t1_eta_y = - n_eta_x
                t1_eta_z = 0.0
                t2_eta_x = n_eta_x * n_eta_z
                t2_eta_y = n_eta_y * n_eta_z
                t2_eta_z = - n_eta_x**2 - n_eta_y**2
                module_t1_eta = np.sqrt(t1_eta_x**2+t1_eta_y**2+t1_eta_z**2)
                module_t2_eta = np.sqrt(t2_eta_x**2+t2_eta_y**2+t2_eta_z**2)
                t1_eta_x /= module_t1_eta
                t1_eta_y /= module_t1_eta
                t1_eta_z /= module_t1_eta
                t2_eta_x /= module_t2_eta
                t2_eta_y /= module_t2_eta
                t2_eta_z /= module_t2_eta
                index_L = j
                index_R = j + 1
                rho_L = rho[i][index_L][k]
                rho_R = rho[i][index_R][k]
                u_L = u[i][index_L][k]
                u_R = u[i][index_R][k]
                v_L = v[i][index_L][k]
                v_R = v[i][index_R][k]
                w_L = w[i][index_L][k]
                w_R = w[i][index_R][k]
                E_L = E[i][index_L][k]
                E_R = E[i][index_R][k]
                s_L = s[i][index_L][k]
                s_R = s[i][index_R][k]
                P_L = P[i][index_L][k]
                P_R = P[i][index_R][k]
                T_L = T[i][index_L][k]
                T_R = T[i][index_R][k]
                a_L = sos[i][index_L][k]
                a_R = sos[i][index_R][k]
                P_rhouvw_L = P_L - P_thermo
                P_rhouvw_R = P_R - P_thermo  # P_Thermo = 0.0 when ACM is deactivated

                V_n_eta_L_p = u_L * n_eta_x + v_L * n_eta_y + w_L * n_eta_z
                V_n_eta_R_p = u_R * n_eta_x + v_R * n_eta_y + w_R * n_eta_z
                V_t1_eta_L_p = (u_L * t1_eta_x + v_L * t1_eta_y + w_L * t1_eta_z)
                V_t1_eta_R_p = (u_R * t1_eta_x + v_R * t1_eta_y + w_R * t1_eta_z)
                V_t2_eta_L_p = u_L * t2_eta_x + v_L * t2_eta_y + w_L * t2_eta_z
                V_t2_eta_R_p = u_R * t2_eta_x + v_R * t2_eta_y + w_R * t2_eta_z

                # In the y direction, the velocity input for v_L, v_R, ... have been changed as the normal velocity components are already obtained

                # rho
                var_type = 0
                rho_F_p_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_p, V_n_eta_R_p, V_t1_eta_L_p, V_t1_eta_R_p, V_t2_eta_L_p, V_t2_eta_R_p, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R,
                                     a_L, a_R, var_type)
                # rhou
                var_type = 1
                rho_n_F_p_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_p, V_n_eta_R_p, V_t1_eta_L_p, V_t1_eta_R_p, V_t2_eta_L_p, V_t2_eta_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhov
                var_type = 2
                rho_t1_F_p_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_p, V_n_eta_R_p, V_t1_eta_L_p, V_t1_eta_R_p, V_t2_eta_L_p, V_t2_eta_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhow
                var_type = 3
                rho_t2_F_p_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_p, V_n_eta_R_p, V_t1_eta_L_p, V_t1_eta_R_p, V_t2_eta_L_p, V_t2_eta_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhoE
                var_type = 4
                rhoE_F_p_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_p, V_n_eta_R_p, V_t1_eta_L_p, V_t1_eta_R_p, V_t2_eta_L_p, V_t2_eta_R_p, E_L, E_R, s_L, s_R, P_L, P_R, T_L,
                                      T_R, a_L, a_R, var_type)
                
                rhou_F_p_y = rho_n_F_p_y * n_eta_x + rho_t1_F_p_y * t1_eta_x + rho_t2_F_p_y * t2_eta_x
                rhov_F_p_y = rho_n_F_p_y * n_eta_y + rho_t1_F_p_y * t1_eta_y + rho_t2_F_p_y * t2_eta_y
                rhow_F_p_y = rho_n_F_p_y * n_eta_z + rho_t1_F_p_y * t1_eta_z + rho_t2_F_p_y * t2_eta_z

                ## y-direction j-1/2  (face-consistent eta metric -> free-stream preserving)
                eta_face = 0.5 * (grid[i][j - 1][k][1] + grid[i][j][k][1])
                deta_dx = - eta_face / L_y[i] * drc_dx[i]
                deta_dy = 1.0 / L_y[i]
                module_grad_eta = np.sqrt(deta_dx**2 + deta_dy**2)
                mge_m = module_grad_eta
                n_eta_x = deta_dx / module_grad_eta
                n_eta_y = deta_dy / module_grad_eta
                n_eta_z = 0.0
                t1_eta_x = n_eta_y
                t1_eta_y = - n_eta_x
                t1_eta_z = 0.0
                t2_eta_x = n_eta_x * n_eta_z
                t2_eta_y = n_eta_y * n_eta_z
                t2_eta_z = - n_eta_x**2 - n_eta_y**2
                module_t1_eta = np.sqrt(t1_eta_x**2+t1_eta_y**2+t1_eta_z**2)
                module_t2_eta = np.sqrt(t2_eta_x**2+t2_eta_y**2+t2_eta_z**2)
                t1_eta_x /= module_t1_eta
                t1_eta_y /= module_t1_eta
                t1_eta_z /= module_t1_eta
                t2_eta_x /= module_t2_eta
                t2_eta_y /= module_t2_eta
                t2_eta_z /= module_t2_eta
                index_L = j - 1
                index_R = j
                rho_L = rho[i][index_L][k]
                rho_R = rho[i][index_R][k]
                u_L = u[i][index_L][k]
                u_R = u[i][index_R][k]
                v_L = v[i][index_L][k]
                v_R = v[i][index_R][k]
                w_L = w[i][index_L][k]
                w_R = w[i][index_R][k]
                E_L = E[i][index_L][k]
                E_R = E[i][index_R][k]
                s_L = s[i][index_L][k]
                s_R = s[i][index_R][k]
                P_L = P[i][index_L][k]
                P_R = P[i][index_R][k]
                T_L = T[i][index_L][k]
                T_R = T[i][index_R][k]
                a_L = sos[i][index_L][k]
                a_R = sos[i][index_R][k]
                P_rhouvw_L = P_L - P_thermo
                P_rhouvw_R = P_R - P_thermo  # P_Thermo = 0.0 when ACM is deactivated

                V_n_eta_L_m = u_L * n_eta_x + v_L * n_eta_y + w_L * n_eta_z
                V_n_eta_R_m = u_R * n_eta_x + v_R * n_eta_y + w_R * n_eta_z
                V_t1_eta_L_m = (u_L * t1_eta_x + v_L * t1_eta_y + w_L * t1_eta_z)
                V_t1_eta_R_m = (u_R * t1_eta_x + v_R * t1_eta_y + w_R * t1_eta_z)
                V_t2_eta_L_m = u_L * t2_eta_x + v_L * t2_eta_y + w_L * t2_eta_z
                V_t2_eta_R_m = u_R * t2_eta_x + v_R * t2_eta_y + w_R * t2_eta_z

                # rho
                var_type = 0
                rho_F_m_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_m, V_n_eta_R_m, V_t1_eta_L_m, V_t1_eta_R_m, V_t2_eta_L_m, V_t2_eta_R_m, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R,
                                     a_L, a_R, var_type)
                # rhou
                var_type = 1
                rho_n_F_m_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_m, V_n_eta_R_m, V_t1_eta_L_m, V_t1_eta_R_m, V_t2_eta_L_m, V_t2_eta_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhov
                var_type = 2
                rho_t1_F_m_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_m, V_n_eta_R_m, V_t1_eta_L_m, V_t1_eta_R_m, V_t2_eta_L_m, V_t2_eta_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhow
                var_type = 3
                rho_t2_F_m_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_m, V_n_eta_R_m, V_t1_eta_L_m, V_t1_eta_R_m, V_t2_eta_L_m, V_t2_eta_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhoE
                var_type = 4
                rhoE_F_m_y = HLLC_flux(rho_L, rho_R, V_n_eta_L_m, V_n_eta_R_m, V_t1_eta_L_m, V_t1_eta_R_m, V_t2_eta_L_m, V_t2_eta_R_m, E_L, E_R, s_L, s_R, P_L, P_R, T_L,
                                      T_R, a_L, a_R, var_type)
                
                rhou_F_m_y = rho_n_F_m_y * n_eta_x + rho_t1_F_m_y * t1_eta_x + rho_t2_F_m_y * t2_eta_x
                rhov_F_m_y = rho_n_F_m_y * n_eta_y + rho_t1_F_m_y * t1_eta_y + rho_t2_F_m_y * t2_eta_y
                rhow_F_m_y = rho_n_F_m_y * n_eta_z + rho_t1_F_m_y * t1_eta_z + rho_t2_F_m_y * t2_eta_z

                ## z-direction k+1/2
                t1_zeta_x = 0.0
                t1_zeta_y = n_zeta_z
                t1_zeta_z = - n_zeta_y
                t2_zeta_x = - n_zeta_y**2  - n_zeta_z**2
                t2_zeta_y =  n_zeta_x*n_zeta_y
                t2_zeta_z = n_zeta_x*n_zeta_z
                module_t1_zeta = np.sqrt(t1_zeta_x**2+t1_zeta_y**2+t1_zeta_z**2)
                module_t2_zeta = np.sqrt(t2_zeta_x**2+t2_zeta_y**2+t2_zeta_z**2)
                t1_zeta_x /= module_t1_zeta
                t1_zeta_y /= module_t1_zeta
                t1_zeta_z /= module_t1_zeta
                t2_zeta_x /= module_t2_zeta
                t2_zeta_y /= module_t2_zeta
                t2_zeta_z /= module_t2_zeta
                index_L = k
                index_R = k + 1
                rho_L = rho[i][j][index_L]
                rho_R = rho[i][j][index_R]
                u_L = u[i][j][index_L]
                u_R = u[i][j][index_R]
                v_L = v[i][j][index_L]
                v_R = v[i][j][index_R]
                w_L = w[i][j][index_L]
                w_R = w[i][j][index_R]
                E_L = E[i][j][index_L]
                E_R = E[i][j][index_R]
                s_L = s[i][j][index_L]
                s_R = s[i][j][index_R]
                P_L = P[i][j][index_L]
                P_R = P[i][j][index_R]
                T_L = T[i][j][index_L]
                T_R = T[i][j][index_R]
                a_L = sos[i][j][index_L]
                a_R = sos[i][j][index_R]
                P_rhouvw_L = P_L - P_thermo
                P_rhouvw_R = P_R - P_thermo  # P_Thermo = 0.0 when ACM is deactivated

                V_n_zeta_L_p = u_L * n_zeta_x + v_L * n_zeta_y + w_L * n_zeta_z
                V_n_zeta_R_p = u_R * n_zeta_x + v_R * n_zeta_y + w_R * n_zeta_z
                V_t1_zeta_L_p = u_L * t1_zeta_x + v_L * t1_zeta_y + w_L * t1_zeta_z
                V_t1_zeta_R_p = u_R * t1_zeta_x + v_R * t1_zeta_y + w_R * t1_zeta_z
                V_t2_zeta_L_p = -(u_L * t2_zeta_x + v_L * t2_zeta_y + w_L * t2_zeta_z)   # Invert the sign of t2 to follow the direction of the flow
                V_t2_zeta_R_p = -(u_R * t2_zeta_x + v_R * t2_zeta_y + w_R * t2_zeta_z)   # Invert the sign of t2 to follow the direction of the flow

                # rho
                var_type = 0
                rho_F_p_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_p, V_n_zeta_R_p, V_t1_zeta_L_p, V_t1_zeta_R_p, V_t2_zeta_L_p, V_t2_zeta_R_p, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R,
                                     a_L, a_R, var_type)
                # rhou
                var_type = 1
                rho_n_F_p_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_p, V_n_zeta_R_p, V_t1_zeta_L_p, V_t1_zeta_R_p, V_t2_zeta_L_p, V_t2_zeta_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhov
                var_type = 2
                rho_t1_F_p_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_p, V_n_zeta_R_p, V_t1_zeta_L_p, V_t1_zeta_R_p, V_t2_zeta_L_p, V_t2_zeta_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhow
                var_type = 3
                rho_t2_F_p_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_p, V_n_zeta_R_p, V_t1_zeta_L_p, V_t1_zeta_R_p, V_t2_zeta_L_p, V_t2_zeta_R_p, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhoE
                var_type = 4
                rhoE_F_p_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_p, V_n_zeta_R_p, V_t1_zeta_L_p, V_t1_zeta_R_p, V_t2_zeta_L_p, V_t2_zeta_R_p, E_L, E_R, s_L, s_R, P_L, P_R, T_L,
                                      T_R, a_L, a_R, var_type)
                
                rhou_F_p_z = rho_n_F_p_z * n_zeta_x + rho_t1_F_p_z * t1_zeta_x + rho_t2_F_p_z * t2_zeta_x
                rhov_F_p_z = rho_n_F_p_z * n_zeta_y + rho_t1_F_p_z * t1_zeta_y + rho_t2_F_p_z * t2_zeta_y
                rhow_F_p_z = rho_n_F_p_z * n_zeta_z + rho_t1_F_p_z * t1_zeta_z + rho_t2_F_p_z * t2_zeta_z
                
                ## z-direction k-1/2
                index_L = k - 1
                index_R = k
                rho_L = rho[i][j][index_L]
                rho_R = rho[i][j][index_R]
                u_L = u[i][j][index_L]
                u_R = u[i][j][index_R]
                v_L = v[i][j][index_L]
                v_R = v[i][j][index_R]
                w_L = w[i][j][index_L]
                w_R = w[i][j][index_R]
                E_L = E[i][j][index_L]
                E_R = E[i][j][index_R]
                s_L = s[i][j][index_L]
                s_R = s[i][j][index_R]
                P_L = P[i][j][index_L]
                P_R = P[i][j][index_R]
                T_L = T[i][j][index_L]
                T_R = T[i][j][index_R]
                a_L = sos[i][j][index_L]
                a_R = sos[i][j][index_R]
                P_rhouvw_L = P_L - P_thermo
                P_rhouvw_R = P_R - P_thermo  # P_Thermo = 0.0 when ACM is deactivated

                V_n_zeta_L_m = u_L * n_zeta_x + v_L * n_zeta_y + w_L * n_zeta_z
                V_n_zeta_R_m = u_R * n_zeta_x + v_R * n_zeta_y + w_R * n_zeta_z
                V_t1_zeta_L_m = u_L * t1_zeta_x + v_L * t1_zeta_y + w_L * t1_zeta_z
                V_t1_zeta_R_m = u_R * t1_zeta_x + v_R * t1_zeta_y + w_R * t1_zeta_z
                V_t2_zeta_L_m = -(u_L * t2_zeta_x + v_L * t2_zeta_y + w_L * t2_zeta_z)   # Invert the sign of t2 to follow the direction of the flow
                V_t2_zeta_R_m = -(u_R * t2_zeta_x + v_R * t2_zeta_y + w_R * t2_zeta_z)   # Invert the sign of t2 to follow the direction of the flow

                # rho
                var_type = 0
                rho_F_m_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_m, V_n_zeta_R_m, V_t1_zeta_L_m, V_t1_zeta_R_m, V_t2_zeta_L_m, V_t2_zeta_R_m, E_L, E_R, s_L, s_R, P_L, P_R, T_L, T_R,
                                     a_L, a_R, var_type)
                # rhou
                var_type = 1
                rho_n_F_m_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_m, V_n_zeta_R_m, V_t1_zeta_L_m, V_t1_zeta_R_m, V_t2_zeta_L_m, V_t2_zeta_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhov
                var_type = 2
                rho_t1_F_m_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_m, V_n_zeta_R_m, V_t1_zeta_L_m, V_t1_zeta_R_m, V_t2_zeta_L_m, V_t2_zeta_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhow
                var_type = 3
                rho_t2_F_m_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_m, V_n_zeta_R_m, V_t1_zeta_L_m, V_t1_zeta_R_m, V_t2_zeta_L_m, V_t2_zeta_R_m, E_L, E_R, s_L, s_R, P_rhouvw_L,
                                      P_rhouvw_R, T_L, T_R, a_L, a_R, var_type)
                # rhoE
                var_type = 4
                rhoE_F_m_z = HLLC_flux(rho_L, rho_R, V_n_zeta_L_m, V_n_zeta_R_m, V_t1_zeta_L_m, V_t1_zeta_R_m, V_t2_zeta_L_m, V_t2_zeta_R_m, E_L, E_R, s_L, s_R, P_L, P_R, T_L,
                                      T_R, a_L, a_R, var_type)
                                
                rhou_F_m_z = rho_n_F_m_z * n_zeta_x + rho_t1_F_m_z * t1_zeta_x + rho_t2_F_m_z * t2_zeta_x
                rhov_F_m_z = rho_n_F_m_z * n_zeta_y + rho_t1_F_m_z * t1_zeta_y + rho_t2_F_m_z * t2_zeta_y
                rhow_F_m_z = rho_n_F_m_z * n_zeta_z + rho_t1_F_m_z * t1_zeta_z + rho_t2_F_m_z * t2_zeta_z
                
                ## Fluxes x-direction  (dF/dxi)
                rho_inv[i][j][k]  = (1.0 / delta_x) * (rho_F_p_x  - rho_F_m_x)
                rhou_inv[i][j][k] = (1.0 / delta_x) * (rhou_F_p_x - rhou_F_m_x)
                rhov_inv[i][j][k] = (1.0 / delta_x) * (rhov_F_p_x - rhov_F_m_x)
                rhow_inv[i][j][k] = (1.0 / delta_x) * (rhow_F_p_x - rhow_F_m_x)
                rhoE_inv[i][j][k] = (1.0 / delta_x) * (rhoE_F_p_x - rhoE_F_m_x)

                ## Fluxes eta-direction: single contravariant flux Q = |grad eta| * F_neta,
                ## differenced conservatively with face-consistent metrics (mge_p, mge_m),
                ## plus the geometric source (L_y'/L_y)*F_x. This replaces the unstable
                ## central cross-term and preserves free-stream (the metric variation cancels).
                src = drc_dx[i] / L_y[i]
                rho_inv[i][j][k]  += (mge_p * rho_F_p_y  - mge_m * rho_F_m_y ) / delta_y + src * 0.5 * (rho_F_p_x  + rho_F_m_x)
                rhou_inv[i][j][k] += (mge_p * rhou_F_p_y - mge_m * rhou_F_m_y) / delta_y + src * 0.5 * (rhou_F_p_x + rhou_F_m_x)
                rhov_inv[i][j][k] += (mge_p * rhov_F_p_y - mge_m * rhov_F_m_y) / delta_y + src * 0.5 * (rhov_F_p_x + rhov_F_m_x)
                rhow_inv[i][j][k] += (mge_p * rhow_F_p_y - mge_m * rhow_F_m_y) / delta_y + src * 0.5 * (rhow_F_p_x + rhow_F_m_x)
                rhoE_inv[i][j][k] += (mge_p * rhoE_F_p_y - mge_m * rhoE_F_m_y) / delta_y + src * 0.5 * (rhoE_F_p_x + rhoE_F_m_x)

                ## Fluxes z-direction  (dH/dzeta)
                rho_inv[i][j][k] += (1.0 / delta_z) * (rho_F_p_z - rho_F_m_z)
                rhou_inv[i][j][k] += (1.0 / delta_z) * (rhou_F_p_z - rhou_F_m_z)
                rhov_inv[i][j][k] += (1.0 / delta_z) * (rhov_F_p_z - rhov_F_m_z)
                rhow_inv[i][j][k] += (1.0 / delta_z) * (rhow_F_p_z - rhow_F_m_z)
                rhoE_inv[i][j][k] += (1.0 / delta_z) * (rhoE_F_p_z - rhoE_F_m_z)
    # print( rho_inv )
    # print( rhou_inv )
    # print( rhov_inv )
    # print( rhow_inv )
    # print( rhoE_inv )


### Calculate viscous fluxes
@njit
def viscous_fluxes(rhou_vis, rhov_vis, rhow_vis, rhoE_vis, work_vis_rhoe, u, v, w, T, mu, kappa, grid):
    # Second-order central finite differences for derivatives:
    # P. Moin.
    # Fundamentals of engineering numerical analysis.
    # Cambridge University Press, 2010.

    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                ## Geometric stuff and derivatives
                delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0])
                delta_y = 0.5 * (grid[i][j + 1][k][1] - grid[i][j - 1][k][1])
                delta_z = 0.5 * (grid[i][j][k + 1][2] - grid[i][j][k - 1][2])
                d_N_d_x = - 1.0 * grid[i][j][k][1]/L_y[i]*drc_dx[i]
                d_N_d_x_d_E = grid[i][j][k][1]/(L_y[i]**2)*drc_dx[i]**2 - grid[i][j][k][1]/L_y[i] * drc_dx_dx[i] # CHANGE CENTRAL DIFFERENCE FOR ANALYTICAL DERIVATIVE (def drc_dx_dx)
                d_N_d_x_d_N = - 1.0/L_y[i]*drc_dx[i]
                # if k == 1:
                #     print(grid[i][1][1][0], drc_dx[i], drc_dx_dx[i], d_N_d_x, d_N_d_x_d_E, d_N_d_x_d_N)
                ## Velocity derivatives
                d_u_x = (u[i + 1][j][k] - u[i - 1][j][k]) / (2.0 * delta_x) - (u[i][j+1][k] - u[i][j-1][k]) / (2.0 * delta_y) * (grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_u_y = (u[i][j + 1][k] - u[i][j - 1][k]) / (2.0 * delta_y) * 1.0/L_y[i]
                d_u_z = (u[i][j][k + 1] - u[i][j][k - 1]) / (2.0 * delta_z)
                d_u_d_xx = (1.0 / delta_x) * ((u[i+1][j][k]-u[i][j][k])/(grid[i+1][j][k][0]-grid[i][j][k][0])-(u[i][j][k]-u[i-1][j][k])/(grid[i][j][k][0]-grid[i-1][j][k][0]))
                d_u_d_yy = (1.0 / delta_y) * ((u[i][j+1][k]-u[i][j][k])/(grid[i][j+1][k][1]-grid[i][j][k][1])-(u[i][j][k]-u[i][j-1][k])/(grid[i][j][k][1]-grid[i][j-1][k][1]))
                d_u_d_zz = (1.0 / delta_z) * ((u[i][j][k+1]-u[i][j][k])/(grid[i][j][k+1][2]-grid[i][j][k][2])-(u[i][j][k]-u[i][j][k-1])/(grid[i][j][k][2]-grid[i][j][k-1][2]))
                d_u_d_xy = (0.25 / delta_x) * ((u[i+1][j+1][k]-u[i+1][j-1][k])/delta_y-(u[i-1][j+1][k]-u[i-1][j-1][k])/delta_y)
                d_u_d_yx = (0.25 / delta_y) * ((u[i+1][j+1][k]-u[i-1][j+1][k])/delta_x-(u[i+1][j-1][k]-u[i-1][j-1][k])/delta_x)
                # d_u_d_xz = (0.25 / delta_x) * ((u[i+1][j][k+1]-u[i+1][j][k-1])/delta_z-(u[i-1][j][k+1]-u[i-1][j][k-1])/delta_z)
                d_u_d_zx = (0.25 / delta_z) * ((u[i+1][j][k+1]-u[i-1][j][k+1])/delta_x-(u[i+1][j][k-1]-u[i-1][j][k-1])/delta_x)
                # d_u_d_yz = (0.25 / delta_y) * ((u[i][j+1][k+1]-u[i][j+1][k-1])/delta_z-(u[i][j-1][k+1]-u[i][j-1][k-1])/delta_z)
                d_u_d_zy = (0.25 / delta_z) * ((u[i][j+1][k+1]-u[i][j-1][k+1])/delta_y-(u[i][j+1][k-1]-u[i][j-1][k-1])/delta_y)
                ####################
                d_v_x = (v[i + 1][j][k] - v[i - 1][j][k]) / (2.0 * delta_x) - (v[i][j+1][k] - v[i][j-1][k]) / (2.0 * delta_y) * (grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_v_y = (v[i][j + 1][k] - v[i][j - 1][k]) / (2.0 * delta_y) * 1.0/L_y[i]
                d_v_z = (v[i][j][k + 1] - v[i][j][k - 1]) / (2.0 * delta_z)
                d_v_d_xx = (1.0 / delta_x) * ((v[i+1][j][k]-v[i][j][k])/(grid[i+1][j][k][0]-grid[i][j][k][0])-(v[i][j][k]-v[i-1][j][k])/(grid[i][j][k][0]-grid[i-1][j][k][0]))
                d_v_d_yy = (1.0 / delta_y) * ((v[i][j+1][k]-v[i][j][k])/(grid[i][j+1][k][1]-grid[i][j][k][1])-(v[i][j][k]-v[i][j-1][k])/(grid[i][j][k][1]-grid[i][j-1][k][1]))
                d_v_d_zz = (1.0 / delta_z) * ((v[i][j][k+1]-v[i][j][k])/(grid[i][j][k+1][2]-grid[i][j][k][2])-(v[i][j][k]-v[i][j][k-1])/(grid[i][j][k][2]-grid[i][j][k-1][2]))
                d_v_d_xy = (0.25 / delta_x) * ((v[i+1][j+1][k]-v[i+1][j-1][k])/delta_y-(v[i-1][j+1][k]-v[i-1][j-1][k])/delta_y)
                d_v_d_yx = (0.25 / delta_y) * ((v[i+1][j+1][k]-v[i-1][j+1][k])/delta_x-(v[i+1][j-1][k]-v[i-1][j-1][k])/delta_x)
                # d_v_d_xz = (0.25 / delta_x) * ((v[i+1][j][k+1]-v[i+1][j][k-1])/delta_z-(v[i-1][j][k+1]-v[i-1][j][k-1])/delta_z)
                # d_v_d_zx = (0.25 / delta_z) * ((v[i+1][j][k+1]-v[i-1][j][k+1])/delta_x-(v[i+1][j][k-1]-v[i-1][j][k-1])/delta_x)
                # d_v_d_yz = (0.25 / delta_y) * ((v[i][j+1][k+1]-v[i][j+1][k-1])/delta_z-(v[i][j-1][k+1]-v[i][j-1][k-1])/delta_z)
                d_v_d_zy = (0.25 / delta_z) * ((v[i][j+1][k+1]-v[i][j-1][k+1])/delta_y-(v[i][j+1][k-1]-v[i][j-1][k-1])/delta_y)
                #####################
                d_w_x = (w[i + 1][j][k] - w[i - 1][j][k]) / (2.0 * delta_x) - (w[i][j+1][k] - w[i][j-1][k]) / (2.0 * delta_y) * (grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_w_y = (w[i][j + 1][k] - w[i][j - 1][k]) / (2.0 * delta_y) * 1.0/L_y[i]
                d_w_z = (w[i][j][k + 1] - w[i][j][k - 1]) / (2.0 * delta_z)
                d_w_d_xx = (1.0 / delta_x) * ((w[i+1][j][k]-w[i][j][k])/(grid[i+1][j][k][0]-grid[i][j][k][0])-(w[i][j][k]-w[i-1][j][k])/(grid[i][j][k][0]-grid[i-1][j][k][0]))
                d_w_d_yy = (1.0 / delta_y) * ((w[i][j+1][k]-w[i][j][k])/(grid[i][j+1][k][1]-grid[i][j][k][1])-(w[i][j][k]-w[i][j-1][k])/(grid[i][j][k][1]-grid[i][j-1][k][1]))
                d_w_d_zz = (1.0 / delta_z) * ((w[i][j][k+1]-w[i][j][k])/(grid[i][j][k+1][2]-grid[i][j][k][2])-(w[i][j][k]-w[i][j][k-1])/(grid[i][j][k][2]-grid[i][j][k-1][2]))
                d_w_d_xy = (0.25 / delta_x) * ((w[i+1][j+1][k]-w[i+1][j-1][k])/delta_y-(w[i-1][j+1][k]-w[i-1][j-1][k])/delta_y)
                d_w_d_yx = (0.25 / delta_y) * ((w[i+1][j+1][k]-w[i-1][j+1][k])/delta_x-(w[i+1][j-1][k]-w[i-1][j-1][k])/delta_x)
                d_w_d_xz = (0.25 / delta_x) * ((w[i+1][j][k+1]-w[i+1][j][k-1])/delta_z-(w[i-1][j][k+1]-w[i-1][j][k-1])/delta_z)
                # d_w_d_zx = (0.25 / delta_z) * ((w[i+1][j][k+1]-w[i-1][j][k+1])/delta_x-(w[i+1][j][k-1]-w[i-1][j][k-1])/delta_x)
                d_w_d_yz = (0.25 / delta_y) * ((w[i][j+1][k+1]-w[i][j+1][k-1])/delta_z-(w[i][j-1][k+1]-w[i][j-1][k-1])/delta_z)
                # d_w_d_zy = (0.25 / delta_z) * ((w[i][j+1][k+1]-w[i][j-1][k+1])/delta_y-(w[i][j+1][k-1]-w[i][j-1][k-1])/delta_y)
                ## Temperature derivatives
                d_T_x = (T[i + 1][j][k] - T[i - 1][j][k]) / (2.0 * delta_x) - (T[i][j+1][k] - T[i][j-1][k]) / (2.0 * delta_y) * (grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_T_y = (T[i][j + 1][k] - T[i][j - 1][k]) / (2.0 * delta_y) * 1.0/L_y[i]
                d_T_z = (T[i][j][k + 1] - T[i][j][k - 1]) / (2.0 * delta_z)
                d_T_d_xx = (1.0 / delta_x) * ((T[i+1][j][k]-T[i][j][k])/(grid[i+1][j][k][0]-grid[i][j][k][0])-(T[i][j][k]-T[i-1][j][k])/(grid[i][j][k][0]-grid[i-1][j][k][0]))
                d_T_d_yy = (1.0 / delta_y) * ((T[i][j+1][k]-T[i][j][k])/(grid[i][j+1][k][1]-grid[i][j][k][1])-(T[i][j][k]-T[i][j-1][k])/(grid[i][j][k][1]-grid[i][j-1][k][1]))
                d_T_d_zz = (1.0 / delta_z) * ((T[i][j][k+1]-T[i][j][k])/(grid[i][j][k+1][2]-grid[i][j][k][2])-(T[i][j][k]-T[i][j][k-1])/(grid[i][j][k][2]-grid[i][j][k-1][2]))
                d_T_d_xy = (0.25 / delta_x) * ((T[i+1][j+1][k]-T[i+1][j-1][k])/delta_y-(T[i-1][j+1][k]-T[i-1][j-1][k])/delta_y)
                d_T_d_yx = (0.25 / delta_y) * ((T[i+1][j+1][k]-T[i-1][j+1][k])/delta_x-(T[i+1][j-1][k]-T[i-1][j-1][k])/delta_x)
                ## Transport coefficients derivatives
                d_mu_x = (mu[i + 1][j][k] - mu[i - 1][j][k]) / (2.0 * delta_x) - (mu[i][j+1][k] - mu[i][j-1][k]) / (2.0 * delta_y) * (grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_mu_y = (mu[i][j + 1][k] - mu[i][j - 1][k]) / (2.0 * delta_y) * 1.0/L_y[i]
                d_mu_z = (mu[i][j][k + 1] - mu[i][j][k - 1]) / (2.0 * delta_z)
                d_kappa_x = (kappa[i + 1][j][k] - kappa[i - 1][j][k]) / (2.0 * delta_x) - (kappa[i][j+1][k] - kappa[i][j-1][k]) / (2.0 * delta_y) * (grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_kappa_y = (kappa[i][j + 1][k] - kappa[i][j - 1][k]) / (2.0 * delta_y) * 1.0/L_y[i]
                d_kappa_z = (kappa[i][j][k + 1] - kappa[i][j][k - 1]) / (2.0 * delta_z)
                ## Divergence of velocity
                div_uvw = d_u_x + d_v_y + d_w_z
                ## Viscous stresses ( symmetric tensor )
                tau_xx = 2.0 * mu[i][j][k] * (d_u_x - (div_uvw / 3.0))
                tau_xy = mu[i][j][k] * (d_u_y + d_v_x)
                tau_xz = mu[i][j][k] * (d_u_z + d_w_x)
                tau_yy = 2.0 * mu[i][j][k] * (d_v_y - (div_uvw / 3.0))
                tau_yz = mu[i][j][k] * (d_v_z + d_w_y)
                tau_zz = 2.0 * mu[i][j][k] * (d_w_z - (div_uvw / 3.0))
               
                ## Divergence of viscous stresses
                # div_tau_x = mu[i][j][k]*( ( 1.00/delta_x )*( ( u[i+1][j][k] - u[i][j][k] )/( grid[i+1][j][k][0] - grid[i][j][k][0] )
                #                                            - ( u[i][j][k] - u[i-1][j][k] )/( grid[i][j][k][0] - grid[i-1][j][k][0] ) )
                #                         + ( 1.00/delta_y )*( ( u[i][j+1][k] - u[i][j][k] )/( grid[i][j+1][k][1] - grid[i][j][k][1] )
                #                                            - ( u[i][j][k] - u[i][j-1][k] )/( grid[i][j][k][1] - grid[i][j-1][k][1] ) )
                #                         + ( 1.00/delta_z )*( ( u[i][j][k+1] - u[i][j][k] )/( grid[i][j][k+1][2] - grid[i][j][k][2] )
                #                                            - ( u[i][j][k] - u[i][j][k-1] )/( grid[i][j][k][2] - grid[i][j][k-1][2] ) ) )                                                                                                                                    + ( 1.0/3.0 )*mu[i][j][k]*( ( 1.00/delta_x )*( ( u[i+1][j][k] - u[i][j][k] )/( grid[i+1][j][k][0] - grid[i][j][k][0] )
                #                                            - ( u[i][j][k] - u[i-1][j][k] )/( grid[i][j][k][0] - grid[i-1][j][k][0] ) )
                #                         + ( 0.25/delta_x )*( ( v[i+1][j+1][k] - v[i+1][j-1][k] )/delta_y
                #                                            - ( v[i-1][j+1][k] - v[i-1][j-1][k] )/delta_y )
                #                         + ( 0.25/delta_x )*( ( w[i+1][j][k+1] - w[i+1][j][k-1] )/delta_z
                #                                            - ( w[i-1][j][k+1] - w[i-1][j][k-1] )/delta_z ) )                                                                                                                                                                + ( d_mu_x*tau_xx + d_mu_y*tau_xy + d_mu_z*tau_xz )/( mu[i][j][k] + epsilon )
                # div_tau_y = mu[i][j][k]*( ( 1.00/delta_x )*( ( v[i+1][j][k] - v[i][j][k] )/( grid[i+1][j][k][0] - grid[i][j][k][0] )
                #                                            - ( v[i][j][k] - v[i-1][j][k] )/( grid[i][j][k][0] - grid[i-1][j][k][0] ) )
                #                         + ( 1.00/delta_y )*( ( v[i][j+1][k] - v[i][j][k] )/( grid[i][j+1][k][1] - grid[i][j][k][1] )
                #                                            - ( v[i][j][k] - v[i][j-1][k] )/( grid[i][j][k][1] - grid[i][j-1][k][1] ) )
                #                         + ( 1.00/delta_z )*( ( v[i][j][k+1] - v[i][j][k] )/( grid[i][j][k+1][2] - grid[i][j][k][2] )
                #                                            - ( v[i][j][k] - v[i][j][k-1] )/( grid[i][j][k][2] - grid[i][j][k-1][2] ) ) )                                                                                                                                    + ( 1.0/3.0 )*mu[i][j][k]*( ( 0.25/delta_y )*( ( u[i+1][j+1][k] - u[i-1][j+1][k] )/delta_x
                #                                            - ( u[i+1][j-1][k] - u[i-1][j-1][k] )/delta_x )
                #                         + ( 1.00/delta_y )*( ( v[i][j+1][k] - v[i][j][k] )/( grid[i][j+1][k][1] - grid[i][j][k][1] )
                #                                            - ( v[i][j][k] - v[i][j-1][k] )/( grid[i][j][k][1] - grid[i][j-1][k][1] ) )
                #                         + ( 0.25/delta_y )*( ( w[i][j+1][k+1] - w[i][j+1][k-1] )/delta_z
                #                                            - ( w[i][j-1][k+1] - w[i][j-1][k-1] )/delta_z ) )                                                                                                                                                                + ( d_mu_x*tau_xy + d_mu_y*tau_yy + d_mu_z*tau_yz )/( mu[i][j][k] + epsilon )
                # div_tau_z = mu[i][j][k]*( ( 1.00/delta_x )*( ( w[i+1][j][k] - w[i][j][k] )/( grid[i+1][j][k][0] - grid[i][j][k][0] )
                #                                            - ( w[i][j][k] - w[i-1][j][k] )/( grid[i][j][k][0] - grid[i-1][j][k][0] ) )
                #                         + ( 1.00/delta_y )*( ( w[i][j+1][k] - w[i][j][k] )/( grid[i][j+1][k][1] - grid[i][j][k][1] )
                #                                            - ( w[i][j][k] - w[i][j-1][k] )/( grid[i][j][k][1] - grid[i][j-1][k][1] ) )
                #                         + ( 1.00/delta_z )*( ( w[i][j][k+1] - w[i][j][k] )/( grid[i][j][k+1][2] - grid[i][j][k][2] )
                #                                            - ( w[i][j][k] - w[i][j][k-1] )/( grid[i][j][k][2] - grid[i][j][k-1][2] ) ) )                                                                                                                                    + ( 1.0/3.0 )*mu[i][j][k]*( ( 0.25/delta_z )*( ( u[i+1][j][k+1] - u[i-1][j][k+1] )/delta_x
                #                                            - ( u[i+1][j][k-1] - u[i-1][j][k-1] )/delta_x )
                #                         + ( 0.25/delta_z )*( ( v[i][j+1][k+1] - v[i][j-1][k+1] )/delta_y
                #                                            - ( v[i][j+1][k-1] - v[i][j-1][k-1] )/delta_y )
                #                         + ( 1.00/delta_z )*( ( w[i][j][k+1] - w[i][j][k] )/( grid[i][j][k+1][2] - grid[i][j][k][2] )
                #                                            - ( w[i][j][k] - w[i][j][k-1] )/( grid[i][j][k][2] - grid[i][j][k-1][2] ) ) )                                                                                                                                    + ( d_mu_x*tau_xz + d_mu_y*tau_yz + d_mu_z*tau_zz )/( mu[i][j][k] + epsilon )
                # ## Fourier term
                # div_q = ( -1.0 )*kappa[i][j][k]*( ( 1.0/delta_x )*( ( T[i+1][j][k] - T[i][j][k] )/( grid[i+1][j][k][0] - grid[i][j][k][0] )
                #                                                   - ( T[i][j][k] - T[i-1][j][k] )/( grid[i][j][k][0] - grid[i-1][j][k][0] ) )
                #                                 + ( 1.0/delta_y )*( ( T[i][j+1][k] - T[i][j][k] )/( grid[i][j+1][k][1] - grid[i][j][k][1] )
                #                                                   - ( T[i][j][k] - T[i][j-1][k] )/( grid[i][j][k][1] - grid[i][j-1][k][1] ) )
                #                                 + ( 1.0/delta_z )*( ( T[i][j][k+1] - T[i][j][k] )/( grid[i][j][k+1][2] - grid[i][j][k][2] )
                #                                                   - ( T[i][j][k] - T[i][j][k-1] )/( grid[i][j][k][2] - grid[i][j][k-1][2] ) ) )                                                                                                                                     - d_kappa_x*d_T_x - d_kappa_y*d_T_y - d_kappa_z*d_T_z



                div_tau_x = ( mu[i][j][k]*( (d_u_d_xx + d_N_d_x_d_E*d_u_y + d_N_d_x*d_u_d_xy + d_N_d_x*d_u_d_yx + d_N_d_x*d_N_d_x_d_N*d_u_y + d_N_d_x**2*d_u_d_yy) + ((1.0/L_y[i])**2*d_u_d_yy) + (d_u_d_zz) )
                + ( 1.0/3.0 )*mu[i][j][k]*( (d_u_d_xx + d_N_d_x_d_E*d_u_y + d_N_d_x*d_u_d_xy + d_N_d_x*d_u_d_yx + d_N_d_x*d_N_d_x_d_N*d_u_y + d_N_d_x**2*d_u_d_yy) + (-1.0/(L_y[i]**2)*drc_dx[i]*d_v_y+1.0/L_y[i]*d_v_d_xy+d_N_d_x*1.0/L_y[i]*d_v_d_yy) + (d_w_d_xz + d_N_d_x*d_w_d_yz) )
                + ( d_mu_x*tau_xx + d_mu_y*tau_xy + d_mu_z*tau_xz )/( mu[i][j][k] + epsilon ) )

                div_tau_y = ( mu[i][j][k]*( (d_v_d_xx + d_N_d_x_d_E*d_v_y + d_N_d_x*d_v_d_xy + d_N_d_x*d_v_d_yx + d_N_d_x*d_N_d_x_d_N*d_v_y + d_N_d_x**2*d_v_d_yy) + ((1.0/L_y[i])**2*d_v_d_yy) + (d_v_d_zz) )
                + ( 1.0/3.0 )*mu[i][j][k]*( (1.0/L_y[i]*(d_u_d_yx-1.0/L_y[i]*drc_dx[i]*d_u_y + d_u_d_yy*d_N_d_x)) + ((1.0/L_y[i])**2*d_v_d_yy) + (1.0/L_y[i]*d_w_d_yz) )
                + ( d_mu_x*tau_xy + d_mu_y*tau_yy + d_mu_z*tau_yz )/( mu[i][j][k] + epsilon ) )

                div_tau_z = ( mu[i][j][k]*( (d_w_d_xx + d_N_d_x_d_E*d_w_y + d_N_d_x*d_w_d_xy + d_N_d_x*d_w_d_yx + d_N_d_x*d_N_d_x_d_N*d_w_y + d_N_d_x**2*d_w_d_yy) + ((1.0/L_y[i])**2*d_w_d_yy) + (d_w_d_zz) )
                + ( 1.0/3.0 )*mu[i][j][k]*( (d_u_d_zx + d_N_d_x*d_u_d_zy) + (1.0/L_y[i]*d_v_d_zy) + (d_w_d_zz) )
                + ( d_mu_x*tau_xz + d_mu_y*tau_yz + d_mu_z*tau_zz )/( mu[i][j][k] + epsilon ) )
            
                # Fourier term
                div_q = ( -1.0 )*kappa[i][j][k]*( (d_T_d_xx + d_N_d_x_d_E*d_T_y + d_N_d_x*d_T_d_xy + d_N_d_x*d_T_d_yx + d_N_d_x*d_N_d_x_d_N*d_T_y + d_N_d_x**2*d_T_d_yy) 
                                                 + (1.0/L_y[i])**2*d_T_d_yy 
                                                 + d_T_d_zz ) - d_kappa_x*d_T_x - d_kappa_y*d_T_y - d_kappa_z*d_T_z


                # print(og_div_tau_x, div_tau_x)
                # print(og_div_tau_y, div_tau_y)
                # print(og_div_tau_z, div_tau_z)
                # print(og_div_q, div_q)

                ## Work of viscous stresses for internal energy
                div_uvw_tau_rhoe = tau_xx * d_u_x + tau_xy * d_u_y + tau_xz * d_u_z + tau_xy * d_v_x + tau_yy * d_v_y + tau_yz * d_v_z + tau_xz * d_w_x + tau_yz * d_w_y + tau_zz * d_w_z
                ## Work of viscous stresses for kinetic energy
                div_uvw_tau_rhoke = u[i][j][k] * div_tau_x + v[i][j][k] * div_tau_y + w[i][j][k] * div_tau_z
                ## Work of viscous stresses for total energy
                div_uvw_tau_rhoE = div_uvw_tau_rhoe + div_uvw_tau_rhoke
                ## Viscous fluxes
                rhou_vis[i][j][k] = div_tau_x
                rhov_vis[i][j][k] = div_tau_y
                rhow_vis[i][j][k] = div_tau_z
                rhoE_vis[i][j][k] = (-1.0) * div_q + div_uvw_tau_rhoE
                work_vis_rhoe[i][j][k] = (-1.0) * div_q + div_uvw_tau_rhoe
    # print( rhou_vis )
    # print( rhov_vis )
    # print( rhow_vis )
    # print( rhoE_vis )
    # print( work_vis_rhoe )


### Sum inviscid & viscous fluxes and source terms
# @njit
def sum_fluxes_source_terms(rho_tot, rhou_tot, rhov_tot, rhow_tot, rhoE_tot, P_tot, rho_inv, rhou_inv, rhov_inv,
                            rhow_inv, rhoE_inv, rhou_vis, rhov_vis, rhow_vis, rhoE_vis, work_vis_rhoe, f_rhou, f_rhov,
                            f_rhow, f_rhoE, rho, u, v, w, P, T, sos, rk_iter, grid):
    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                # Calculate specific heat capacities
                c_v = -1.0
                c_p = -1.0
                c_v, c_p = thermodynamics.calculateSpecificHeatCapacities(c_v, c_p, P[i][j][k], T[i][j][k],
                                                                          rho[i][j][k])
                ## Geometric stuff
                delta_x = 0.5 * (grid[i + 1][j][k][0] - grid[i - 1][j][k][0])
                delta_y = 0.5 * (grid[i][j + 1][k][1] - grid[i][j - 1][k][1])
                delta_z = 0.5 * (grid[i][j][k + 1][2] - grid[i][j][k - 1][2])
                ## Pressure and velocity derivatives
                d_P_x = ((P[i + 1][j][k] - P[i - 1][j][k]) / (2.0 * delta_x)) - ((P[i][j+1][k] - P[i][j-1][k]) / (2.0 * delta_y))*(grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_P_y = (P[i][j + 1][k] - P[i][j - 1][k]) / (2.0 * delta_y) * (1.0/L_y[i])
                d_P_z = (P[i][j][k + 1] - P[i][j][k - 1]) / (2.0 * delta_z)
                d_u_x = (u[i + 1][j][k] - u[i - 1][j][k]) / (2.0 * delta_x) - ((u[i][j+1][k] - u[i][j-1][k]) / (2.0 * delta_y))*(grid[i][j][k][1]/L_y[i]*drc_dx[i])
                d_v_y = (v[i][j + 1][k] - v[i][j - 1][k]) / (2.0 * delta_y) * (1.0/L_y[i])
                d_w_z = (w[i][j][k + 1] - w[i][j][k - 1]) / (2.0 * delta_z)
                ## Divergence of velocity
                div_uvw = d_u_x + d_v_y + d_w_z
                ## Pressure inviscid flux
                P_inv_flux = u[i][j][k] * d_P_x + v[i][j][k] * d_P_y + w[i][j][k] * d_P_z + rho[i][j][k] * (
                            sos[i][j][k] ** 2.0) * div_uvw
                ## Pressure viscous flux
                volume_expansivity = 1.0 / T[i][j][k]
                isothermal_compressibility = 1.0 / P[i][j][k]
                P_vis_flux = (volume_expansivity / (rho[i][j][k] * c_v * isothermal_compressibility)) * \
                             work_vis_rhoe[i][j][k]
                ## Calculate total right-hand side
                rho_tot[i][j][k][rk_iter] = (-1.0) * rho_inv[i][j][k]
                rhou_tot[i][j][k][rk_iter] = (-1.0) * rhou_inv[i][j][k] + rhou_vis[i][j][k] + f_rhou[i][j][k]
                rhov_tot[i][j][k][rk_iter] = (-1.0) * rhov_inv[i][j][k] + rhov_vis[i][j][k] + f_rhov[i][j][k]
                rhow_tot[i][j][k][rk_iter] = (-1.0) * rhow_inv[i][j][k] + rhow_vis[i][j][k] + f_rhow[i][j][k]
                rhoE_tot[i][j][k][rk_iter] = (-1.0) * rhoE_inv[i][j][k] + rhoE_vis[i][j][k] + f_rhoE[i][j][k]
                P_tot[i][j][k][rk_iter] = (-1.0) * P_inv_flux + P_vis_flux + f_rhoE[i][j][k]
    # print( rho_tot )
    # print( rhou_tot )
    # print( rhov_tot )
    # print( rhow_tot )
    # print( rhoE_tot )
    # print( P_tot )


### Time integration of conserved variables
@njit
def time_integration(y, y_0, h, k_s, rk_iter):
    # Explicit third-order strong-stability-preserving Runge-Kutta (SSP-RK3) method:
    # S. Gottlieb, C.-W. Shu & E. Tadmor.
    # Strong stability-preserving high-order time discretization methods.
    # SIAM Review 43, 89-112, 2001.

    # Internal points
    for i in range(1, num_grid_x + 1):
        for j in range(1, num_grid_y + 1):
            for k in range(1, num_grid_z + 1):
                if (rk_iter == 0):
                    y[i][j][k] = y_0[i][j][k] + h * k_s[i][j][k][0]
                elif (rk_iter == 1):
                    y[i][j][k] = y_0[i][j][k] + (h / 4.0) * (k_s[i][j][k][0] + k_s[i][j][k][1])
                elif (rk_iter == 2):
                    y[i][j][k] = y_0[i][j][k] + (h / 6.0) * (k_s[i][j][k][0] + k_s[i][j][k][1] + 4.0 * k_s[i][j][k][2])
    # print( y )


### Update primitive variables from conserved variables
@njit
def update_primitive(primitive, conserved, rho):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                primitive[i][j][k] = (1.0 / rho[i][j][k]) * conserved[i][j][k]
    # print( primitive )


### Update thermodynamic variables from primitive variables
# @njit
def thermodynamic_state(rhoE, P, T, rho, u, v, w, E, s):
    # All points
    for i in range(0, num_grid_x + 2):
        for j in range(0, num_grid_y + 2):
            for k in range(0, num_grid_z + 2):
                # Specific kinetic energy
                ke = 0.5 * (u[i][j][k] ** 2.0 + v[i][j][k] ** 2.0 + w[i][j][k] ** 2.0)
                if (transport_pressure_scheme):
                    # Temperature
                    T_aux = T[i][j][k]
                    T_aux = thermodynamics.calculateTemperatureFromPressureDensityWithInitialGuess(T_aux, P[i][j][k],
                                                                                                   rho[i][j][k])
                    T[i][j][k] = T_aux
                    # Specific internal energy
                    e = thermodynamics.calculateInternalEnergyFromPressureTemperatureDensity(P[i][j][k], T[i][j][k],
                                                                                             rho[i][j][k])
                    # Specific total energy and rhoE
                    E[i][j][k] = e + ke
                    rhoE[i][j][k] = rho[i][j][k] * E[i][j][k]
                else:
                    # Specific internal energy
                    e = E[i][j][k] - ke
                    # Pressure & Temperature
                    P_aux = P[i][j][k]
                    T_aux = T[i][j][k]
                    P_aux, T_aux = thermodynamics.calculatePressureTemperatureFromDensityInternalEnergy(P_aux, T_aux,
                                                                                                        rho[i][j][k], e)
                    P[i][j][k] = P_aux
                    T[i][j][k] = T_aux
                s[i][j][k] = thermodynamics.calculateEntropyFromPressureTemperatureDensity(P[i][j][k], T[i][j][k],
                                                                                           rho[i][j][k])
    # print( E )
    # print( s )
    # print( rhoE )
    # print( P )
    # print( T )


########## MAIN ##########

### START SIMULATION
print('RHEA: START SIMULATION')

### Define spatial discretization
spatial_discretization(physical_plane)

### Define computational domain
generate_computationalDomain(physical_plane, computational_plane)

# Uncomment segment to generate physical and computational grids plots
#################################################################
plt.figure(); plt.scatter(physical_plane[:,:,1,0],physical_plane[:,:,1,1],s=2); plt.title('physical grid'); plt.savefig('grid_physical.png',dpi=120); plt.close()
plt.figure(); plt.scatter(computational_plane[:,:,1,0],computational_plane[:,:,1,1],s=2); plt.title('computational grid'); plt.savefig('grid_computational.png',dpi=120); plt.close()
# exit()
#################################################################

### Generate L_y vector in terms of x-distance (wall radius at each i-column)
L_y = np.zeros(num_grid_x+2)
for i in range(0,num_grid_x+2):
    L_y[i] = 0.5*(physical_plane[i][-1][1][1]+physical_plane[i][-2][1][1])

### Differentiate the contour numerically for the transform metrics.
# The curvilinear transform eta = y/L_y(x) needs drc_dx = dL_y/dx and drc_dx_dx = d2L_y/dx2.
# Computing them by finite differences of L_y(x) (instead of per-zone analytic formulas)
# lets the legacy method run ANY single-contour nozzle, the 4-zone one included.
x_axis = physical_plane[:, 1, 1, 0]   # x depends only on i
for i in range(1, num_grid_x + 1):
    h_m = x_axis[i]   - x_axis[i-1]
    h_p = x_axis[i+1] - x_axis[i]
    drc_dx[i]    = (L_y[i+1] - L_y[i-1]) / (x_axis[i+1] - x_axis[i-1])
    drc_dx_dx[i] = 2.0 * (h_m * L_y[i+1] - (h_m + h_p) * L_y[i] + h_p * L_y[i-1]) \
                   / (h_m * h_p * (h_m + h_p))
# Copy into the boundary (ghost) columns.
drc_dx[0]    = drc_dx[1];          drc_dx[num_grid_x+1]    = drc_dx[num_grid_x]
drc_dx_dx[0] = drc_dx_dx[1];       drc_dx_dx[num_grid_x+1] = drc_dx_dx[num_grid_x]

### Initialize u, v, w, P and T variables
if (use_restart):
    time, time_iter = read_file_uvwPT(u_field, v_field, w_field, P_field, T_field, computational_plane)
else:
    time = initial_time
    time_iter = 0
    initialize_uvwPT(u_field, v_field, w_field, P_field, T_field, computational_plane)

### Initialize thermodynamic variables
initialize_thermodynamics(rho_field, E_field, s_field, u_field, v_field, w_field, P_field, T_field)
if (artificial_compressibility_method):
    P_thermo = calculate_volume_averaged_value(P_field, computational_plane)
    alpha_acm = calculate_alpha_acm(P_field, P_thermo, computational_plane)
calculate_speed_sound(sos_field, rho_field, P_field, P_thermo, T_field)

### Update boundaries
update_boundaries(sos_field, rho_field, rhou_field, rhov_field, rhow_field, rhoE_field, u_field, v_field,
                          w_field, P_field, T_field, computational_plane)

### Calculate transport coefficients
calculate_transport_coefficients(mu_field, kappa_field, P_field, T_field, rho_field)

### Update conserved variables from primitive variables
update_conserved(rhou_field, u_field, rho_field)
update_conserved(rhov_field, v_field, rho_field)
update_conserved(rhow_field, w_field, rho_field)
update_conserved(rhoE_field, E_field, rho_field)

### Update old fields of conserved variables
update_field(rho_0_field, rho_field)
update_field(rhou_0_field, rhou_field)
update_field(rhov_0_field, rhov_field)
update_field(rhow_0_field, rhow_field)
update_field(rhoE_0_field, rhoE_field)
update_field(P_0_field, P_field)

while time_iter < max_num_time_iter:
    ### Calculate time step
    delta_t = time_step(rho_field, u_field, v_field, w_field, P_field, T_field, sos_field, mu_field, kappa_field, computational_plane)
    if ((time + delta_t) > final_time):
        delta_t = final_time - time
        # print( delta_t )

    ### Print time iteration information
    inlet_V = np.sqrt(
        (0.5 * (u_field[0, 1, 1] + u_field[1, 1, 1])) ** 2 + (0.5 * (v_field[0, 1, 1] + v_field[1, 1, 1])) ** 2 + (
                    0.5 * (w_field[0, 1, 1] + w_field[1, 1, 1])) ** 2)
    outlet_V = np.sqrt(
        (0.5 * (u_field[-1, 1, 1] + u_field[-2, 1, 1])) ** 2 + (0.5 * (v_field[-1, 1, 1] + v_field[-2, 1, 1])) ** 2 + (
                    0.5 * (w_field[-1, 1, 1] + w_field[-2, 1, 1])) ** 2)
    inlet_Ma = inlet_V / (0.5 * (sos_field[0, 1, 1] + sos_field[1, 1, 1]))
    outlet_Ma = outlet_V / (0.5 * (sos_field[-1, 1, 1] + sos_field[-2, 1, 1]))
    inlet_data = ["Inlet south",  physical_plane[1,1,1,0], physical_plane[1,1,1,1],
                  0.5 * (P_field[0, 1, 1] + P_field[1, 1, 1]) / 1e5,
                  0.5 * (T_field[0, 1, 1] + T_field[1, 1, 1]), 0.5 * (rho_field[0, 1, 1] + rho_field[1, 1, 1]),
                  0.5 * (u_field[0, 1, 1] + u_field[1, 1, 1]), 0.5 * (v_field[0, 1, 1] + v_field[1, 1, 1]),
                  0.5 * (w_field[0, 1, 1] + w_field[1, 1, 1]), 0.5 * (sos_field[0, 1, 1] + sos_field[1, 1, 1]),
                  inlet_Ma]
    outlet_data = ["Outlet south",  physical_plane[-1,1,1,0], physical_plane[-1,1,1,1], 
                   0.5 * (P_field[-1, 1, 1] + P_field[-2, 1, 1]) / 1e5,
                   0.5 * (T_field[-1, 1, 1] + T_field[-2, 1, 1]), 0.5 * (rho_field[-1, 1, 1] + rho_field[-2, 1, 1]),
                   0.5 * (u_field[-1, 1, 1] + u_field[-2, 1, 1]), 0.5 * (v_field[-1, 1, 1] + v_field[-2, 1, 1]),
                   0.5 * (w_field[-1, 1, 1] + w_field[-2, 1, 1]), 0.5 * (sos_field[-1, 1, 1] + sos_field[-2, 1, 1]),
                   outlet_Ma]
    outlet_data_2 = ["Outlet north",  physical_plane[-1,-2,1,0], physical_plane[-1,-2,1,1], 
                   0.5 * (P_field[-1, -2, 1] + P_field[-2, -2, 1]) / 1e5,
                   0.5 * (T_field[-1, -2, 1] + T_field[-2, -2, 1]), 0.5 * (rho_field[-1, -2, 1] + rho_field[-2, -2, 1]),
                   0.5 * (u_field[-1, -2, 1] + u_field[-2, -2, 1]), 0.5 * (v_field[-1, -2, 1] + v_field[-2, -2, 1]),
                   0.5 * (w_field[-1, -2, 1] + w_field[-2, -2, 1]), 0.5 * (sos_field[-1, -2, 1] + sos_field[-2, -2, 1]),
                   outlet_Ma]
    inlet_data_2 = ["Inlet north",  physical_plane[1,-2,1,0], physical_plane[1,-2,1,1], 
                   P_field[1,-2,1] / 1e5,
                   T_field[1, -2, 1], rho_field[1, -2, 1],
                   u_field[1, -2, 1], v_field[1, -2, 1],
                   w_field[1, -2, 1], sos_field[1, -2, 1], 
                   np.sqrt(u_field[1,-2,1]**2+v_field[1,-2,1]**2+w_field[1,-2,1]**2)/sos_field[1,-2,1]]
    display_data = np.array([inlet_data, inlet_data_2, outlet_data_2, outlet_data])
    print('Time iteration: ' + str(time_iter) + ' | time-step = ' + str(delta_t) + ' [s] | time = ' + str(time) + ' [s]')
    for row in display_data:
        print(*row)
    print(' ')

    ### Output data to file
    if (time_iter % output_iter == 0):
        data_output(time, time_iter, rho_field, u_field, v_field, w_field, E_field, s_field, P_field, T_field,
                    sos_field, computational_plane)
        
        # plt.figure()
        # plt.scatter(physical_plane[1:num_grid_x,1:num_grid_y,1,0],physical_plane[1:num_grid_x,1:num_grid_y,1,1],c=P_field[1:num_grid_x,1:num_grid_y,1],s=2)
        # plt.colorbar()
        # plt.show()
        
   ### Runge-Kutta sub-steps
    for rk in range( 0, rk_order ):

        ### Calculate transport coefficients
        calculate_transport_coefficients( mu_field, kappa_field, P_field, T_field, rho_field )
        
        ### Calculate inviscid fluxes
        inviscid_fluxes( rho_inv_flux, rhou_inv_flux, rhov_inv_flux, rhow_inv_flux, rhoE_inv_flux, rho_field, u_field, v_field, w_field, E_field, s_field, P_field, P_thermo, T_field, sos_field, computational_plane )
        
        ### Calculate viscous fluxes
        viscous_fluxes( rhou_vis_flux, rhov_vis_flux, rhow_vis_flux, rhoE_vis_flux, work_vis_rhoe_flux, u_field, v_field, w_field, T_field, mu_field, kappa_field, computational_plane )
        
        ### Calculate source terms
        source_terms( f_rhou_field, f_rhov_field, f_rhow_field, f_rhoE_field, rho_field, u_field, v_field, w_field, computational_plane )
        
        ### Sum fluxes & source terms
        sum_fluxes_source_terms( rho_rk_fluxes, rhou_rk_fluxes, rhov_rk_fluxes, rhow_rk_fluxes, rhoE_rk_fluxes, P_rk_fluxes, rho_inv_flux, rhou_inv_flux, rhov_inv_flux, rhow_inv_flux, rhoE_inv_flux, rhou_vis_flux, rhov_vis_flux, rhow_vis_flux, rhoE_vis_flux, work_vis_rhoe_flux, f_rhou_field, f_rhov_field, f_rhow_field, f_rhoE_field, rho_field, u_field, v_field, w_field, P_field, T_field, sos_field, rk, computational_plane )
        
        ### Advance conserved variables in time
        time_integration( rho_field,  rho_0_field,  delta_t, rho_rk_fluxes,  rk )
        time_integration( rhou_field, rhou_0_field, delta_t, rhou_rk_fluxes, rk )
        time_integration( rhov_field, rhov_0_field, delta_t, rhov_rk_fluxes, rk )
        time_integration( rhow_field, rhow_0_field, delta_t, rhow_rk_fluxes, rk )
        time_integration( rhoE_field, rhoE_0_field, delta_t, rhoE_rk_fluxes, rk )
        if( transport_pressure_scheme ):
            time_integration( P_field, P_0_field, delta_t, P_rk_fluxes, rk )

        ### Update primitive variables from conserved variables
        update_primitive( u_field, rhou_field, rho_field )
        update_primitive( v_field, rhov_field, rho_field )
        update_primitive( w_field, rhow_field, rho_field )
        update_primitive( E_field, rhoE_field, rho_field )
       
        ### Update thermodynamic variables from primitive variables
        thermodynamic_state( rhoE_field, P_field, T_field, rho_field, u_field, v_field, w_field, E_field, s_field )
        if( artificial_compressibility_method ):
            P_thermo  = calculate_volume_averaged_value( P_field, computational_plane )
            alpha_acm = calculate_alpha_acm( P_field, P_thermo, computational_plane )
        calculate_speed_sound( sos_field, rho_field, P_field, P_thermo, T_field )

        ### Update boundaries
        update_boundaries( sos_field, rho_field, rhou_field, rhov_field, rhow_field, rhoE_field, u_field, v_field, w_field, P_field, T_field, computational_plane )
    
    ### Update old fields of conserved variables
    update_field( rho_0_field,  rho_field )  
    update_field( rhou_0_field, rhou_field )  
    update_field( rhov_0_field, rhov_field )  
    update_field( rhow_0_field, rhow_field )  
    update_field( rhoE_0_field, rhoE_field )  
    update_field( P_0_field, P_field )  

    ### Update time and iteration counter
    time += delta_t
    time_iter += 1

    ### Check if simulation is completed (time > final_time)
    if( time >= final_time ):
        break

### Output data to file
data_output(time, time_iter, rho_field, u_field, v_field, w_field, E_field, s_field, P_field, T_field, sos_field, computational_plane)

### Print data output information
print('Data output at time' + ': t = ' + str(time) + ' [s]')

### END SIMULATION
print('RHEA: END SIMULATION')
