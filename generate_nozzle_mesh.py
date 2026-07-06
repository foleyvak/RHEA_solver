import os
import numpy as np
import math
import matplotlib.pyplot as plt

############# GEOMETRY CONFIGURATION #############
num_grid_x   = 64                                   # Number of grid points in x-direction
num_grid_y   = 64                                    # Number of grid points in y-direction
num_grid_z   = 1                                    # Number of grid points in z-direction

A_x           = -1.0                                     # Grid stretching factor in x-direction
A_y           = -1.0                                     # Grid stretching factor in y-direction
A_z           = 0.0                                     # Grid stretching factor in z-direction

x_0           = 0.0                                     # Domain origin in x-direction [m]
y_0           = 0.0                                     # Domain origin in y-direction [m]
z_0           = 0.0                                     # Domain origin in z-direction [m]

r_t           = 0.8e-3         	                        # Nozzle throat radius (for nozzle geometries)
r_c           = 2.0e-3         	                        # Nozzle chamber radius (for nozzle geometries)
R1_rt         = 10.0         	                        # Convergent-throat arc ratio as (R1/rt)
R2_R1         = 3.0                                     # Chamber-convergent arc ratio as (R2/R1)
Rexp_rt       = 30.0                                    # Expansion arc ratio as (Rexp/rt)
theta         = 10.0                                     # Convergent segment inclination angle [deg]
alpha         = 3.0                                     # Conical nozzle half-angle [deg]
L_N           = 50.0e-3                                   # Conical section length [m]
L_c           = 3.0e-3                                    # Chamber section length [m]

RHEA_python_mesh    = True                                 # Set to True to generate a basic RHEA python mesh
advanced_mesh       = False                                # Set to True to generate an advanced mesh with stretching and refinement
####################################################

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

L             = 1.0					                    # Cavity size [m]
#L_x = x_t + L_N + L_c
L_x = 1.0*L
L_z           = 0.01*L         	                        # Size of domain in z-direction
L_y = np.ones(num_grid_x+2)                            # Initialize L_y array for each x-grid point
L_y *= 1.0*L

grid = np.zeros((num_grid_x+2, num_grid_y+2, num_grid_z+2, 3)) # Initialize grid array
physical_grid = np.zeros((num_grid_x+2, num_grid_y+2, num_grid_z+2, 3)) # Initialize physical grid array

for i in range(0, num_grid_x+2):
    for j in range(0, num_grid_y+2):
        for k in range(0, num_grid_z+2):
            # Evenly-spaced (dummy variable) cell centroids
            eta_x = ( i - 0.5 )/num_grid_x
            eta_y = ( j - 0.5 )/num_grid_y
            eta_z = ( k - 0.5 )/num_grid_z
            # Unevenly-spaced (stretched) cell centroids
            grid[i][j][k][0] = x_0 + L_x*eta_x + A_x*( 0.5*L_x - L_x*eta_x )*( 1.0 - eta_x )*eta_x

            x = grid[i][j][k][0]  # Current x-coordinate for determining L_y

           # if x <= x_c:
           #     L_y[i] = r_c        
           # elif x_c < x <= x2:
           #     # Arc 2 (Centered at x_c)
           #     L_y[i] = r_c - R2 * (1 - np.sqrt(1 - ((x - x_c) / R2)**2))                    
           # elif x2 < x <= x1:
           #     # Straight convergent section
           #     L_y[i] = r1 - (x - x1) * math.tan(theta_rad)                    
           # elif x1 < x <= x_t:
           #     # Arc 1 (Centered at x_t)
           #     L_y[i] = r_t + R1 * (1 - np.sqrt(1 - ((x - x_t) / R1)**2))                    
           # elif x_t < x <= x_exp:
           #     # Expansion Arc (Centered at x_t)
           #     L_y[i] = r_t + Rexp * (1 - np.sqrt(1 - ((x - x_t) / Rexp)**2))                  
           # else:
           #     # Straight divergent section
           #     L_y[i] = r_exp + (x - x_exp) * math.tan(alpha_rad)

            grid[i][j][k][1] = y_0 + L_y[i]*eta_y + A_y*( 0.5*L_y[i] - L_y[i]*eta_y )*( 1.0 - eta_y )*eta_y
            grid[i][j][k][2] = z_0 + L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z

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

            physical_grid[i][j][k] = grid[i][j][k]     # For grid visualization
            grid[i][j][k][1] /= L_y[i]                 # coordinate transformation to computational space (xi, eta, zeta) -- y is normalized by local L_y. Change if needed.
                
# ############## GRID VISUALIZATION ##############
fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=( 12, 6 ) )
ax1.scatter(physical_grid[:,:,1,0], physical_grid[:,:,1,1], s=2, c='blue', marker='o')
ax2.scatter(grid[:,:,1,0], grid[:,:,1,1], s=2, c='red', marker='o')
ax1.set_title('Physical Grid')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax2.set_title('Computational Grid')
ax2.set_xlabel('xi')
ax2.set_ylabel('eta')
plt.show()
# exit()
# ################################################

file_name = 'external_mesh_file.txt'
data_file_out = open( file_name, 'wt' )

data_file_out.write( '# INSTRUCTIONS:								                    ... do not remove this line!\n')
data_file_out.write( '# 1.- Set mesh origin (x_0, y_0, z_0),					        ... do not remove this line!\n')
data_file_out.write( '#         mesh size (L_x, L_y, L_z),					            ... do not remove this line!\n')
data_file_out.write( '#         number grid points (num_grid_x, num_grid_y, num_grid_z)	... do not remove this line!\n')
data_file_out.write( '#     in configuration_file.yaml					                ... do not remove this line!\n')
data_file_out.write( "# 2.- Activate external mesh to 'TRUE'					        ... do not remove this line!\n")
data_file_out.write( '#     and set name of external mesh file				            ... do not remove this line!\n')
data_file_out.write( '#     in configuration_file.yaml					                ... do not remove this line!\n')
data_file_out.write( '# 3.- Provide below list of x-, y- and z-direction gridpoints		... do not remove this line!\n')
data_file_out.write( '# x-direction points:							                    ... do not remove this line!\n')

for i in range(1, num_grid_x+1):
    data_file_out.write( str(physical_grid[i][1][1][0]) + '\n')

data_file_out.write( '# y-direction points:							                    ... do not remove this line!\n')

for j in range(1, num_grid_y+1):
    data_file_out.write(str(physical_grid[1][j][1][1]) + '\n')

data_file_out.write( '# z-direction points:							                    ... do not remove this line!\n')

for k in range(1, num_grid_z+1):
    data_file_out.write( str(physical_grid[1][1][k][2]) + '\n')
