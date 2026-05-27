import os
import numpy as np
import math
import matplotlib.pyplot as plt

num_grid_x = 32
num_grid_y = 16
num_grid_z = 1
num_sptl_dim = 3

x_0 = 0.0
y_0 = 0.0
z_0 = 0.0

L = 1.0
L_y = 1.0 * L

L_x = 2.0
L_y_var = 1.0
L_z = 0.01

A_x = 0.0
A_y = 0.0
A_z = 0.0

### Define centroids of spatial discretization
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

                ### Define y as y(x)
                # L_y = L - ( grid[i][j][k][0]-L_x) * math.tan(10.0 * math.pi/180.0) # Example geometry -- a wedge with 10 degree angle

                grid[i][j][k][1] = y_0 + L_y*eta_y + A_y*( 0.5*L_y - L_y*eta_y )*( 1.0 - eta_y )*eta_y
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
                    grid[i][j][k][1] = y_0 - ( L_y*eta_y + A_y*( 0.5*L_y - L_y*eta_y )*( 1.0 - eta_y )*eta_y )
                if( grid[i][j][k][1] > ( y_0 + L_y ) ):
                    eta_y = ( num_grid_y - 0.5 )/num_grid_y
                    grid[i][j][k][1] = y_0 + 2.0*L_y - ( L_y*eta_y + A_y*( 0.5*L_y - L_y*eta_y )*( 1.0 - eta_y )*eta_y )
                if( grid[i][j][k][2] < z_0 ):
                    eta_z = ( 1.0 - 0.5 )/num_grid_z
                    grid[i][j][k][2] = z_0 - ( L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z )
                if( grid[i][j][k][2] > ( z_0 + L_z ) ):
                    eta_z = ( num_grid_z - 0.5 )/num_grid_z
                    grid[i][j][k][2] = z_0 + 2.0*L_z - ( L_z*eta_z + A_z*( 0.5*L_z - L_z*eta_z )*( 1.0 - eta_z )*eta_z )

                # physical_grid[i][j][k] = grid[i][j][k]     # For grid visualization
                # grid[i][j][k][1] /= L_y                 # coordinate transformation to computational space (xi, eta, zeta) -- y is normalized by local L_y. Change if needed.

                
    # ############## GRID VISUALIZATION ##############
    # fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=( 12, 6 ) )
    # ax1.scatter(physical_grid[:,:,1,0], physical_grid[:,:,1,1], s=2, c='blue', marker='o')
    # ax2.scatter(grid[:,:,1,0], grid[:,:,1,1], s=2, c='red', marker='o')
    # ax1.set_title('Physical Grid')
    # ax1.set_xlabel('x [m]')
    # ax1.set_ylabel('y [m]')
    # ax2.set_title('Computational Grid')
    # ax2.set_xlabel('xi [m]')
    # ax2.set_ylabel('eta')
    # plt.show()
    # ################################################

    return grid