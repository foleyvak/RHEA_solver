import os
import numpy as np
import math

num_grid_x = 30
num_grid_y = 15
num_grid_z = 1

x_0 = 0.0
y_0 = 0.0
z_0 = 0.0

L_x = 1.0
L_y_var = 1.0
L_z = 0.01

A_x = 0.0
A_y = 0.0
A_z = 0.0

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
                # x_val = grid[i][j][k][0]
                # L_y_var = nz_cont.nozzle_top_contour(x_val, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
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
                    
    return grid