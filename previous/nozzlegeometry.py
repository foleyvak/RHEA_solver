import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def convergent_segment(x, rt, rc, R1_rt, R2_R1, theta):

    R1 = rt*R1_rt
    R2 = R1*R2_R1

    x1 = -R1*math.sin(theta)
    r1 = rt + R1*(1-math.cos(theta))
    r2 = rc + R2 * ( math.cos(theta) - 1 )
    x2 = x1 - (r2 - r1) / math.tan(theta)
    xc = x2 - R2*math.sin(theta)

    if x > x1:                                                                      # FIRST CONVERGENT ARC
        # print("x > x1")
        # print(x)

        L_y = rt + R1*(1-math.cos(math.asin(-x/R1)))                                #
    elif (x <= x1) and (x >= x2):                                                     # CONVERGENT TRANSITION
        # print("x1 > x > x2")
        # print(x)

        L_y = r1 - (x-x1)*math.tan(theta)                               #
    elif (x <= x2) and (x >= xc):                                                                    # SECOND CONVERGENT ARC
        # print("x < x2")
        # print(x, x2, R2)

        L_y = rc - R2*(1-np.sqrt(1-(-(x-xc)/R2)**2))
    else:
        L_y = rc

    return L_y

def conical_nozzle(x, rt, Rexp_rt, alpha):
    Rexp = rt * Rexp_rt

    x_nozzle = Rexp*math.sin(alpha)
    r_nozzle = rt+Rexp*(1-math.cos(alpha))

    if x < x_nozzle:
        L_y = rt + Rexp*(1-np.sqrt(1-(x/Rexp)**2))
    else:
        L_y = r_nozzle+(x-x_nozzle)*math.tan(alpha)

    return L_y

# def bell_nozzle_dualnozzle(x, Rexp2_rt, ContourCoefficients)

# rt = 0.8e-3
# rc = 2.5e-3
# R1_rt = 4.0
# R2_R1 = 2.0
# theta = 30.0

# n_R1 = 100
# n_R2 = 100

# R1 = rt*R1_rt
# R2 = R1*R2_R1

# L = -math.sin(theta*math.pi/180)*(R1+R2)-(rc-rt+(math.cos(theta*math.pi/180)-1)*(R1+R2))/math.tan(theta*math.pi/180)
# x = np.linspace(L,0,100)
# r = np.zeros(100)
# for i in range(0,100):
#     r[i] = convergent_segment(x[i],rt,rc,R1_rt,R2_R1,theta)


#
# dtheta_1 = theta/n_R1
# dtheta_2 = theta/n_R2
#
# x_arc1 = np.zeros(n_R1)
# r_arc1 = np.zeros(n_R1)
#
# for i in range(1, n_R1+1):
#     x_arc1[i-1] = -R1 * math.sin(dtheta_1*math.pi/180 * (i-1))
#     r_arc1[i-1] = rt + R1 * ( 1 - math.cos( dtheta_2*math.pi/180 * (i-1) ) )
#
# x1 = x_arc1[-1]
# r1 = r_arc1[-1]
# r2 = rc + R2 * ( math.cos(theta*math.pi/180 ) - 1 )
# x2 = x1 - (r2 - r1) / math.tan(theta*math.pi/180)
#
# dx = x2 - x1
# dx_i = dx/5
#
# x_linear = np.zeros(4)
# r_linear = np.zeros(4)
#
# for i in range(1,5):
#     x_linear[i-1] = x1 + dx_i * i
#     r_linear[i-1] = r1 - ( x_linear[i-1] - x1 ) * math.tan(theta*math.pi/180)
#
# x_arc2 = np.zeros(n_R2)
# r_arc2 = np.zeros(n_R2)
#
# for i in range(n_R2,0,-1):
#     x_arc2[i-1] = x2 - R2 * math.sin( math.pi/180 * ( theta - dtheta_2 * ( i - 1 )))
#     r_arc2[i-1] = rc - R2 * ( 1 - math.cos( dtheta_2*math.pi/180 * ( i - 1 )))
#
# x_coords = np.concatenate((x_arc2, np.flip(x_linear), np.flip(x_arc1),[0]))
# r_coords = np.concatenate((r_arc2, np.flip(r_linear), np.flip(r_arc1), [rt]))