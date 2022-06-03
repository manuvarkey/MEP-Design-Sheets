#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  
#  Copyright 2022 Manu Varkey <manuvarkey@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

from math import sqrt, log, pi, ceil, floor
import numpy as np

## Resistance functions
    
def resistance_plate(rho, A, h):
    """ As per ENA EREC S34
        rho: resistivity (ohm-m)
        A: plate area (m^2)
        h = burrial depth
    """
    r = sqrt(A/pi)
    return rho/(8*r)*(1 + r/(2.5*h+r))

def resistance_pipe(rho, L, d):
    """ As per IEEE 80
        rho: resistivity (ohm-m)
        L: length of pipe (m)
        d: dia of pipe (m)
    """
    return rho/(2*pi*L)*(log(8*L/d) - 1)

def resistance_strip(rho, L, w, t):
    """ Resistance calculation as per IS-3043
        rho: resistivity (ohm-m)
        L: length of strip (m)
        w: depth of burial (m)
        t: thickness of strip/ 2 x dia for pipe  (m)
    """
    return rho/(2*pi*L)*log(2*L**2/(w*t))
        
def resistance_grid(rho, A, L):
    """ Return grid resistance as per ENA EREC S34
        rho: resistivity (ohm-m)
        A: area occupied by the ground grid (m^2)
        L: total buried length of conductors (m)
    """
    return (rho/4*sqrt(pi/A) + rho/L)

def resistance_grid_with_rods(rho, A, L_E, L_R, N, S, d, w):
    """ Return grid resistance with earth rods as per ENA EREC S34
        rho: resistivity (ohm-m)
        A: area occupied by the ground grid (m^2)
        L_E: length of horizontal electrode (m)
        L_R: length of vertical rod electrode (m)
        N: total number of rods
        s: seperation between rods
        d: diameter of rod electrode
        w: width of tape
    """
    data = np.array([[0,0], [4,2.6], [8,4.3], [12,5.3], [16,6], [20,6.5], [24,6.8], [40,8], [64,9]])
    fit = np.polyfit(data[:,0], data[:,1], 5)
    poly = np.poly1d(fit)
    k = poly(N)  # Use polynomial fit to get k

    r = sqrt(A/pi)
    R1 = rho/(4*r) + rho/L_E
    
    R_R = rho/(2*pi*L_R)*(log(8*L_R/d)-1)
    alpha = rho/(2*pi*R_R*S)
    R2 = R_R*(1+k*alpha)/N
    
    b = w/pi
    R12 = R1 - rho/(pi*L_E)*(log(L_R/b)-1)
    
    R_E = (R1*R2 - R12**2)/(R1 + R2 - 2*R12)
    return R_E

def resistance_parallel(*res):
    """ Return effective of parallel resistances """
    R = 0
    for r in res:
        R += r**-1
    return R**-1


## Touch & step potential fucntions

def e_step_70(rho, rho_s, h_s, t_s):
    """ Step potential calculation as per IEEE 80
        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duratioin (s)
    """
    C_s = 1 - 0.09*(1-rho/rho_s)/(2*h_s+0.09)
    E_step_70 = (1000 + 6*C_s*rho_s)*0.157/sqrt(t_s)
    return E_step_70
    
def e_touch_70(rho, rho_s, h_s, t_s):
    """ Touch potential calculation as per IEEE 80
        rho: resistivity (ohm-m)
        rho_s: resistivity of surface layer (ohm-m)
        h_s = thickness of surface layer (m)
        t_s = shock duratioin (s)
    """
    C_s = 1 - 0.09*(1-rho/rho_s)/(2*h_s+0.09)
    E_touch_70 = (1000 + 1.5*C_s*rho_s)*0.157/sqrt(t_s)
    return E_touch_70
    
def e_mesh_step_grid(rho, Lx, Ly, Lr, Nx, Ny, Nr, d, h, Ig):
    """ Touch potential calculation as per IEEE 80
        
        Parameters:
            rho: resistivity (ohm-m)
            Lx: maximum length of the grid in the x direction (m)
            Ly: maximum length of the grid in the y direction (m)
            Lr: average earth rod length (m)
            Nx: number of grid conductors in x direction
            Ny: number of grid conductors in y direction
            Nr: number of earth rods
            d: diameter of the earth conductors (m)
            h: grid burial depth (m)
        Return:
            Em: mesh voltage (V)
            Es: step voltage (V)
    """
    
    # Calculate geometric parameters
    A = Lx*Ly  # Area of grid
    Lc = Lx*Nx + Ly*Ny  # Total length of horizontal conductor in the grid (m)
    L_R = Lr*Nr  # Total length of all earth rods (m)
    Lp = (Lx + Ly)*2  # Length of perimeter conductor (m)
    D = ( (Lx/(Ny-1)) + (Ly/(Nx-1)) )/2  # Spacing between parallel conductors in the mesh (m)
    h0 = 1  # Grid reference depth (m)
    
    # Grid shape components
    n_a = 2*Lc/Lp
    n_b = sqrt(Lp/(4*sqrt(A)))
    n_c = (Lx*Ly/A)**(0.7*A/(Lx*Ly))
    n_d = 1
    n = n_a*n_b*n_c*n_d
    
    Kh = sqrt(1+h/h0)
    if Nr:
        Kii = 1
    else:
        Kii = 1/(2*n)**(2/n)
    
    Ki = 0.644 + 0.148*n
    Km = 1/(2*pi)*( log(D**2/(16*h*d) + (D+2*h)**2/(8*D*d) - h/(4*d)) + Kii/Kh*log(8/(pi*(2*n-1))) )
    Ks = 1/(pi)*( 1/(2*h) + 1/(D+h) + 1/D*(1-0.5**(n-2)) )
    
    Em = rho*Km*Ki*Ig/( Lc + (1.55 + 1.22*(Lr/sqrt(Lx**2+Ly**2)))*L_R )
    Es = rho*Ks*Ki*Ig/(0.75*Lc + 0.85*L_R)
    
    return Em, Es


## Fault current functions

def fault_current(Un, c, Z):
    """ Fault current calculation as per IEC-60909
        Un: Nominal line-line voltage (V)
        c: Voltage factor
        Z: Complex impedence
    """
    return c*Un/(sqrt(3)*abs(Z))

def max_current_density(rho, t):
    """ Electrode maximum current density calculation as per IS-3043
        rho: resistivity (ohm-m)
        t: Loading time
        return (A/m2)
    """
    return (7.57e3/sqrt(rho*t))

def current_ratio(rho, R, C, Un, A, L):
    """ Current ratio by C factor method as per ENA EREC S34
        rho: resistivity (ohm-m)
        R: combined sorce and destination resistance
        C: C factor
        Un: System voltage (kV)
        A: Cross sectional area (mm2)
        L: Cable length (km)
    """
    return C/(A+9*Un) / sqrt((C/(A+9*Un) + R/L)**2 + 0.6*(rho/A*Un)**0.1)
    
    
