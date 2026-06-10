# CONSTANTES DEL SISTEMA SOLAR
import numpy as np

UA = 149597870.7 # km, distancia media Tierra-Sol
MU =  1.32712440041279419e11  # km^3/s^2, parámetro gravitacional del Sol
DIA =  24 * 3600        # segundos por día
# ÓRBITAS PLANETARIAS (circulares)

# Radios orbitales
r_tierra = 1.00000261  * UA  # km
r_marte =1.52371034 * UA  # km

# Velocidades angulares
omega_tierra = np.sqrt(MU / r_tierra**3)   # rad/s
omega_marte  = np.sqrt(MU / r_marte**3)    # rad/s

# Períodos orbitales
T_tierra = 2 * np.pi / omega_tierra   # segundos (365.25 días)
T_marte  = 2 * np.pi / omega_marte    # segundos (687 días)

# Velocidades orbitales (tangenciales)
v_tierra = omega_tierra * r_tierra   # km/s
v_marte  = omega_marte * r_marte     # km/s

# Fase inicial de Marte (44.35° en radianes)
phi_marte= 0.77405352  # rad

# Valores óptimos para transferencia de Hohmann
a_hohmann = (r_tierra + r_marte) / 2
delta_v_hohmann = np.sqrt(MU * (2 / r_tierra - 1 / a_hohmann)) - v_tierra
t_vuelo_hohmann = np.pi * np.sqrt(a_hohmann**3 / MU)
phi_marte = np.pi - omega_marte * t_vuelo_hohmann