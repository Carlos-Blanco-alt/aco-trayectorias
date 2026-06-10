# ===================================================
# SIMULACIÓN DE LA TRAYECTORIA DE LA NAVE
# ===================================================

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# IMPORTAR CONSTANTES

from constantes_parametros import (
    UA, MU, DIA,
    r_tierra, r_marte,
    omega_tierra, omega_marte,
    v_tierra, v_marte,
    T_tierra, T_marte,
    phi_marte)

# Valores para transferencias tipo Hohmann
from constantes_parametros import (
    delta_v_hohmann,
    t_vuelo_hohmann)

# Integrador

def campo(t, u):
    """Campo gravitacional"""
    x, y, vx, vy = u
    r = np.sqrt(x**2 + y**2)
    ax = -MU * x / r**3
    ay = -MU * y / r**3
    return np.array([vx, vy, ax, ay])

def integrate_to(estado_inicial, t_vuelo, paso_dias=0.25):
    """Integración de la trayectoria"""
    step_seconds = paso_dias * DIA
    n_intervalos = int(round(t_vuelo / step_seconds))
    t_eval = np.linspace(0, t_vuelo, n_intervalos + 1)
    
    sol = solve_ivp(
        campo,
        [0, t_vuelo],
        estado_inicial,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-6)
    return sol.y[0], sol.y[1], sol.y[2], sol.y[3]

# Funciones para los estados de los planetas

def estado_tierra(t):
    """Posición y velocidad de la Tierra en órbita circular"""
    theta = omega_tierra * t
    
    x = r_tierra * np.cos(theta)
    y = r_tierra * np.sin(theta)
    vx = -r_tierra * omega_tierra * np.sin(theta)
    vy = r_tierra * omega_tierra * np.cos(theta)
    
    return np.array([x, y]), np.array([vx, vy])

def estado_marte(t):
    """Posición y velocidad de Marte en órbita circular con fase inicial"""
    theta = omega_marte * t + phi_marte
    
    x = r_marte * np.cos(theta)
    y = r_marte * np.sin(theta)
    vx = -r_marte * omega_marte * np.sin(theta)
    vy = r_marte * omega_marte * np.cos(theta)
    
    return np.array([x, y]), np.array([vx, vy])

# PARÁMETROS DE LA MISIÓN 

# Fecha de lanzamiento (segundos)
t_L = 19* DIA  # segundos

# Cambio de velocidad (km/s)
#delta_v = delta_v_hohmann# km/s
delta_v = 3.67

# Dirección del cambio de velocidad (grados)
theta_deg = -33.9231 # grados
theta = theta_deg * np.pi / 180  # radianes

# Paso de integración (días)
paso_dias = 0.5  # días

# Tiempo de vuelo (días)
t_vuelo_dias = 4* 186.55 # días
t_vuelo = t_vuelo_dias * DIA  # segundos

# Tiempo de vuelo (días) Hohmann
#t_vuelo_dias = t_vuelo_hohmann / DIA  # días
#t_vuelo = t_vuelo_hohmann  # segundos 

# Cálculo del estado inicial de la nave

print("\n" + "="*70)
print(" PARÁMETROS DE LA SIMULACIÓN")
print("="*70)
print(f"t_L = {t_L/DIA:.6f} días")
print(f"delta_v = {delta_v:.6f} km/s")
print(f"theta = {theta_deg:.6f}°")
print(f"t_vuelo = {t_vuelo_dias:.6f} días")
print("="*70)

# Estado de la Tierra en el lanzamiento
pos_tierra_ini, vel_tierra_ini = estado_tierra(t_L)

# Base local
t_hat = vel_tierra_ini / np.linalg.norm(vel_tierra_ini)  # Tangente
n_hat = np.array([-t_hat[1], t_hat[0]])                  # Normal

# Dirección del cambio de velocidad
vec_dir = np.cos(theta) * t_hat + np.sin(theta) * n_hat

# Cambio de velocidad
cambio_vel = delta_v * vec_dir

# Estado inicial de la nave
estado_inicial = np.array([
    pos_tierra_ini[0],
    pos_tierra_ini[1],
    vel_tierra_ini[0] + cambio_vel[0],
    vel_tierra_ini[1] + cambio_vel[1]
])


# Salida de información

print("\n" + "="*70)
print(" SIMULACIÓN - SALIDA")
print("="*70)

print(f"\nEstado de la Tierra en t=0:")
print(f"  Posición: ({pos_tierra_ini[0]/UA:.6f}, {pos_tierra_ini[1]/UA:.6f}) UA")
print(f"  Velocidad: ({vel_tierra_ini[0]:.6f}, {vel_tierra_ini[1]:.6f}) km/s")
print(f"  Rapidez: {np.linalg.norm(vel_tierra_ini):.6f} km/s")

print(f"\nEstado inicial de la nave:")
print(f"  Posición: ({estado_inicial[0]/UA:.6f}, {estado_inicial[1]/UA:.6f}) UA")
print(f"  Velocidad: ({estado_inicial[2]:.6f}, {estado_inicial[3]:.6f}) km/s")
print(f"  Rapidez total: {np.linalg.norm(estado_inicial[2:4]):.6f} km/s")
print(f"  Delta-V experimentando: {delta_v:.6f} km/s")
print(f"  Dirección del cambio de velocidad: {theta_deg:.6f}°")


# Integración 

print("\n" + "="*70)
print(" SIMULACIÓN - LLEGADA")
print("="*70)
x_nave, y_nave, vx_nave, vy_nave = integrate_to(estado_inicial, t_vuelo, paso_dias)
print(f"\nIntegrando durante {t_vuelo/DIA:.6f} días, con paso cada {paso_dias} día(s).")

# Posiciones finales
pos_tierra_fin, vel_tierra_fin = estado_tierra(t_L + t_vuelo)
pos_marte_fin, vel_marte_fin = estado_marte(t_L + t_vuelo)

print(f"\nTierra en t={t_vuelo/DIA:.6f} días después del lanzamiento:")
print(f"  Posición: ({pos_tierra_fin[0]/UA:.6f}, {pos_tierra_fin[1]/UA:.6f}) UA")
print(f"  Velocidad: ({vel_tierra_fin[0]:.6f}, {vel_tierra_fin[1]:.6f}) km/s")

print(f"\nMarte en t={t_vuelo/DIA:.6f} días después del lanzamiento:")
print(f"  Posición: ({pos_marte_fin[0]/UA:.6f}, {pos_marte_fin[1]/UA:.6f}) UA")
print(f"  Velocidad: ({vel_marte_fin[0]:.6f}, {vel_marte_fin[1]:.6f}) km/s")

print(f"\nNave en t={t_vuelo/DIA:.6f} días después del lanzamiento:")
print(f"  Posición: ({x_nave[-1]/UA:.6f}, {y_nave[-1]/UA:.6f}) UA")
print(f"  Velocidad: ({vx_nave[-1]:.6f}, {vy_nave[-1]:.6f}) km/s")

distancia_marte = np.linalg.norm([x_nave[-1] - pos_marte_fin[0], y_nave[-1] - pos_marte_fin[1]])
print(f"\n*** DISTANCIA NAVE - MARTE: {distancia_marte:.3f} km ***")

# Excentricidad
x, y = estado_inicial[0], estado_inicial[1]
vx, vy = estado_inicial[2], estado_inicial[3]
r = np.sqrt(x**2 + y**2)
h = x * vy - y * vx
e_x = (vy * h)/MU - x/r
e_y = (-vx * h)/MU - y/r
e = np.sqrt(e_x**2 + e_y**2)

print(f"\nExcentricidad de la órbita: {e:.6f}")

# Órbitas para la gráfica

t_tierra_orb = np.linspace(0, T_tierra, 500)
pos_tierra_orb = np.array([estado_tierra(t)[0] for t in t_tierra_orb])

t_marte_orb = np.linspace(0, T_marte, 500)
pos_marte_orb = np.array([estado_marte(t)[0] for t in t_marte_orb])

pos_marte_ini, _ = estado_marte(t_L)

# Gráfica

plt.figure(figsize=(10, 8))

plt.plot(pos_tierra_orb[:,0]/UA, pos_tierra_orb[:,1]/UA, 'b-', 
        linewidth=1.2, alpha=0.6, label='Órbita Tierra')
plt.plot(pos_marte_orb[:,0]/UA, pos_marte_orb[:,1]/UA, 'r-',
        linewidth=1.2, alpha=0.6, label='Órbita Marte')

plt.plot(x_nave/UA, y_nave/UA, 'k-', linewidth=1.8, label='Trayectoria nave')

plt.scatter(pos_tierra_ini[0]/UA, pos_tierra_ini[1]/UA, 
            color='blue', s=120, marker='o', edgecolors='darkblue', linewidth=1.5,
            label='Tierra (Salida)', zorder=5)
plt.scatter(pos_tierra_fin[0]/UA, pos_tierra_fin[1]/UA, 
            color='lightblue', s=120, marker='o', edgecolors='blue', linewidth=1.5,
            label='Tierra (Llegada)', zorder=5)

plt.scatter(pos_marte_ini[0]/UA, pos_marte_ini[1]/UA, 
            color='darkred', s=120, marker='o', edgecolors='red', linewidth=1.5,
            label='Marte (Salida)', zorder=5)
plt.scatter(pos_marte_fin[0]/UA, pos_marte_fin[1]/UA, 
            color='salmon', s=120, marker='o', edgecolors='red', linewidth=1.5,
            label='Marte (Llegada)', zorder=5)

plt.scatter(x_nave[0]/UA, y_nave[0]/UA, color='white', s=20, marker='o',
            edgecolors='red', linewidth=1.2, label='Nave (Salida)', zorder=5)
plt.scatter(x_nave[-1]/UA, y_nave[-1]/UA, color='black', s=20, marker='o',
            edgecolors='black', linewidth=1.2, label='Nave (Llegada)', zorder=5)

plt.scatter(0, 0, color='gold', s=250, edgecolors='darkorange', linewidth=1.5, label='Sol', zorder=4)

plt.title("Trayectoria Tierra-Marte", fontsize=13, fontweight='bold')
plt.xlabel("x (UA)", fontsize=11)
plt.ylabel("y (UA)", fontsize=11)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axis("equal")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
plt.tight_layout()

# Guardar imagen de cada ejecución

"""
base_nombre = 'simulacion'
contador = 1
while os.path.exists(f'{base_nombre}_{contador}.png'):
    contador += 1
plt.savefig(f'{base_nombre}_{contador}.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Imagen guardada como: simulacion_{contador}.png")
"""
plt.show()