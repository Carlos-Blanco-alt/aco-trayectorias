
# ==================================================
# INTEGRADOR DE LA TRAYECTORIA
# ==================================================
import numpy as np
from scipy.integrate import solve_ivp
from constantes_parametros import DIA, MU

def campo(t, u):
    """Campo vectorial para la integración numérica de la trayectoria.
    
    t: tiempo en segundos
    u: [x, y, vx, vy] con x,y en km, v en km/s
    """
    x, y, vx, vy = u
    r = np.sqrt(x**2 + y**2)
    ax = -MU * x / r**3
    ay = -MU * y / r**3
    return np.array([vx, vy, ax, ay])


def integrador_eventos(est_ini, t_vuelo,paso, dist_obj, fact_diver, marte, t_L):
    """Integra las ecuaciones del movimiento para encontrar la trayectoria de la nave.
    
    Parámetros:
        est_ini: estado inicial [x0, y0, vx0, vy0] (km, km/s)
        t_vuelo: tiempo máximo de vuelo (SEGUNDOS)
        dist_obj: distancia objetivo para considerar éxito (km)
        fact_diver: factor de divergencia (>1)
        marte: instancia de la clase Marte
        t_L: instante de lanzamiento (SEGUNDOS)
        paso: paso de integracion en seg 
    
    Retorna:
        x_list, y_list, t_list, suceso, razon, tiempo_final, mejor_distancia, aptitud
    """
    # Validaciones
    if t_vuelo <= 0:
        raise ValueError("t_vuelo debe ser positivo")
    if dist_obj <= 0:
        raise ValueError("dist_obj debe ser positivo")
    if fact_diver <= 1:
        raise ValueError("fact_diver debe ser > 1")
    
    # t_L y t_vuelo ya están en segundos
    x0, y0 = est_ini[0], est_ini[1]
    x_m0, y_m0 = marte.posicion(t_L)
    dist_inicial = np.hypot(x0 - x_m0, y0 - y_m0)
    d_max = fact_diver * dist_inicial

    # Evento de éxito
    def evento_exito(t, u):
        x_m, y_m = marte.posicion(t_L + t)
        return np.hypot(u[0] - x_m, u[1] - y_m) - dist_obj
    evento_exito.terminal = True
    evento_exito.direction = -1

    # Evento de divergencia
    def evento_divergencia(t, u):
        x_m, y_m = marte.posicion(t_L + t)
        dist = np.hypot(u[0] - x_m, u[1] - y_m)
        return dist - d_max
    evento_divergencia.terminal = True
    evento_divergencia.direction = 0

    # Integración
    sol = solve_ivp(
        campo,
        [0, t_vuelo],
        est_ini,
        events=[evento_exito, evento_divergencia],
        rtol=1e-6,
        atol=1e-6,
        max_step=paso  # paso en segundos
    )

    # Resultados (tiempos en días para salida)
    x_list = sol.y[0]
    y_list = sol.y[1]
    t_list = sol.t / DIA

    # Evaluar éxito
    suceso = False
    razon = "tiempo_max"
    tiempo_final = t_list[-1]

    if len(sol.t_events[0]) > 0:
        suceso = True
        razon = "exito"
        tiempo_final = sol.t_events[0][0] / DIA
    elif len(sol.t_events[1]) > 0:
        razon = "divergencia"
        tiempo_final = sol.t_events[1][0] / DIA

    # Mejor distancia
    mejor_distancia = float('inf')
    for i, t_sec in enumerate(sol.t):
        x_m, y_m = marte.posicion(t_L + t_sec)
        dist = np.hypot(x_list[i] - x_m, y_list[i] - y_m)
        if dist < mejor_distancia:
            mejor_distancia = dist

    # Aptitud
    if suceso:
        aptitud = tiempo_final
    else:
        aptitud = mejor_distancia

    return x_list, y_list, t_list, suceso, razon, tiempo_final, mejor_distancia, aptitud