# ==================================================
# EJECUCIÓN DEL ACO PARA MISIÓN A MARTE
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from constantes_parametros import DIA, UA, T_tierra, T_marte
from dinamicas_orbitales import Tierra, Marte, Nave
from integrador_numerico import integrador_eventos
from optimizacion_hormigas import Grafo, Colonia, Heuristicas

# 1. CONFIGURACIÓN INICIAL


tierra = Tierra()
marte = Marte()

t_vuelo_dias = 260  # días, tiempo máximo de vuelo para evaluar cada solución
t_vuelo_seg = t_vuelo_dias * DIA
distancia_objetivo = 10000 # km
factor_divergencia = 2 # >1
paso = 0.5 * DIA  # paso de integracion


# 2. CONFIGURACIÓN DEL ACO

rangos = {
    't_L': [0, 780 * DIA],  # segundos
    'delta_v': [2.95, 5.65],  # km/s
    'theta': [-np.pi/2, np.pi/4]}  # radianes

resolucion = {'t_L': 781,
            'delta_v': 136,
            'theta': 781}

num_hormigas = 400
num_iteraciones = 200
alpha_aco = 0.6
beta_aco =0.4
rho = 0.5
Q = 10

# 3. INICIALIZACIÓN


grafo = Grafo(rangos, resolucion)
colonia = Colonia(grafo, num_hormigas)

# Calcular heurísticas

"""

heuristicas_obj = None
heuristicas = None

"""

heuristicas_obj = Heuristicas(tierra, marte)
heuristicas = heuristicas_obj.construir_heuristicas(
    grafo.t_L_opciones,
    grafo.dv_opciones,
    grafo.theta_opciones
)


mejor_solucion_global = None


# 4. BUCLE PRINCIPAL

for iteracion in range(num_iteraciones):
    print(f"\n--- Iteración {iteracion + 1} ---")

    if iteracion == 0:
        hormigas = colonia.construir_soluciones(aleatorio=True)
    else:
        hormigas = colonia.construir_soluciones(
            alpha_aco,
            beta_aco,
            heuristicas,
            aleatorio=False
        )

    print("  Primeras 5 hormigas:")
    for i, h in enumerate(hormigas[:5]):
        theta_deg = np.degrees(h.theta)
        t_L_dias = h.t_L / DIA

        print(
            f"    {i+1}: t_L={t_L_dias:.4f} días, "
            f"dv={h.delta_v:.4f} km/s, "
            f"theta={theta_deg:.4f}°"
        )

    resultados_iter = []
    mejor_solucion_iter = None
    exitos = 0 

    for h in hormigas:

        t_L_seg = h.t_L
        nave = Nave(tierra, t_L_seg, h.delta_v, h.theta)
        
        estado_ini = nave.obtener_estado()

        x_list, y_list, t_list, suceso, razon, tiempo_final, mejor_distancia, aptitud = integrador_eventos(
            estado_ini,
            t_vuelo_seg,
            paso,
            distancia_objetivo,
            factor_divergencia,
            marte,
            t_L_seg
        )

        if suceso: 
            exitos += 1

        resultados_iter.append({
            'idx_t_L': h.idx_t_L,
            'idx_dv': h.idx_dv,
            'idx_theta': h.idx_theta,
            'aptitud': aptitud
        })

        # Crear solución actual
        sol_actual = {
            't_L': h.t_L,
            'delta_v': h.delta_v,
            'theta': h.theta,
            'theta_deg': np.degrees(h.theta),
            'suceso': suceso,
            'tiempo_final': tiempo_final,
            'mejor_distancia': mejor_distancia,
            'x_list': x_list,
            'y_list': y_list
        }

        # Guardar mejor de la iteración
        if mejor_solucion_iter is None:
            mejor_solucion_iter = sol_actual

        elif suceso and not mejor_solucion_iter['suceso']:
            mejor_solucion_iter = sol_actual

        elif suceso == mejor_solucion_iter['suceso']:

            if suceso:
                if tiempo_final < mejor_solucion_iter['tiempo_final']:
                    mejor_solucion_iter = sol_actual
            else:
                if mejor_distancia < mejor_solucion_iter['mejor_distancia']:
                    mejor_solucion_iter = sol_actual

    print(f"  Éxitos: {exitos}/{len(hormigas)} ({100*exitos/len(hormigas):.1f}%)")

    if mejor_solucion_iter:
        t_L_dias = mejor_solucion_iter['t_L'] / DIA

        print(
            f"  Mejor: t_L={t_L_dias:.4f} días, "
            f"dv={mejor_solucion_iter['delta_v']:.4f} km/s, "
            f"theta={mejor_solucion_iter['theta_deg']:.4f}°"
        )

        if mejor_solucion_iter['suceso']:
            print(f"  Tiempo: {mejor_solucion_iter['tiempo_final']:.2f} días")
        else:
            print(
                f"  Distancia: "
                f"{mejor_solucion_iter['mejor_distancia']:.2f} km"
            )

    # Actualizar mejor global
    if mejor_solucion_iter:

        if mejor_solucion_global is None:
            mejor_solucion_global = mejor_solucion_iter

        elif mejor_solucion_iter['suceso'] and not mejor_solucion_global['suceso']:
            mejor_solucion_global = mejor_solucion_iter

        elif mejor_solucion_iter['suceso'] == mejor_solucion_global['suceso']:

            if mejor_solucion_iter['suceso']:
                if mejor_solucion_iter['tiempo_final'] < mejor_solucion_global['tiempo_final']:
                    mejor_solucion_global = mejor_solucion_iter
            else:
                if mejor_solucion_iter['mejor_distancia'] < mejor_solucion_global['mejor_distancia']:
                    mejor_solucion_global = mejor_solucion_iter

    grafo.actualizar_feromonas(resultados_iter, rho, Q)

# 5. MEJOR SOLUCIÓN

print("\n" + "=" * 50)
print("MEJOR SOLUCIÓN")
print("=" * 50)

if mejor_solucion_global:

    s = mejor_solucion_global
    t_L_dias = s['t_L'] / DIA

    print(f"t_L: {t_L_dias:.4f} días")
    print(f"Delta_v: {s['delta_v']:.4f} km/s")
    print(f"Theta: {s['theta_deg']:.4f}° ({s['theta']:.4f} rad)")

    if s['suceso']:
        print("Resultado: ÉXITO")
        print(f"Tiempo: {s['tiempo_final']:.4f} días")
    else:
        print("Resultado: FRACASO")  
        print(
            f"Distancia mínima: "
            f"{s['mejor_distancia']:.2f} M km"
        )

else:
    print("No se encontró solución")


# 6. GRÁFICA

if mejor_solucion_global and mejor_solucion_global['suceso']:

    print("\nGenerando gráfica...")

    s = mejor_solucion_global

    x_nave = s['x_list']
    y_nave = s['y_list']

    t_L_seg = s['t_L']
    tiempo_vuelo_seg = s['tiempo_final'] * DIA

    pos_tierra_ini = tierra.posicion(t_L_seg)
    pos_tierra_fin = tierra.posicion(t_L_seg + tiempo_vuelo_seg)

    pos_marte_ini = marte.posicion(t_L_seg)
    pos_marte_fin = marte.posicion(t_L_seg + tiempo_vuelo_seg)

    t_tierra = np.linspace(0, T_tierra, 365)
    orb_tierra = np.array([tierra.posicion(t) for t in t_tierra])

    t_marte = np.linspace(0, T_marte, 687)
    orb_marte = np.array([marte.posicion(t) for t in t_marte])

    plt.figure(figsize=(12, 10))

    plt.plot(
        orb_tierra[:, 0] / UA,
        orb_tierra[:, 1] / UA,
        'b-',
        linewidth=1,
        alpha=0.5,
        label='Órbita Tierra'
    )

    plt.plot(
        orb_marte[:, 0] / UA,
        orb_marte[:, 1] / UA,
        'r-',
        linewidth=1,
        alpha=0.5,
        label='Órbita Marte'
    )

    plt.plot(
        x_nave / UA,
        y_nave / UA,
        'k-',
        linewidth=2,
        label='Trayectoria'
    )

    plt.scatter(
        pos_tierra_ini[0]/UA,
        pos_tierra_ini[1]/UA,
        color='blue',
        s=100,
        label='Tierra (salida)'
    )

    plt.scatter(
        pos_tierra_fin[0]/UA,
        pos_tierra_fin[1]/UA,
        color='lightblue',
        s=100,
        label='Tierra (llegada)'
    )

    plt.scatter(
        pos_marte_ini[0]/UA,
        pos_marte_ini[1]/UA,
        color='darkred',
        s=100,
        label='Marte (salida)'
    )

    plt.scatter(
        pos_marte_fin[0]/UA,
        pos_marte_fin[1]/UA,
        color='salmon',
        s=100,
        label='Marte (llegada)'
    )

    plt.scatter(
        x_nave[0]/UA,
        y_nave[0]/UA,
        color='white',
        s=10,
        edgecolors='red',
        label='Nave (salida)'
    )

    plt.scatter(
        x_nave[-1]/UA,
        y_nave[-1]/UA,
        color='black',
        s=10,
        label='Nave (llegada)'
    )

    plt.scatter(
        0,
        0,
        color='yellow',
        s=200,
        edgecolors='orange',
        label='Sol'
    )

    plt.title(
        f"Trayectoria - t_L={t_L_dias:.4f} días, "
        f"Δv={s['delta_v']:.4f} km/s, "
        f"θ={s['theta_deg']:.4f}°"
    )

    plt.xlabel("x (UA)")
    plt.ylabel("y (UA)")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()