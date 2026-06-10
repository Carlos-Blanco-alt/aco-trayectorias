"""
==================================================
ACO - Ant Colony Optimization
==================================================

Implementación de las clases principales para el
algoritmo metaheurístico ACO.

Clases incluidas:
- Hormiga
- Colonia
- Grafo
- Heuristicas
"""

# Importar librerías.
import numpy as np

class Hormiga:
    def __init__(self, grafo):
        """
        Representa una hormiga artificial dentro del algoritmo ACO.
        Cada hormiga construye una solución candidata recorriendo el grafo.

        Args:
            grafo (Grafo): Grafo sobre el cual la hormiga construye su solución.
        """
        self.grafo = grafo
        
        # Variables de decisión.
        self.t_L = None
        self.delta_v = None
        self.theta = None
        
        # Índices de las opciones elegidas.
        self.idx_t_L = None
        self.idx_dv = None
        self.idx_theta = None
        
        # Aptitud de la solución construida.
        self.aptitud = None

    def construir_solucion_aleatoria(self):
        """
        Construye una solución inicial aleatoria seleccionando valores posibles
        para las variables de decisión del problema.
        
        Returns:
            tuple: Valores seleccionados para (t_L, delta_v, theta).
        """
        # Elegir índice aleatorio para t_L
        self.idx_t_L = np.random.randint(len(self.grafo.t_L_opciones))
        self.t_L = self.grafo.t_L_opciones[self.idx_t_L]

        # Elegir índice aleatorio para delta_v
        self.idx_dv = np.random.randint(len(self.grafo.dv_opciones))
        self.delta_v = self.grafo.dv_opciones[self.idx_dv]

        # Elegir índice aleatorio para theta
        self.idx_theta = np.random.randint(len(self.grafo.theta_opciones))
        self.theta = self.grafo.theta_opciones[self.idx_theta]

        return self.t_L, self.delta_v, self.theta

    def calcular_probabilidades(self, feromonas, heuristica, alpha, beta):
        """
        Calcula las probabilidades de transición utilizadas por la hormiga para
        seleccionar una opción dentro del algoritmo ACO.
        
        Args:
            feromonas (ndarray): Valores de feromona asociados a las
            posibles decisiones.
            
            heuristica (ndarray): Valores heurísticos asociados a las posibles
            decisiones.
            
            alpha (float): Importancia relativa de las feromonas.
            
            beta (float): Importancia relativa de la heurística.

        Returns:
            ndarray: Vector de probabilidades.
        """
        numerador = (feromonas ** alpha) * (heuristica ** beta)
        denominador = np.sum(numerador)
        
        # Evitar división por cero.
        if denominador == 0:
            return np.ones_like(numerador) / len(numerador)
        probabilidades = numerador / denominador
        return probabilidades

    def eleccion_ruleta(self, probabilidades):
        """
        Selecciona un índice utilizando el método de ruleta.
        Cada opción tiene una probabilidad de ser elegida proporcional a su peso
        asociado.

        Args:
            probabilidades (ndarray): Vector de probabilidades normalizadas.

        Returns:
            int: Índice seleccionado.
        """
        # Probabilidades acumuladas.
        acumulada = np.cumsum(probabilidades)
        
        # Fijar explícitamente para evitar errores de redondeo
        acumulada[-1] = 1.0
        
        # Número aleatorio en [0, 1]
        valor_aleatorio = np.random.random()
        
        # Selección probabilística
        for i, prob_acumulada in enumerate(acumulada):
            if valor_aleatorio <= prob_acumulada:
                return i
        return len(probabilidades) - 1

    def construir_solucion_probabilistica(self, alpha, beta, heuristicas=None):
        """
        Construye una solución utilizando la regla probabilística del
        algoritmo ACO.
        La selección de cada variable de decisión depende de:
            - Intensidad de feromonas
            - Información heurística

        Args:
            alpha(float): Peso de las feromonas.

            beta(float): Peso de la heurística.

            heuristicas (dict, optional): Diccionario con información heurística
                para cada variable.

        Returns:
            tuple: Valores seleccionados para (t_L, delta_v, theta).
        """
        # Heurísticas por defecto
        if heuristicas is None:
            heuristicas = {
                't_L': np.ones(len(self.grafo.t_L_opciones)),
                'delta_v': np.ones(len(self.grafo.dv_opciones)),
                'theta': np.ones((len(self.grafo.t_L_opciones),
                                len(self.grafo.theta_opciones)))
            }

        # PASO 1: Elegir t_L
        prob_t_L = self.calcular_probabilidades(
            self.grafo.C_t_L,
            heuristicas['t_L'],
            alpha,
            beta)
        self.idx_t_L = self.eleccion_ruleta(prob_t_L)
        self.t_L = self.grafo.t_L_opciones[self.idx_t_L]

        # PASO 2: Elegir theta (dependiente de t_L)
        eta_theta_fila = heuristicas['theta'][self.idx_t_L, :]

        prob_theta = self.calcular_probabilidades(
            self.grafo.C_theta,
            eta_theta_fila,
            alpha,
            beta)
        self.idx_theta = self.eleccion_ruleta(prob_theta)
        self.theta = self.grafo.theta_opciones[self.idx_theta]
        
        # PASO 3: Elegir delta_v (independiente)
        prob_dv = self.calcular_probabilidades(
            self.grafo.C_dv,
            heuristicas['delta_v'],
            alpha,
            beta)
        self.idx_dv = self.eleccion_ruleta(prob_dv)
        self.delta_v = self.grafo.dv_opciones[self.idx_dv]

        return self.t_L, self.delta_v, self.theta


class Colonia:
    def __init__(self, grafo, num_hormigas):
        """
        Inicializa la colonia de hormigas.

        Args:
            grafo (Grafo): Grafo utilizado por las hormigas.

            num_hormigas (int): Número de hormigas de la colonia.
        """
        self.grafo = grafo
        self.num_hormigas = num_hormigas

    def construir_soluciones(self, alpha=1.0, beta=1.0, heuristicas=None, aleatorio=False):
        """
        Construye las soluciones de todas las hormigas de la colonia.

        La construcción puede realizarse:
            - De forma aleatoria
            - Mediante reglas probabilísticas basadas en feromonas y heurísticas

        Args:
            alpha (float, optional): Peso de las feromonas.

            beta (float, optional): Peso de la información heurística.

            heuristicas (dict, optional): Información heurística utilizada 
            durante la construcción.

            aleatorio (bool, optional): 
                -Si es True, utiliza construcción aleatoria. 
                -Si es False, utiliza construcción probabilística ACO.

        Returns:
            list:
                Lista de hormigas con soluciones
                construidas.
        """

        hormigas = []
        for _ in range(self.num_hormigas):
            h = Hormiga(self.grafo)
            if aleatorio:
                h.construir_solucion_aleatoria()
            else:
                h.construir_solucion_probabilistica(alpha, beta, heuristicas)
            hormigas.append(h)
        return hormigas


class Grafo:
    def __init__(self, rangos, resolucion):
        """
        Inicializa el espacio de búsqueda.

        Args:
            rangos (dict):Rangos mínimo y máximo de cada variable de decisión.

            resolucion (dict): Número de discretizaciones para cada variable.
        """
        # Valores posibles para cada variable
        self.t_L_opciones = np.linspace(
            rangos['t_L'][0],
            rangos['t_L'][1],
            resolucion['t_L'])

        self.dv_opciones = np.linspace(
            rangos['delta_v'][0],
            rangos['delta_v'][1],
            resolucion['delta_v'])

        self.theta_opciones = np.linspace(
            rangos['theta'][0],
            rangos['theta'][1],
            resolucion['theta'])

        # Dimensiones D_i
        self.D_t_L = resolucion['t_L']
        self.D_dv = resolucion['delta_v']
        self.D_theta = resolucion['theta']

        # Vectores de feromonas C_i (inicializadas en 0.1)
        self.C_t_L = np.full(self.D_t_L, 0.01)
        self.C_dv = np.full(self.D_dv, 0.01)
        self.C_theta = np.full(self.D_theta, 0.01)

    def actualizar_feromonas(self, resultados, rho, Q=1):
        """
        Actualiza las feromonas del espacio de búsqueda utilizando los resultados
        de las hormigas.

        La actualización consta de:
            - Evaporación de feromonas
            - Depósito de nuevas feromonas

        Args:
            resultados (list): Lista con los resultados obtenidos por las hormigas.

            rho (float): Tasa de evaporación de feromonas.

            Q (float, optional): Factor de depósito de feromona.
        """
        # Evaporación de la feromona
        self.C_t_L = (1 - rho) * self.C_t_L
        self.C_dv = (1 - rho) * self.C_dv
        self.C_theta = (1 - rho) * self.C_theta

        # Depósito de la feromona
        for res in resultados:
            delta = Q / res['aptitud']
            self.C_t_L[res['idx_t_L']] += delta
            self.C_dv[res['idx_dv']] += delta
            self.C_theta[res['idx_theta']] += delta


class Heuristicas:
    def __init__(self, tierra, marte):
        """
        Inicializa las heurísticas orbitales.

        Args:
            tierra (Tierra): Objeto que modela la dinámica orbital de la Tierra.

            marte (Marte): Objeto que modela la dinámica orbital de Marte.
        """
        self.tierra = tierra
        self.marte = marte

    # Heurística de instante de lanzamiento
    def t_L(self, t_L_opciones):
        """
        Calcula la heurística asociada a las posibles instantes de lanzamiento.
        La heurística favorece configuraciones donde la distancia entre la Tierra y
        Marte es menor.

        Args:
            t_L_opciones (ndarray): Posibles instantes de lanzamiento.

        Returns:
            ndarray: Valores heurísticos normalizados.
        """
        eta_t_L = np.zeros(len(t_L_opciones))

        for i, t_L in enumerate(t_L_opciones):
            
            # Posiciones planetarias
            r_t = np.array(self.tierra.posicion(t_L))
            r_m = np.array(self.marte.posicion(t_L))
            
            # Distancia Tierra-Marte
            d = np.linalg.norm(r_m - r_t)
            eta_t_L[i] = 1 / (1 + d)

        return eta_t_L / np.max(eta_t_L)

    # Heurística del cambio de velocidad
    def delta_v(self, dv_opciones):
        """
        Calcula la heurística asociada a los posibles cambios de velocidad.
        
        La heurística favorece valores menores de delta_v con el fin de priorizar
        trayectorias energéticamente más económicas.

        Args:
            dv_opciones (ndarray): Valores posibles de delta_v.

        Returns:
            ndarray: Valores heurísticos normalizados.
        """
        eta_dv = np.zeros(len(dv_opciones))

        for i, dv in enumerate(dv_opciones):
            
            # Favorecer cambios de velocidad pequeños
            eta_dv[i] = 1 / (1 + dv)

        return eta_dv / np.max(eta_dv)

    # Heurística de dirección    
    def theta(self, t_L_opciones, theta_opciones):
        """
        Calcula la heurística asociada a la dirección del cambio de velocidad inicial.
        
        La heurística favorece direcciones alineadas con el vector desde la Tierra 
        hacia Marte en el instante de lanzamiento.

        La dirección del cambio de velocidad se expresa en una base local definida por:
            - La dirección tangencial a la órbita
            - La dirección perpendicular

        Returns:
            ndarray: Matriz heurística dependiente de t_L y theta.
        """
        H = np.zeros((len(t_L_opciones), len(theta_opciones)))
        
        for i, t_L in enumerate(t_L_opciones):
            
            # Posición y velocidad de la Tierra
            r_t = np.array(self.tierra.posicion(t_L))
            v_t = np.array(self.tierra.velocidad(t_L))
            
            # Vector Tierra -> Marte
            r_m = np.array(self.marte.posicion(t_L))
            r_tm = r_m - r_t
            norma = np.linalg.norm(r_tm)
            
            # Base local (tangencial y perpendicular)
            tangente = v_t / np.linalg.norm(v_t)
            perpendicular = np.array([-tangente[1], tangente[0]])
            
            for j, ang in enumerate(theta_opciones):
                
                # Dirección del impulso
                v_dir = np.cos(ang) * tangente + np.sin(ang) * perpendicular
                
                # Alineamiento angular
                cos_theta = np.dot(v_dir, r_tm) / norma
                H[i, j] = (1 + cos_theta) / 2
                
        return H / np.max(H)

    # Construir todas las heurísticas
    
    def construir_heuristicas(self, t_L_opciones, dv_opciones, theta_opciones):
        """
        Construye todas las heurísticas utilizadas por el algoritmo ACO.

        Se generan heurísticas para:
            - Instante de lanzamiento (t_L)
            - Magnitud del cambio de velocidad (delta_v)
            - Dirección del cambio de velocidad (theta)

        Args:
            t_L_opciones (ndarray): Posibles instantes de lanzamiento.

            dv_opciones (ndarray): Posibles magnitudes de cambio de velocidad.

            theta_opciones (ndarray): Posibles direcciones del cambio de velocidad.

        Returns:
            dict: Diccionario con las heurísticas normalizadas del problema.
        """
        return {
            "t_L": self.t_L(t_L_opciones),
            "delta_v": self.delta_v(dv_opciones),
            "theta": self.theta(t_L_opciones, theta_opciones)
        }
