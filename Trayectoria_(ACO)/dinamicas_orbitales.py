"""
==================================================
Dinamicas - Ant Colony Optimization
==================================================

Implementación de las clases que modelan las dinámicas de los 
planetas para el problema de optimización.

Clases incluidas:
- Tierra
- Marte
- Nave
"""
# Importar las librerías y constantes necesarias para modelar las dinámicas de los planetas y la nave.
import numpy as np
from constantes_parametros import (r_tierra, r_marte, omega_tierra, omega_marte, phi_marte)


class Tierra:
    def __init__(self, pradio=r_tierra, pomega=omega_tierra, pphi_tierra=0):
        """
        Inicializa un objeto de la clase Tierra con parámetros físicos básicos.

        Args:
            pradio (float): Radio de la Tierra en kilómetros (km).
            
            pomega (float): Velocidad angular de la Tierra en radianes por segundo (rad/s).
        """
        # Radio medio terrestre en kilómetros
        self.radio = pradio  
        
        # Velocidad angular terrestre en radianes/segundo
        self.omega = pomega
        
        # Fase inicial de la Tierra en radianes (opcional, por defecto 0)
        self.phi_tierra = pphi_tierra
        
    def posicion(self, t): 
        """
        Calcula la posición de la Tierra en función del tiempo.

        Args:
            t (float): Tiempo en segundos.

        Returns:
            numpy.ndarray: Vector de posición en coordenadas cartesianas [x, y].
        """

        x = self.radio * np.cos(self.omega * t + self.phi_tierra)
        y = self.radio * np.sin(self.omega * t + self.phi_tierra)
        return np.array([x, y])
    
    def velocidad(self, t): 
        """
        Calcula la velocidad de la Tierra en función del tiempo.

        Args:
            t (float): Tiempo en segundos.

        Returns:
            numpy.ndarray: Vector de velocidad en coordenadas cartesianas [vx, vy].
        """
        vx = -self.radio * self.omega * np.sin(self.omega * t + self.phi_tierra)
        vy = self.radio * self.omega * np.cos(self.omega * t + self.phi_tierra)
        return np.array([vx, vy])


class Marte:
    def __init__(self, pradio_m=r_marte, pomega_m=omega_marte, phi=phi_marte):
        """Modela Marte en órbita circular alrededor del Sol.
        Args:
            pradio_m (float): Radio de la órbita de Marte en kilómetros (km).
            pomega_m (float): Velocidad angular de Marte en radianes por segundo (rad/s).
            phi (float): Fase inicial de Marte en radianes.
        """
        self.radio_m = pradio_m
        self.omega_m = pomega_m
        self.phi = phi
        
    def posicion(self, t): 
        """
        Calcula la posición de Marte en función del tiempo.

        Args:
            t (float): Tiempo en segundos.

        Returns:
            numpy.ndarray: Vector de posición en coordenadas cartesianas [x, y].
        """
        angulo = self.omega_m * t + self.phi
        x = self.radio_m * np.cos(angulo)
        y = self.radio_m * np.sin(angulo)
        return np.array([x, y])
    
    def velocidad(self, t): 
        """
        Calcula la velocidad de Marte en función del tiempo.

        Args:
            t (float): Tiempo en segundos.

        Returns:
            numpy.ndarray: Vector de velocidad en coordenadas cartesianas [vx, vy].
        """
        angulo = self.omega_m * t + self.phi
        vx = -self.radio_m * self.omega_m * np.sin(angulo)
        vy = self.radio_m * self.omega_m * np.cos(angulo)
        return np.array([vx, vy])

class Nave:
    def __init__(self, tierra, t_L, pdelta_v, ptheta):
        """
        Inicializa una nave espacial a partir del estado orbital de la Tierra
        y un cambion de velocidad.

        Args:
            tierra (Tierra): Objeto de la clase Tierra utilizado como referencia
                            para obtener la posición y velocidad inicial.

            t_L (float): Instante inicial de lanzamiento en segundos.

            pdelta_v (float): Magnitud del cambio de velocidad inicial aplicado a la nave en km/s.

            ptheta (float): Ángulo de dirección del cambio de velocidad medido en la base local.
        """

        # velocidad y posición de la Tierra
        vel_tierra = np.array(tierra.velocidad(t_L))
        pos_tierra = np.array(tierra.posicion(t_L))

        # base local
        t = vel_tierra / np.linalg.norm(vel_tierra)   # tangente
        n = np.array([-t[1], t[0]])                   # perpendicular

        # dirección del cambio de velocidad en base local
        vec_dir = np.cos(ptheta)*t + np.sin(ptheta)*n

        # estado inicial
        self.delta_v = pdelta_v
        self.theta = ptheta
        self.pos = pos_tierra
        self.vel = vel_tierra + pdelta_v * vec_dir
        
    def obtener_estado(self):
        return np.concatenate([self.pos, self.vel])