# ACO - Optimización de Trayectorias Interplanetarias

Repositorio correspondiente al trabajo de grado enfocado en la optimización de trayectorias interplanetarias mediante algoritmos de colonias de hormigas (Ant Colony Optimization, ACO).

## Descripción

Este proyecto desarrolla un modelo computacional para la simulación y optimización de trayectorias interplanetarias entre la Tierra y Marte, utilizando herramientas de mecánica orbital, integración numérica y algoritmos metaheurísticos inspirados en el comportamiento colectivo de las hormigas.

El trabajo combina:

- Dinámica orbital simplificada
- Integración numérica de ecuaciones diferenciales
- Algoritmos de optimización por colonia de hormigas
- Simulación computacional
- Visualización de trayectorias

---

## Objetivo

Desarrollar un sistema computacional capaz de encontrar trayectorias interplanetarias eficientes mediante técnicas de optimización metaheurística basadas en ACO.

---

## Tecnologías utilizadas

- Python
- NumPy
- SciPy
- Matplotlib

---

## Estructura del proyecto

```text
src/
│
├── constantes_parametros.py     # Parámetros físicos y constantes
├── dinamicas_orbitales.py       # Modelo dinámico orbital
├── integrador_numerico.py       # Integración numérica
├── optimizacion_hormigas.py     # Algoritmo ACO
├── validacion.py                # Validación de resultados
└── ejecucion_principal.py       # Ejecución principal del proyecto
```

---

## Dependencias

Instalar las dependencias con:

```bash
pip install numpy scipy matplotlib
```

---

## Ejecución

Ejecutar el archivo principal:

```bash
python ejecucion_principal.py
```

---

## Resultados esperados

El sistema permite:

- Simular trayectorias orbitales
- Evaluar soluciones de transferencia interplanetaria
- Optimizar rutas mediante ACO
- Visualizar trayectorias y resultados numéricos

---

## Autor

Carlos Blanco

Trabajo de grado — Matemática Aplicada
