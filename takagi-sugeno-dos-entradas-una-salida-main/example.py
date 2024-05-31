from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from lkfuzzy import *


def main():
    food = InputVariable('food', range=[0, 10])
    service = InputVariable('service', range=[0, 10])
    
    food['baja'] = TriangularFunction(0, 0, 5)
    food['buena'] = TriangularFunction(0, 5, 10)
    food['excelente'] = TriangularFunction(5, 10, 10)

    service['baja'] = TriangularFunction(0, 0, 5)
    service['buena'] = TriangularFunction(0, 5, 10)
    service['excelente'] = TriangularFunction(5, 10, 10)

    rules = [
        Rule(food['baja'] & service['baja'], 0),
        Rule(food['baja'] & service['buena'], 5),
        Rule(food['buena'] & service['baja'], 8),
        Rule(food['baja'] & service['excelente'], 10),
        Rule(food['excelente'] & service['baja'], 9),
        Rule(food['buena'] & service['buena'], 10),
        Rule(food['buena'] & service['excelente'], 12),
        Rule(food['excelente'] & service['buena'], 15),
        Rule(food['excelente'] & service['excelente'], 20),
    ]

    system = FuzzySystem(rules)

    test_on_examples(system)
    draw_heatmap(system)
    draw_surface(system)


def test_on_examples(system):
    ValorPrueba = namedtuple('valorPrueba', 'food service')

    valoresPrueba = [
        ValorPrueba(10, 10),
        ValorPrueba(4, 4),
        ValorPrueba(0, 0),
        ValorPrueba(10, 0),
        ValorPrueba(0, 10),
        ValorPrueba(8, 8),
        ValorPrueba(2, 6),
        ValorPrueba(6, 2),
        ValorPrueba(9, 1),
        ValorPrueba(1, 9),
        ValorPrueba(3, 7),
        ValorPrueba(0, 0),
    ]

    for valorPrueba in valoresPrueba:
        
        propina = system.compute(food=valorPrueba.food, service=valorPrueba.service)
        print(f'valor entrada calificación comida: {valorPrueba.food:2}/10, valor entrada calificación servicio: {valorPrueba.service:2}/10 -> propina: {propina:.1f}%')


def draw_heatmap(system):
    resolution = 20
    food_values = np.linspace(0, 10, resolution)
    service_values = np.linspace(0, 10, resolution)

    food_grid, service_grid = np.meshgrid(food_values, service_values)
    tip_grid = np.zeros_like(food_grid)
    for food_index in range(resolution):
        for service_index in range(resolution):
            food_value = food_values[food_index]
            service_value = service_values[service_index]
            tip_grid[food_index, service_index] = system.compute(food=food_value, service=service_value)

    fig, ax = plt.subplots()
    cp = ax.contourf(service_grid, food_grid, tip_grid, levels=np.linspace(0, 20, num=21), cmap=plt.cm.RdYlGn, extend='both')
    cbar = fig.colorbar(cp, label='% Propina')
    cbar.set_ticks(np.arange(0, 21, 2))  # Establecer los ticks del colorbar cada 2 unidades de propina
    ax.set_xlabel('Calidad del Servicio')
    ax.set_ylabel('Calidad de la Comida')
    ax.set_title('Comida y Servicio vs % Propina')

    # Definir la ubicación de las anotaciones
    step = resolution // 4
    for i in range(0, resolution, step):
        for j in range(0, resolution, step):
            x = service_values[j]
            y = food_values[i]
            tip = tip_grid[i, j]
            ax.text(x, y, f'{tip:.1f}%', ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)  # Agregar grid

  
    fig.show()
    plt.pause(1)

            
def draw_surface(system):
    resolution = 20
    food_values = np.linspace(0, 10, resolution)
    service_values = np.linspace(0, 10, resolution)

    food_grid, service_grid = np.meshgrid(food_values, service_values)
    tip_grid = np.zeros_like(food_grid)
    for food_index in range(resolution):
        for service_index in range(resolution):
            food_value = food_values[food_index]
            service_value = service_values[service_index]
            tip_grid[food_index, service_index] = system.compute(food=food_value, service=service_value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Superficie principal
    surf = ax.plot_surface(service_grid, food_grid, tip_grid, cmap='RdYlGn', edgecolor='none', alpha=0.8)

    # Contornos en los planos xy, xz y yz
    ax.contour(service_grid, food_grid, tip_grid, zdir='z', offset=0, cmap='RdYlGn')
    ax.contour(service_grid, food_grid, tip_grid, zdir='x', offset=0, cmap='RdYlGn')
    ax.contour(service_grid, food_grid, tip_grid, zdir='y', offset=10, cmap='RdYlGn')

    # Etiquetas de los ejes
    ax.set_xlabel('Calidad Servicio')
    ax.set_ylabel('Calidad Comida')
    ax.set_zlabel('% Propina')

    # Título del gráfico
    ax.set_title('Superficie de Propina en función de la Calidad de Comida y Servicio')

    # Agregar una superficie de referencia (un plano en el nivel máximo de propina)
    ax.plot_surface(service_grid, food_grid, np.full_like(tip_grid, 20), color='gray', alpha=0.3)

    # Agregar sombras
    ax.set_proj_type('persp')

    # Agregar una leyenda para la superficie principal
   # fig.colorbar(surf, ax=ax, label='% Propina')

    plt.show()
    plt.pause(1)

if __name__ == '__main__':
    main()
