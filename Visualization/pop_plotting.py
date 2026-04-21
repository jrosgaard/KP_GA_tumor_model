# Population plotting function
# This module contains functions for plotting population dynamics and fitness over generations.

def plot_pop(population1, population2, population3, title="Population Dynamics over Generations", xlabel="Generation", ylabel="Population", xrange=None, yrange=None):
    """
    Plots the population over generations.

    Parameters:
    population1 (list): A list of population values for the first population over each generation.
    population2 (list): A list of population values for the second population over each generation.
    population3 (list): A list of population values for the third population over each generation.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """
    import os
    import matplotlib.pyplot as plt

    generations = list(range(len(population1)))
    
    plt.figure(figsize=(6, 4))
    plt.plot(generations, population1, color='blue', label='Effector Cells')
    plt.plot(generations, population2, color='red', label='Tumor Cells')
    plt.plot(generations, population3, color='green', label='IL-2 Levels')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if yrange:
        plt.ylim(yrange)
    if xrange:
        plt.xlim(xrange)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig("population_dynamics.png")

def pop_fit_plot(fitness_history, population1, population2, population3, title="Populations and Fitness", yrange=None):
    """
    Plots the fitness over generations.

    Parameters:
    population1 (list): A list of population values for the first population over each generation.
    population2 (list): A list of population values for the second population over each generation.
    population3 (list): A list of population values for the third population over each generation.
    fitness_history (list): A list of fitness values over each generation.
    """
    import os
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)

    generations_pop = list(range(len(population1)))
    
    plt.plot(generations_pop, population1,  color='blue', label='Effector Cells')
    plt.plot(generations_pop, population2,  color='red', label='Tumor Cells')
    plt.plot(generations_pop, population3,  color='green', label='IL-2 Levels')
    plt.title("Populations over Generations")
    plt.xlabel('Generation')
    plt.ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        plt.ylim(yrange_pop)
   
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    generations_fit = list(range(len(fitness_history)))
    plt.plot(generations_fit, fitness_history, marker='o', color='purple', label='Fitness')
    plt.title("Fitness over Generations")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    
    yrange_fit = None
    if yrange_fit is not None:
        plt.ylim(yrange_fit)
   
    plt.grid(True)
    plt.legend(loc='upper right')
    #plt.savefig("fitness_over_generations.png")

def plot_quad(t, tau, fitness_history, x, y, z, population1, population2, population3, s_1, s_2, title="Populations and Fitness", yrange=None):
    """
    Plots the fitness over generations.

    Parameters:
    population1 (list): A list of population values for the first population over each generation.
    population2 (list): A list of population values for the second population over each generation.
    population3 (list): A list of population values for the third population over each generation.
    fitness_history (list): A list of fitness values over each generation.
    """
    import os
    import matplotlib.pyplot as plt

    time = tau

    # create subplots    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plt.subplot(2, 2, 1)
    
    plt.plot(time, x, color='blue', label='x(t)')
    plt.plot(time, y, color='red', label='y(t)')
    plt.plot(time, z, color='green', label='z(t)')
    plt.title("Populations over Time")
    plt.xlabel('tau')
    plt.ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        plt.ylim(yrange_pop)
   
    plt.grid(True)
    plt.legend(loc='upper right')
    
    
    plt.subplot(2, 2, 2)
    generations_pop = list(range(len(population1)))
    plt.plot(time, population1, color='blue', label='Effector Cells')
    plt.plot(time, population2, color='red', label='Tumor Cells')
    plt.plot(time, population3, color='green', label='IL-2 Levels')
    plt.title("Populations over Time")
    plt.xlabel('tau')
    plt.ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        plt.ylim(yrange_pop)
    plt.grid(True)
    plt.legend(loc='upper right')


    plt.subplot(2, 2, 3)
    generations_treatment = list(range(len(s_1)))
    plt.plot(time, s_1, color='orange', label=' Ext. Immune cells  s_1')
    plt.plot(time, s_2, color='purple', label='Ext. IL-2 dosing s_2')
    plt.title("Treatment over Time")
    plt.xlabel('tau')
    plt.ylabel('Dose')
    
    yrange_fit = None
    if yrange_fit is not None:
        plt.ylim(yrange_fit)
    plt.grid(True)
    plt.legend(loc='upper right')


    plt.subplot(2, 2, 4)
    generations_fit = list(range(len(fitness_history)))
    plt.plot(generations_fit, fitness_history, color='purple', label='Fitness')
    plt.title("Fitness over Generations")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    
    yrange_fit = None
    if yrange_fit is not None:
        plt.ylim(yrange_fit)
    plt.grid(True)
    plt.legend(loc='upper right')

    #plt.savefig("fitness_over_generations.png")

    return fig, axes

def plot_quad_init():
    import os
    import matplotlib.pyplot as plt

    # create subplots    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.4, 'wspace': 0.2})

    return fig, axes

def plot_quad_update(axes, t, tau, fitness_history, x, y, z, population1, population2, population3, s_1, s_2, title="Populations and Fitness", yrange=None):
    # Population plotting function for the slider notebook
    """
    Plots the fitness over generations.

    Parameters:
    population1 (list): A list of population values for the first population over each generation.
    population2 (list): A list of population values for the second population over each generation.
    population3 (list): A list of population values for the third population over each generation.
    fitness_history (list): A list of fitness values over each generation.
    """
    import os
    import matplotlib.pyplot as plt

    # unpack axes
    ax1, ax2, ax3, ax4 = axes.ravel()

    for ax in axes.ravel():
        ax.clear()

    time = tau

    # create subplots
    ax1.plot(time, x, color='blue', label='x(t)')
    ax1.plot(time, y, color='red', label='y(t)')
    ax1.plot(time, z, color='green', label='z(t)')
    ax1.set_title("Populations over Time")
    ax1.set_xlabel('tau')
    ax1.set_ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        ax1.set_ylim(yrange_pop)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    
    generations_pop = list(range(len(population1)))
    ax2.plot(time, population1, color='blue', label='Effector Cells')
    ax2.plot(time, population2, color='red', label='Tumor Cells')
    ax2.plot(time, population3, color='green', label='IL-2 Levels')
    ax2.set_title("Populations over Time")
    ax2.set_xlabel('tau')
    ax2.set_ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        ax2.set_ylim(yrange_pop)
    ax2.grid(True)
    ax2.legend(loc='upper right')


    generations_treatment = list(range(len(s_1)))
    ax3.plot(time, s_1, color='orange', label=' Ext. Immune cells  s_1')
    ax3.plot(time, s_2, color='purple', label='Ext. IL-2 dosing s_2')
    ax3.set_title("Treatment over Time")
    ax3.set_xlabel('tau')
    ax3.set_ylabel('Dose')
    
    yrange_fit = None
    if yrange_fit is not None:
        ax3.set_ylim(yrange_fit)
    ax3.grid(True)
    ax3.legend(loc='upper right')


    generations_fit = list(range(len(fitness_history)))
    ax4.plot(generations_fit, fitness_history, color='purple', label='Fitness')
    ax4.set_title("Fitness over Generations")
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness')
    
    yrange_fit = None
    if yrange_fit is not None:
        ax4.set_ylim(yrange_fit)
    ax4.grid(True)
    ax4.legend(loc='upper right')

    axes[0, 0].figure.suptitle(title)

def plot_doub_init():
    import os
    import matplotlib.pyplot as plt

    # create subplots    
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'hspace': 0.6})

    return fig, axes

def plot_doub_update(axes, t, tau, x, y, z, population1, population2, population3, s_1, s_2, title="Populations and Fitness", yrange=None):
    # Population plotting function for the slider notebook
    """
    Plots the fitness over generations.

    Parameters:
    population1 (list): A list of population values for the first population over each generation.
    population2 (list): A list of population values for the second population over each generation.
    population3 (list): A list of population values for the third population over each generation.
    fitness_history (list): A list of fitness values over each generation.
    """
    import os
    import matplotlib.pyplot as plt

    # unpack axes
    ax1, ax2 = axes.ravel()

    for ax in axes.ravel():
        ax.clear()


    time = tau

    # create subplots
    
    ax1.plot(time, x, color='blue', label='x(t)')
    ax1.plot(time, y, color='red', label='y(t)')
    ax1.plot(time, z, color='green', label='z(t)')
    ax1.set_title("Populations over Time")
    ax1.set_xlabel('tau')
    ax1.set_ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        ax1.set_ylim(yrange_pop)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    ax2.plot(time, population1, color='blue', label='Effector Cells')
    ax2.plot(time, population2, color='red', label='Tumor Cells')
    ax2.plot(time, population3, color='green', label='IL-2 Levels')
    ax2.set_title("Populations over Time")
    ax2.set_xlabel('tau')
    ax2.set_ylabel('Population')
    
    yrange_pop = None
    if yrange_pop is not None:
        ax2.set_ylim(yrange_pop)
    ax2.grid(True)
    ax2.legend(loc='upper right')

    axes[0].figure.suptitle(title)
