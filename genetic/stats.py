import numpy as np
import matplotlib.pyplot as plt
import sys


def average_fitness(genetic_algorithm, generations_per_run, number_of_runs, generational_callback=None, runwise_callback=None):
    """
    Returns the average best fitness over the specified number of runs,
    calling generational_callback(run_index, PopulationData) between each generation if included
    and runwise_callback(run_index, PopulationData) between each run if included,
    """



    total = 0
    for run in range(number_of_runs):
        if generational_callback is not None:
            data = genetic_algorithm.run(generations_per_run, lambda data: generational_callback(run, data))
        else:
            data = genetic_algorithm.run(generations_per_run)
        total += data.best_fitness
        if runwise_callback is not None:
            runwise_callback(run, data)
    return total / float(number_of_runs)


def plot_fitness_curves(genetic_algorithm, generations_per_run, number_of_runs):
    x = np.array([[0]])


    a = [0]*number_of_runs

    fig, ax = plt.subplots()
    graphs = [graph] = ax.plot(a)
    ax.set_ylim((0, 1))

    plt.ion()


    def generational_callback(run, data):
        a[data.generation] = data.best_fitness
        # print data.best_fitness
        graph.set_ydata(a)
        # fig.canvas.draw()
        # plt.plot(a)
        # plt.pause(0.05)

    def runwise_callback(run, data):
        [g] = ax.plot(a)
        graphs.append(g)
        fig.canvas.draw()

    # def press(event):
    #     # print('press', event.key)
    #     sys.stdout.flush()
    #     a = []
    #     for i in range(300):
    #         c.mutate(x, 1)
    #         a.append(np.asscalar(x[0]))
    #     graph.set_ydata(a)
    #     fig.canvas.draw()
    #     # if event.key == 'x':
    #     #    pass

    # fig.canvas.mpl_connect('key_press_event', press)
    plt.show()
    return average_fitness(genetic_algorithm, generations_per_run, number_of_runs, generational_callback, runwise_callback)