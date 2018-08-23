from evomath import *
import numpy as np
import matplotlib.pyplot as plt

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def vline(x):
    axes = plt.gca()
    y_vals = np.array(axes.get_ylim())
    x_vals = 0.0 * y_vals + x
    plt.plot(x_vals, y_vals, '--')

class EvoStatsHandler:
    def __init__(self):
        pass

    def update(self, old_stats, population_data):
        if old_stats is None:
            old_stats = {}
            old_stats["generations"] = []
            old_stats["best_fitness"] = []
            old_stats["best_fitness_all_time"] = []
            old_stats["avg_fitness"] = []
            old_stats["decoded_variable_vectors"] = []
        if population_data.generation <= old_stats["generations"][-1]:
            diff = 1 + old_stats["generations"][-1] - population_data.generation
            print "Current generation " + str(population_data.generation) + " is <= latest stats generation " + str(old_stats["generations"][-1])
            print "Removing " + str(diff) + " latest stat entries."
            del old_stats["generations"][-diff:]
            del old_stats["best_fitness"][-diff:]
            del old_stats["best_fitness_all_time"][-diff:]
            del old_stats["avg_fitness"][-diff:]
        old_stats["generations"].append(population_data.generation)
        old_stats["best_fitness"].append(population_data.best_fitness)
        old_stats["best_fitness_all_time"].append(max(old_stats["best_fitness"]))
        old_stats["avg_fitness"].append(mean(population_data.fitness_scores))
        old_stats["decoded_variable_vectors"] = np.ndarray.flatten(np.array(population_data.decoded_variable_vectors))
        return old_stats

    def produce_graph(self, stats, filename):
        plt.clf()
        plt.figure(1, figsize=(5, 3 * 3.13), facecolor='whitesmoke')
        num_plots = 2
        current_plot = 1
        fitness_plot = plt.subplot(num_plots, 1, current_plot)
        # plt.title('Fitness')
        plt.plot(stats["generations"], stats["best_fitness_all_time"], "r",
                 stats["generations"], stats["best_fitness"], "m",
                 stats["generations"], stats["avg_fitness"], "b")
        for i in range(stats["generations"][-1]/10 + 1):
            vline(i * 10)
        current_plot += 1

        positions_plot = plt.subplot(num_plots, 1, current_plot)
        # plt.title('Particle positions')
        plt.hist(stats["decoded_variable_vectors"], normed=True, color="b")
        current_plot += 1

        # plt.scatter(stats["generations"], stats["best_fitness"])
        # plt.show()
        plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    h = EvoStatsHandler()
    stats = {}
    stats["generations"] = range(5)
    stats["best_fitness"] = [0, 4, 4, 5, 9]
    stats["best_fitness_last_only"] = [0, 4, 3, 5, 9]
    stats["avg_fitness"] = [0, 2.5, 1.5, 2.3, 5.1]
    stats["inertia_weight"] = [1.4, 1.35, 1.32, 1.28, 1.23]
    stats["current_velocities_flattened"] = [0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6]
    stats["current_positions_flattened"] = [0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6]
    h.produce_graph(stats, None)