


def average_fitness(genetic_algorithm, generations_per_run, number_of_runs):
    return sum(genetic_algorithm.run(generations_per_run).best_fitness
               for run in range(number_of_runs)) / float(number_of_runs)