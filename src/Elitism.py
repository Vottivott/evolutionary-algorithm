class Elitism:
    def __init__(self, num_copies):
        """
        :param num_copies: typically one or a few
        """
        self.num_copies = num_copies

    def elitism(self, population, best_individual, generation):
        for i in range(self.num_copies):
            population[i] = best_individual