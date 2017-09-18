class Elitism:
    def __init__(self, num_copies):
        """
        :param num_copies: typically one or a few
        """
        self.num_copies = num_copies

    def elitism(self, population, best_individual, generation):
        for i in range(self.num_copies):
            population[i] = best_individual

if __name__ == "__main__":
    import numpy as np
    e = Elitism(1)
    x = [np.array([[0]]), np.array([[2]]), np.array([[6]]), np.array([[5]])]

    e.elitism(x, np.array([[6]]), 1)
    print x