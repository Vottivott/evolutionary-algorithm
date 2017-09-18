import random
import numpy as np

class CreepMutation:
    def __init__(self, mutation_probability, creep_probability, creep_rate, use_normal_distribution=False):
        """
        :param mutation_probability: the gene-wise probability that a mutation will take place
        :param creep_probability: the probability of using creep mutation given that a mutation will be made
        :param creep_rate: the width of the distribution of possible new positions on the number line; the value may (if using uniform distribution) maximally decrease to -creep_rate/2 or increase to creep_rate/2
        :param use_normal_distribution: if a normal distribution should be used for the creep step, in which case creep_rate/2 is used as the standard deviation. If false, a uniform distribution is used.
        """
        self.mutation_probability = mutation_probability
        self.creep_probability = creep_probability
        self.creep_rate = creep_rate
        self.use_normal_distribution = use_normal_distribution

    def mutate(self, chromosome, generation):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_probability:
                if random.random() < self.creep_probability:
                    if self.use_normal_distribution:
                        chromosome[i] = np.clip(chromosome[i] + np.random.normal(0, self.creep_rate/2.0),
                                                0,
                                                1)
                    else:
                        chromosome[i] = np.clip(chromosome[i] - self.creep_rate/2.0 + random.random() * self.creep_rate,
                                                0,
                                                1)
                else:
                    chromosome[i] = random.random() # ordinary mutation





if __name__ == "__main__":
    x = np.array([[0.5]])

    c = CreepMutation(0.02, 0.8, 0.005, True)

    import matplotlib.pyplot as plt
    import sys

    a = []
    for i in range(300):
       c.mutate(x, 1)
       a.append(np.asscalar(x[0]))

    fig, ax = plt.subplots()
    [graph] = ax.plot(a)
    ax.set_ylim((0,1))

    def press(event):
        #print('press', event.key)
        sys.stdout.flush()
        a = []
        for i in range(300):
            c.mutate(x, 1)
            a.append(np.asscalar(x[0]))
        graph.set_ydata(a)
        fig.canvas.draw()
        #if event.key == 'x':
        #    pass

    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

