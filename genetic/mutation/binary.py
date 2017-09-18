import random
import numpy as np

class BinaryMutation:
    def __init__(self, mutation_probability):
        """
        :param mutation_probability: the gene-wise probability that a mutation will take place, typically c/m where c is constant of order 1 and m is the chromosome length
        """
        self.mutation_probability = mutation_probability

    def mutate(self, chromosome, generation):
        selected_genes = np.random.choice([0, 1], (len(chromosome), 1), p=[1-self.mutation_probability, self.mutation_probability])
        np.logical_xor(selected_genes, chromosome, chromosome)





if __name__ == "__main__":
    # c = BinaryMutation(0.2)
    # x = np.array([[1],[0],[1],[0]])
    # c.mutate(x,1)
    # print x
    x = np.array([[0]])

    c = BinaryMutation(0.02)

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

