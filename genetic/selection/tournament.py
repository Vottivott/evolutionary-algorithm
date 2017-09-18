import random
import numpy as np

class TournamentSelection:

    def __init__(self, tournament_selection_parameter, tournament_size):
        """
        :param tournament_selection_parameter: typically between 0.7 and 0.9
        :param tournament_size:
        """
        self.tournament_selection_parameter = tournament_selection_parameter
        self.tournament_size = tournament_size

    def select(self, fitness_scores, generation):
        competitors = (random.randint(0, len(fitness_scores)-1) for _ in range(self.tournament_size))
        ranked_competitors = sorted(competitors, key=fitness_scores.__getitem__, reverse=True)
        # print map(fitness_scores.__getitem__, ranked_competitors)
        for best in ranked_competitors:
            if random.random() < self.tournament_selection_parameter:
                # print str(fitness_scores[best]) + " won"
                return best
            # else:
            #     print str(fitness_scores[best]) + " lost"
        return ranked_competitors[-1]


if __name__=="__main__":
    fitness_scores = [7, 8,5,2,10]
    t = TournamentSelection(0.75, 3)
    i = t.select(fitness_scores, 1)
    #f = t.select
    #i = f(fitness_scores, 1)
    print fitness_scores[i]
