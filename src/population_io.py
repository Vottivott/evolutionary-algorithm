import pickle
import os

def save_population(subfoldername, population, generation):
    directory_path = "../saved_populations/" + subfoldername + "/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + str(generation) + ".pkl", 'w') as out:
        pickle.dump(population, out)


def load_population(subfoldername, generation):
    directory_path = "../saved_populations/" + subfoldername + "/"
    with open(directory_path + str(generation) + ".pkl") as file:
        return pickle.load(file)