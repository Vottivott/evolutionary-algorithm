import json
import os

from genetic.algorithm import PrunedPopulationData
from pso.algorithm import PrunedPSOPopulationData

def get_main_dir():
    directory_path = "saved_populations/"
    if os.path.exists(directory_path):
        return directory_path
    else:
        return "../saved_populations/"

def prune_population_data(subfoldername, num):
    p = load_population_data(subfoldername, num)
    if "swarm_best_performance" in p.keys():
        pruned = PrunedPSOPopulationData(p)
    else:
        pruned = PrunedPopulationData(p)
    directory_path = get_main_dir() + subfoldername + "/pruned/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + str(pruned.generation) + ".json", 'w') as out:
        json.dump(pruned, out, separators=(',',':'))
    os.remove(get_main_dir() + subfoldername + "/" + str(num) + ".json")

def save_population_data(subfoldername, population_data, keep_last_n=None, keep_mod = 100):
    directory_path = get_main_dir() + subfoldername + "/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if keep_last_n:
        files = os.listdir(directory_path)
        for filename in files:
            if filename[-5:]==".json":
                num = int(filename[:-5])
                diff = population_data["generation"] - num
                if diff >= keep_last_n and not (keep_mod is not None and num % keep_mod == 0):
                    prune_population_data(subfoldername, num)#os.remove(directory_path + filename)
    with open(directory_path + str(population_data["generation"]) + ".json", 'w') as out:
        json.dump(population_data, out, separators=(',',':'))


def get_latest_generation_number(subfoldername):
    directory_path = get_main_dir() + subfoldername + "/"
    file_loaded = False
    if not os.path.exists(directory_path):
        return None
    files = os.listdir(directory_path)
    if not len(files):
        return None

    nums = (int(file[:-5]) for file in (f for f in files if f[-5:] == ".json"))
    nums = sorted(nums)
    generation = nums[-1]
    return generation


def load_population_data(subfoldername, generation):
    directory_path = get_main_dir() + subfoldername + "/"
    if generation == -1:
        file_loaded = False
        if not os.path.exists(directory_path):
            return None
        files = os.listdir(directory_path)
        if not len(files):
            return None

        nums = (int(file[:-5]) for file in (f for f in files if f[-5:]==".json"))
        nums = sorted(nums)
        if not len(nums):
            return None
        generation = nums[-1]
        print "Loading latest generation of " + str(subfoldername) + ": " + str(generation)
        while 1:
            try:
                with open(directory_path + str(generation) + ".json") as file:
                    return json.load(file)
            except ValueError:
                old = nums[-1]
                del nums[-1]
                if len(nums) == 0:
                    print "No file left to try! Returning None"
                    return None
                else:
                    generation = nums[-1]
                    print "ValueError on " + str(old) + ", trying with " + str(generation) + " instead!"
    else:
        try:
            with open(directory_path + str(generation) + ".json") as file:
                return json.load(file)
        except IOError:
            print "Loading pruned version of " + str(generation) + ".json"
            with open(directory_path + "pruned/" + str(generation) + ".json") as file:
                return json.load(file)


def count_genes():
    import numpy as np
    #subfoldername = "7 counters (length 4), 15 radars (max_steps = 250, step_size = 4), velocity up+down"
    # subfoldername = "feedforward_larger_no_enemies"
    # prune_population_data(subfoldername, 144)
    subfoldername = "enemy"
    p = load_population_data(subfoldername, -1)
    print sum(p.population[40][-1500:])
    print sum(sum(ind[-1500:] for ind in p.population))
    subfoldername = "copter"
    p = load_population_data(subfoldername, -1)
    print sum(p.population[40][-1500:])
    print sum(sum(ind[-1500:] for ind in p.population))

if __name__ == "__main__":
    pass
    # count_genes()

    # save_temp_fitness("hej", 4, 25.3231)
    # save_temp_fitness("hej", 2, -5.3231)
    # save_temp_fitness("hej", 3, -25.3231)
    # save_temp_fitness("hej", 1, -55.3231)
    # save_temp_fitness("hej", 0, -1525.3231)
    # print load_temp_fitness_scores("hej", 5)
    # clear_temp_folder("hej")

    # p = load_population_data(subfoldername, 105)
    # print p
    # p = load_population_data(subfoldername, 106)
    # print p
    # p.generation += 1
    # save_population_data(subfoldername, p)
    # print p
    # num_extra_genes = 30 * 50
    # for i,chromosome in enumerate(p.population):
    #     initial_h = np.zeros((num_extra_genes,1))
    #     p.population[i] = np.vstack((chromosome, initial_h))
    # print p
    # save_population_data(subfoldername, p)
    # print "Finished!"