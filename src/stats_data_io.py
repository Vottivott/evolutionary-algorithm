import json
import os

# from genetic.algorithm import PrunedPopulationData
from pso.algorithm import PrunedPSOPopulationData


def get_main_dir():
    directory_path = "saved_stats/"
    if os.path.exists(directory_path):
        return directory_path
    else:
        return "../saved_stats/"

def save_stats(subfilename, stats_handler, population_data):
    directory_path = get_main_dir()
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    old_stats = None
    try:
        with open(directory_path + subfilename + ".json") as file:
            old_stats = json.load(file)
    except ValueError:
        old_stats = None
    except IOError:
        old_stats = None
    with open(directory_path + subfilename + ".json", 'w') as out:
        json.dump(stats_handler.update(old_stats, population_data), out, separators=(',',':'))


def load_stats(subfilename):
    directory_path = get_main_dir()
    with open(directory_path + subfilename + ".json") as file:
        return json.load(file)

if __name__ == "__main__":
    s = load_stats("PSO_worm_3segs_planar")
    print s.generations
