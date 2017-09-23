import pickle
import os

def save_population_data(subfoldername, population_data, keep_last_n=None):
    directory_path = "../saved_populations/" + subfoldername + "/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if keep_last_n:
        files = os.listdir(directory_path)
        for filename in files:
            diff = population_data.generation - int(filename[:-4])
            if diff >= keep_last_n:
                os.remove(directory_path + filename)
    with open(directory_path + str(population_data.generation) + ".pkl", 'w') as out:
        pickle.dump(population_data, out)


def load_population_data(subfoldername, generation):
    directory_path = "../saved_populations/" + subfoldername + "/"
    if generation == -1:
        files = os.listdir(directory_path)
        nums = (int(file[:-4]) for file in files)
        generation = max(nums)
        print "Loading latest generation: " + str(generation)
    with open(directory_path + str(generation) + ".pkl") as file:
        return pickle.load(file)


if __name__ == "__main__":
    #subfoldername = "7 counters (length 4), 15 radars (max_steps = 250, step_size = 4), velocity up+down"
    subfoldername = "recurrent_no_enemies"
    p = load_population_data(subfoldername, -1)
    print p