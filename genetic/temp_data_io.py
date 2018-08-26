import json
import os, sys
import time

def get_main_dir():
    directory_path = "saved_populations/"
    if os.path.exists(directory_path):
        return directory_path
    else:
        return "../saved_populations/"


def save_temp_data(subfoldername, temp_name, temp_data, ):
    directory_path = get_main_dir() + subfoldername + "/temp/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + temp_name + ".json", 'w') as out:
        json.dump(temp_data, out, separators=(',',':'))

def load_temp_data(subfoldername, temp_name):
    directory_path = get_main_dir() + subfoldername + "/temp/"
    with open(directory_path + temp_name + ".json") as file:
        return json.load(file)

def wait_and_open_temp_data(subfoldername, temp_name):
    while 1:
        try:
            return load_temp_data(subfoldername, temp_name)
        except IOError:
            print "IOError in wait_and_open_temp_data()"
        except ValueError:
            print "ValueError in wait_and_open_temp_data()"
        except WindowsError:
            print "WindowsError in wait_and_open_temp_data()"
        except KeyError:
            print "KeyError in wait_and_open_temp_data()"
        except EOFError:
            print "EOFError in wait_and_open_temp_data()"
        except:
            print "Unexpected error: " + str(sys.exc_info()[0])


        time.sleep(1)

def save_temp_fitness(subfoldername, individual_index, individual_fitness):
    directory_path = get_main_dir() + subfoldername + "/temp/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    temp_name =  str(float(individual_fitness)) + "=" + str(individual_index)
    open(directory_path + temp_name, 'w').close()

def load_temp_fitness_scores(subfoldername, population_size):
    directory_path = get_main_dir() + subfoldername + "/temp/"
    files = os.listdir(directory_path)
    fitness_scores = [None] * population_size
    for filename in files:
        if filename != "generation_and_decoded_variable_vectors.json":
            s = filename.split("=")
            individual_fitness = float(s[0])
            individual_index = int(s[1])
            fitness_scores[individual_index] = individual_fitness
    return fitness_scores

def clear_temp_folder(subfoldername):
    directory_path = get_main_dir() + subfoldername + "/temp/"
    if not os.path.exists(directory_path):
        print "Cannot clear nonexistent folder"
    else:
        files = os.listdir(directory_path)
        for file in files:
            os.remove(directory_path + file)

def wait_for_temp_fitness_scores(subfoldername, population_size, num_extra_files=1):
    directory_path = get_main_dir() + subfoldername + "/temp/"
    while 1:
        if len(os.listdir(directory_path)) == population_size + num_extra_files:
            break
        time.sleep(1)
