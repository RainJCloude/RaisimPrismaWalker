import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading the pickle file: {e}")
        return None

def print_loss(x, y ):

    plt.plot(x, label=y)

    plt.title('Loss function varying lambda')
    plt.xlabel('Episode')
    plt.ylabel('Loss value')

    plt.legend()
    plt.grid()
    


# Example usage
if __name__ == "__main__":
    task_path = os.path.dirname(os.path.realpath(__file__))

    lam = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    #lam = [0.9, 0.91] #gamma= 0.995
    #lam = [0.9, 0.92, 0.95]  #Vince il 0.92 per gamma=0.996 con min loss 0.65
    #lam = [0.9, 0.91] #gamma= 0.997 fa schifosissimo
    #lam = [0.9, 0.91, 0.94, 0.95, 0.98, 0.99] #gamma=0.998 fa schifomegasissimo arriva anche a 4
    #lam = [0.9, 0.98, 0.99] #0.98 e' il migliore per gamma=0.999
    lam = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    file_paths = []
    for i in lam:
        path_to_data = task_path + "/../../../.." + "/data/prisma_walker_locomotion/" 
        file_paths.append(path_to_data + "lam_" + str(i) + "__gamma_1" + "/loss/loss.pkl")

        # Read the pickle file
        loss_dict = read_pickle_file(file_paths[-1])

        # Process the loaded data (replace this with your actual data processing logic)
        if loss_dict is not None:
            x_axis = loss_dict.values() #prendo solo i valori del dizionario
            x_axis = list(x_axis)

            print_loss(x_axis, "lambda = " + str(i) )
    
    plt.show()
