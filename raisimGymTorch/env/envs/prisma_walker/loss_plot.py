import pickle
import matplotlib.pyplot as plt
import os

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

def print_loss(x,y):
    plt.plot(x,y)
    plt.title('List Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


# Example usage
if __name__ == "__main__":
    task_path = os.path.dirname(os.path.realpath(__file__))

    lam = [9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1]
    file_paths = []
    for i in lam:
        path_to_data = task_path + "/../.." + "/data/prisma_walker_locomotion/" 
        file_paths.append(path_to_data + str(i) + "/loss")

        # Read the pickle file
        task_path = os.path.dirname(os.path.realpath(__file__))

        loss_dict = read_pickle_file(file_paths[-1])

    # Process the loaded data (replace this with your actual data processing logic)
    if loss_dict is not None:
        x_axis = loss_dict.values() #prendo solo i valori del dizionario
        x_axis = list(x_axis)

        y_axis = np.arange(0, len(loss_dict), 0.01)
 
        print_loss(x_axis, y_axis )
