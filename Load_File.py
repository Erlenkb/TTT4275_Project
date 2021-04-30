import pandas as pd
import numpy as np
from sklearn.datasets import load_iris



# Function creating train and test DataFrames for each species
def create_train_test_arrays(filename, training_start, training_end, testing_start, testing_end):
    # Import the iris csv data to a pandas DataFrame
    iris_df = pd.read_csv(filename)

    #Create a DataFrame for each species
    setosa_df = iris_df[iris_df['Species'].str.match('Iris-setosa')]
    versicolor_df = iris_df[iris_df['Species'].str.match('Iris-versicolor')]
    virginica_df = iris_df[iris_df['Species'].str.match('Iris-virginica')]

    # Remove species column from each DataFrame
    no_string_setosa_df = setosa_df.drop('Species', axis=1)
    no_string_versicolor_df = versicolor_df.drop('Species', axis=1)
    no_string_virginica_df = virginica_df.drop('Species', axis=1)

    # Divide species pandas DataFrames into numpy arrays for training and testing
    setosa_training_array = no_string_setosa_df.iloc[training_start: training_end].to_numpy()
    setosa_testing_array = no_string_setosa_df.iloc[testing_start: testing_end].to_numpy()

    versicolor_training_array = no_string_versicolor_df.iloc[training_start: training_end].to_numpy()
    versicolor_testing_array = no_string_versicolor_df.iloc[testing_start: testing_end].to_numpy()

    virginica_training_array = no_string_virginica_df.iloc[training_start: training_end].to_numpy()
    virginica_testing_array = no_string_virginica_df.iloc[testing_start: testing_end].to_numpy()

    return setosa_training_array, setosa_testing_array, versicolor_training_array, versicolor_testing_array, virginica_training_array, virginica_testing_array

def Create_Two_Training_And_Test_Sets(filename):
    Data_Set_Training = []
    Data_Set_Testing = []
    setosa_training_array, setosa_testing_array, versicolor_training_array, versicolor_testing_array, virginica_training_array, virginica_testing_array = create_train_test_arrays(
        filename, 0, 30, 30, 50)

    Training_Set_1 = [setosa_training_array, versicolor_training_array, virginica_training_array]
    Testing_Set_1 = [setosa_testing_array, versicolor_testing_array, virginica_testing_array]

    setosa_training_array, setosa_testing_array, versicolor_training_array, versicolor_testing_array, virginica_training_array, virginica_testing_array = create_train_test_arrays(
        filename, 20, 50, 0, 20)

    Training_Set_2 = [setosa_training_array, versicolor_training_array, virginica_training_array]
    Testing_Set_2 = [setosa_testing_array, versicolor_testing_array, virginica_testing_array]

    """
    for n in setosa_training_array:
        Data_Set_Training += n
    for n in versicolor_training_array:
        Data_Set_Training += n
    for n in virginica_training_array:
        Data_Set_Training += n
    for n in setosa_testing_array:
        Data_Set_Testing += n
    for n in versicolor_training_array:
        Data_Set_Testing += n
    for n in virginica_testing_array:
        Data_Set_Training += n
    """
    return Data_Set_Testing, Data_Set_Training


def split_training_and_test(file_location_1,file_location_2,file_location_3,split,Swap_To_Zero):
    """
    :param file_location_1:
    :param file_location_2:
    :param file_location_3:
    :param split:
    :param Swap_To_Zero: 0 = SepalLength, 1 = SepalWidth, 2 = PetalLength,3 = PetalWidth : Set 1 for feature you want, binary
    :return:
    """

    files = [file_location_1,file_location_2,file_location_3]

    data_set_training = [[],[],[],[]]
    data_set_testing = [[],[],[],[]]

    for file in files:
        data_set = [[], [], [], []]



        temp_store = np.loadtxt(file,delimiter =",")
        for element in temp_store:

            sepal_length = element[0]
            sepal_width = element[1]
            petal_length = element[2]
            petal_width = element[3]

            if(Swap_To_Zero[0]):
                data_set[0].append(sepal_length)
            else:
                data_set[0].append(0)
            if (Swap_To_Zero[1]):
                data_set[1].append(sepal_width)
            else:
                data_set[1].append(0)
            if (Swap_To_Zero[2]):
                data_set[2].append(petal_length)
            else:
                data_set[2].append(0)
            if (Swap_To_Zero[3]):
                data_set[3].append(petal_width)
            else:
                data_set[3].append(0)

        if split >= 25:
            for i in range(4):
                data_set_training[i] +=data_set[i][:split]
                data_set_testing[i] +=data_set[i][split:]
        else:
            for i in range(4):
                data_set_testing[i] += data_set[i][:split]
                data_set_training[i] += data_set[i][split:]


    data_set_testing = np.array(data_set_testing)
    data_set_training = np.array(data_set_training)
    return data_set_training, data_set_testing