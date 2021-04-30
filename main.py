import numpy as np
import MSE_Trainer as Mt
import Math_Functions as Mf
import Load_File as Lf




setosa_training_array, setosa_testing_array, versicolor_training_array, versicolor_testing_array, virginica_training_array, virginica_testing_array = Lf.create_train_test_arrays("iris.csv",0,30,30,50)

Training_Set_1 = [setosa_training_array,versicolor_training_array,virginica_training_array]
Testing_Set_1 = [setosa_testing_array,versicolor_testing_array,virginica_testing_array]

setosa_training_array, setosa_testing_array, versicolor_training_array, versicolor_testing_array, virginica_training_array, virginica_testing_array = Lf.create_train_test_arrays("iris.csv",20,50,0,20)

Training_Set_2 = [setosa_training_array,versicolor_training_array,virginica_training_array]
Testing_Set_2 = [setosa_testing_array,versicolor_testing_array,virginica_testing_array]


def LottoGen(Antall_Tall, Antall_Trekk):
    Temp = []
    while(len(Temp) < 4):
        PremieFordeling = np.random.randint(1,Antall_Tall, size=Antall_Trekk)
        Temp = list(dict.fromkeys(PremieFordeling))
    return Temp





####### Oppgaver -- Iris ########

Oppg1a = True
Oppg1b = True
Oppg1c = False
Oppg1d = False
Oppg1e = False


Oppg2a = False
Oppg2b = False
Oppg2c = False


##### 1a ##########

if(Oppg1a):
    Training_Set_1, Training_Set_2, Testing_Set_1, Testing_Set_2 = Lf.Create_Two_Training_And_Test_Sets("iris.csv")

##### 1b ##########

if(Oppg1b):
    W, W_Temp, MSE_Temp = Mt.Training_Linear_Classifier(Training_Set_1,10000, 0.1)
    print("W: ",W,"\t W_Temp",W_Temp,"\t MSE_Temp", MSE_Temp)