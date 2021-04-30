import numpy as np
import MSE_Trainer as Mt
import Load_File as Lf

def Sigmoid(x_k,W):
    z_k = np.matmul(W,x_k)
    g_k = 1 / (1 + np.exp(-z_k))
    return g_k



def Gradient_MSE(Training_Data_set1, W, N_Classes, Training_Const):
    MSE = 0


    for Class in range(N_Classes):
        Upper_Limit = (Class + 1) * Training_Const
        Lower_Limit = Class * Training_Const
        Data_Set = np.transpose(Training_Data_set1[:,Lower_Limit:Upper_Limit])
        Target_Vector = np.zeros((N_Classes, 1))
        Target_Vector[Class] = 1
        Target_Vector = np.transpose(Target_Vector)
        temp = []

        for x in Data_Set:
            g_k = Sigmoid(x,W)
            x = np.array([x])

            MSE = MSE + ((g_k-Target_Vector).dot(g_k.dot(1-g_k))) * x.T
            temp.append(MSE)
    return MSE, temp

def Train_Weight_Matrix(Iterations,N_Classes, Elements, Alpha, Data_Set, Training_Const=30):
    W = np.zeros((N_Classes, Elements))
    MSE_Temp = []
    for n in range(Iterations):
        gradient_MSE, MSE_Temp = Gradient_MSE(Data_Set, W, N_Classes, Training_Const)
        #MSE_Temp.append(gradient_MSE)
        W = W - Alpha * gradient_MSE.T

    return W, MSE_Temp


def Test_Classifier(W, Data_Set_Test, N_Classes, Training_Const):
    Confusion_Matrix = np.zeros((3,3))
    for Class in range(N_Classes):
        Upper_Limit = (Class + 1) * Training_Const
        Lower_Limit = Class * Training_Const
        Data_Set = np.transpose(Data_Set_Test[:,Lower_Limit:Upper_Limit])
        temp = []

        for x in Data_Set:
            g = np.matmul(W,x)
            i = np.argmax(g)
            Confusion_Matrix[Class][i] = Confusion_Matrix[Class][i] +1
        Error = 1 - (Confusion_Matrix[0][0] + Confusion_Matrix[1][1] + Confusion_Matrix[2][2]) / np.sum(Confusion_Matrix)
    return Confusion_Matrix, Error

def Get_MSE_Gradient(Training_Set, N_Classes, Training_Const=30, Features=4):
    W = np.ones((30, 4))
    MSE_Gradient = np.zeros((N_Classes, Features))
    MSE = 0


    for Class in range(N_Classes):
        Upper_Limit = (Class+1)*30
        Lower_Limit = Class*30
        Target_Vector = np.zeros((N_Classes,1))

        Target_Vector[Class] = 1
        x_add_ones = np.ones((Features, Training_Const))
        for x in range(Features):
            x_add_ones[x] = Training_Set[x,Lower_Limit:Upper_Limit]

        x_k = np.zeros((Features,1))
        for i in range(Training_Const):
            for j in range(Features):
                Mid = x_add_ones[j,i]
                x_k[j] = x_add_ones[j,i]
            g_k = Sigmoid(x_k,W)

            Gradient_gk_MSE = g_k[0]-Target_Vector
            Gradient_z_k = g_k * (1-g_k)

            #MSE_Gradient += np.matmul(Gradient_gk_MSE * Gradient_z_k,x_k.T)

            #MSE += np.matmul(Gradient_z_k.T,Gradient_z_k)

        MSE = MSE / 2
    return MSE, MSE_Gradient

Training_Set_1,Testing_Set_1 = Lf.split_training_and_test("class_1","class_2","class_3",30,[0,0,1,0])

#MSE1, MSE_Grad1 = Get_MSE_Gradient(Training_Set_1,3)

W,MSE_List = Train_Weight_Matrix(3000,3,4,0.005,Training_Set_1,30)
Confusion, ERR = Test_Classifier(W,Testing_Set_1,3,20)
print(Confusion)
print(ERR)
print(W)
#print(Train_Weight_Matrix(1500, 0.01, Training_Set_1))
#MSE, MSE_Grad = get_MSE_gradient(Training_Set_1,30,4)
#print(MSE1,"\t", MSE_Grad1)