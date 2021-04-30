import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Load_File as Lf
import Math_Functions as Mf


def Training_Linear_Classifier(Training_Set, Iterations, Alpha, Training_Const=30, Features=4):
    W = np.ones((Training_Const,Features))
    MSE_Temp = []
    W_Temp = []

    print("Begin Training *DUM DUM DUUUUUUM")
    Count = 0
    MSE_Delta = 1
    while(MSE_Delta > 0.0001):
        MSE,Gradient_MSE = Mf.Get_MSE_Gradient(Training_Set, W, Training_Const, Features)
        if(Count > 0):
            MSE_Delta = np.abs((MSE_Temp[-1] - MSE[0][0]) / 2)
        MSE_Temp.append(MSE[0][0])
        W -= Alpha * Gradient_MSE
        W_Temp.append(W)
        if(Count > 100000):
            print("Iteration Error: Exceeded maximum number of iterations")
            print("MSE_Delta was", MSE_Delta)
            print("Last computed MSE was", MSE[0][0])
            return
        Count += 1

    print("Training finished. Iteration Count: ", Count)

    return W, W_Temp, MSE_Temp


