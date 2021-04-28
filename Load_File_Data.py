import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.pyplot as plt

def Get_Data_From_File(file,duration_mode=0):
    """
    :param file: Filepath for the .dat file
    :param duration_mode: Define the duration mode for which formants you want. 0= steady state, 1=25%, 2=50%, 3=80%
    :return: Data_Set array
    """
    #Headers = ["Type", "Ms", "f0", "f1", "f2", "f3", "f4", "f11", "f12", "f13", "f21", "f22", "f23", "f31", "f32", "f33"]
    #Headers = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"]
    Headers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    if(duration_mode>0):
        Shift = 1
        Headers_Shift = 0
    else:
        Shift = -1
        Headers_Shift = 2
    Elements_Axis1 = 3 * (duration_mode + 1) + Shift
    Headers = Headers[Elements_Axis1: Elements_Axis1 + 3 + Headers_Shift]
    Headers.append(0)
    df = pd.read_csv(file,
                     sep="\s+",  # separator whitespace
                     index_col=False,
                     usecols=Headers,
                     header=None)

    Arr = df.to_numpy()
    return Arr[1:,:]


def split_At_Index(Data_Array,splitter, Splitt=False):
    """
    :param Data_Array: Data_Set to be split into wovels
    :param Splitter: b=boy, w=woman, m=man,g=girl
    :return:
    """
    #Data_Array = Data_Array.T
    Data_Set_Split = [[],[],[],[]]
    character1 = splitter
    #character2_3 = ItNumber
    #character4_5 = wovel
    vowel = ("ae","ah","aw","eh","er","ei","ih","iy","oa","oo","uh","uw")
    talker_Group = ("m","w","b","g")
    for x in Data_Array:
        if(x[0] != 1):
            verdi = []
            Name = x[0]
            verdi.append(Name)
            for n in range(1,len(x)):
                verdi.append(int(x[n]))

            Talker = talker_Group.index(Name[0])
            verdi = np.array(verdi)
            Data_Set_Split[Talker].append(verdi)
        else: print("Error!")
    data_Set_Vowel = [[], [], [], [], [], [], [], [], [], [], [], []]
    for x in Data_Set_Split:

        x = np.array(x)
        if(Splitt):
            for element in x:
                name = element[0]
                if(name[0] == splitter):
                    TempString = name[3]+name[4]
                    Vowel_Index = vowel.index(TempString)
                    data_Set_Vowel[Vowel_Index].append(element)
    count = 0
    FinalShit = []
    for x in data_Set_Vowel:
        Temp = np.array(x)
        data_Set_Vowel[0] = Temp
        FinalShit.append(Temp)
        count += 1
    return x, FinalShit

#x, Final = split_At_Index(W,"m",True)



def String_Array_To_Int_Array(data_Set):
    datatemp = data_Set
    axis0 = len(datatemp[0])
    #axis1 = datatemp.shape[1]
    Mean_Vector = np.zeros((len(datatemp),len(datatemp[0])))
    for x in range(len(datatemp)):
        for y in range(len(datatemp[0])):
            temp = datatemp[x][y]
            Mean_Vector[x][y] = int(temp)
    return Mean_Vector

def sample_mean_vector(data):
    """
    :param data:
    :return:
    """
    # Find mean of f0(3), F1(4,8,11,14), F2(5,9,12,15),
    # F3(6,10,13,16) and F4(7)
    # Returns an array with mean values for each measurement point
    avg_vector = []
    for key, value in data.items():
        data_mean = np.mean(value,axis=0)
        avg_vector.append(data_mean)
    return np.array(avg_vector)

def cov_matrix(data_set):
    N = len(data_set[0])
    size = len(data_set)
    mean = sample_mean_vector(data_set)


def sample_covariance_matrix(measurements, Diagonal=False):
    # Find the covariance matrix of measurements matrix.
    # rowvar = False, as each column is a variable with one corresponding observation in each row.
    cov_vector = []
    cov_vector_temp = []
    avg_cov_map = {}
    for key, value in measurements.items():
        covMatrix = np.cov(value, rowvar=False)
        if(Diagonal):
            covMatrix = np.diag(np.diag(covMatrix))
        cov_vector_temp.append(covMatrix)
        #avg = np.mean(value,axis=0)
        #avg_cov_map[key] = avg
        #avg_cov_map[key].append(covMatrix)
    cov_vector.append(cov_vector_temp)
    return cov_vector


def extract_classes_map(filename):
    class_map = {}

    with open(filename, "r") as file:
        lines = file.read().split("\n")
        try:
            for line in lines:
                line = line.split(" ")
                x_i = []
                for element in range(1, len(line)):
                    if line[element] != '':
                        x_i.append(line[element])
                Wovel = str(line[0][-2] + line[0][-1])
                if Wovel not in class_map:
                    class_map[Wovel] = [x_i]
                else:
                    class_map[Wovel].append(x_i)
        except IndexError:
            print("End of File")

    return class_map


def equal_representation(dataset,duration_mode,N_Test_Men=26,N_Test_Women=27,N_Test_Boy=7,N_Test_Girl=9,N_Train_Men=20,N_Train_Women=20,N_Train_Boy=20,N_Train_Girl=10):
    """
    Split the given dataset into two arrays containing test and training data.
    :param dataset: Dataset to split
    :param duration_mode: Define which state to use. 0=steady state,1=25%,2=50%,3=80%,4=all
    :param N_Test_Men: Number of men samples for the testing set
    :param N_Test_Women: Number of women samples for the testing set
    :param N_Test_Boy: Number of boy samples for the testing set
    :param N_Test_Girl: Number of girl samples for the testing set
    :param N_Train_Men: Number of men for the training set
    :param N_Train_Women: Number of women for the training set
    :param N_Train_Boy: Number of boy samples for the training set
    :param N_Train_Girl: Number of girl samples for the training set
    :return:
    """
    test_set = []
    training_set = []
    Transform = 1
    if (duration_mode > 0):
        Shift = 1
        Headers_Shift = 0

    else:
        Shift = -1
        Headers_Shift = 2
    if (duration_mode == 4):
        Transform = 0
        duration_mode = 3
    lower_limit = (3 * (duration_mode + 1)) * Transform + Shift
    upper_limit = 3 * (duration_mode + 1) + Shift + 3 + Headers_Shift

    if(N_Train_Men != 0):
        for man in range(0,N_Train_Men): #20
            training_set.append(dataset[man][lower_limit: upper_limit])
    if(N_Test_Men != 0):
        for man in range(20,20 + N_Test_Men): #26
            test_set.append(dataset[man][lower_limit: upper_limit])
    if(N_Train_Boy != 0):
        for boy in range(93, 93 + N_Train_Boy):  # 20
            training_set.append(dataset[boy][lower_limit: upper_limit])
    if(N_Test_Boy != 0):
        for boy in range(113,113 + N_Test_Boy): #7
            test_set.append(dataset[boy][lower_limit: upper_limit])
    if(N_Train_Women != 0):
        for woman in range(46,46 + N_Train_Women): #20
            training_set.append(dataset[woman][lower_limit: upper_limit])
    if(N_Test_Women != 0):
        for woman in range(66,66 + N_Test_Women): #27
            test_set.append(dataset[woman][lower_limit: upper_limit])
    if(N_Train_Girl != 0):
        for girl in range(120,120 + N_Train_Girl): #10
            training_set.append(dataset[girl][lower_limit: upper_limit])
    if(N_Test_Girl != 0):
        for girl in range(130,130 + N_Test_Girl): #9
            test_set.append(dataset[girl][lower_limit: upper_limit])
    test_set = String_Array_To_Int_Array(test_set)
    training_set = String_Array_To_Int_Array(training_set)

    return training_set,test_set



def generate_x(filename,duration = 0,N_Test_Men=26,N_Test_Women=27,N_Test_Boy=7,N_Test_Girl=9,N_Train_Men=20,N_Train_Women=20,N_Train_Boy=20,N_Train_Girl=10):
    """
    :param filename: The Used .txt file for formants data
    :param duration: Define the wanted duration. 0=Steady state, 1=25%, 2=50%, 3=80%
    :param N_Test_Men: Number of men samples for the testing set
    :param N_Test_Women: Number of women samples for the testing set
    :param N_Test_Boy: Number of boy samples for the testing set
    :param N_Test_Girl: Number of girl samples for the testing set
    :param N_Train_Men: Number of men for the training set
    :param N_Train_Women: Number of women for the training set
    :param N_Train_Boy: Number of boy samples for the training set
    :param N_Train_Girl: Number of girl samples for the training set
    :return: two dictionaries containing test data and train data
    """

    test_map = {}
    train_map={}
    classes_map = extract_classes_map(filename)
    for sound in classes_map:
        training_Set,test_Set = equal_representation(classes_map[sound],duration,N_Test_Men,N_Test_Women,N_Test_Boy,N_Test_Girl,N_Train_Men,N_Train_Women,N_Train_Boy,N_Train_Girl)
        test_map[sound] = np.array(test_Set)
        train_map[sound] = np.array(training_Set)

    return train_map,test_map

Train, Test = generate_x("Temp.txt",1,N_Test_Women=0,N_Test_Girl=0,N_Test_Boy=0,N_Train_Women=0,N_Train_Girl=0,N_Train_Boy=0)

def train_and_test_gm(file,duration,diag=False):
    Train, Test = generate_x(file, duration)
    for TrainOrTest in range(2):
        if(TrainOrTest < 1): File = Train
        else: File = Test
        avg_matrix = sample_mean_vector(File)

        vowel = ("ae", "ah", "aw", "eh", "er", "ei", "ih", "iy", "oa", "oo", "uh", "uw")

        cov_matrix = sample_covariance_matrix(File,diag)
        cov_matrix = cov_matrix[0]

        #probability_vector = np.empty((12,1))
        probability_vector = np.empty((12,1))
        correct = 0
        wrong = 0
        total = 0
        confusion_matrix = np.zeros((12,12))
        diagonal = 0
        for key, value in File.items():
            for sample in range(len(value)):
                jdfjhdfd = avg_matrix.shape[0]
                for i in range(avg_matrix.shape[0]):
                    probability = multivariate_normal.pdf(value[sample],mean=avg_matrix[i],cov=cov_matrix[i],allow_singular=True)
                    probability_vector[i] = probability

                pred_index = np.argmax(probability_vector)
                pred_vowel = vowel[pred_index]
                total += 1
                if key == pred_vowel: correct += 1
                else: wrong += 1
                confusion_matrix[diagonal][pred_index] += 1
            diagonal += 1

        print(confusion_matrix)
        total = 0
        correct_vec = []
        error_rate_vec = []
        for x in range(12):
            individual_set = sum(confusion_matrix[x,:])
            correct_vec.append(confusion_matrix[x][x])
            error_rate_vec.append(round((1 - confusion_matrix[x][x] / individual_set) * 100 ,2))
            total += individual_set
        total_error = (1 - (sum(correct_vec) / total)) * 100

        print(error_rate_vec, "\n", total_error)
    return confusion_matrix, error_rate_vec, total_error

#confusion, error_rate_vec,total_error = train_and_test_gm("Temp.txt",4,False)


def train_gmm(training_data_set, n_components):
    gmm_list = []
    probs = []
    for key, value in training_data_set.items():
        gmm = GMM(n_components=n_components, covariance_type="diag")
        gmm.fit(value)
        gmm_list.append(gmm)
        #probs.append(gmm.predict_proba(value))
    return gmm_list

def test_gmm(training_data_set, gmm_list,n_components):
    vowel = ("ae", "ah", "aw", "eh", "er", "ei", "ih", "iy", "oa", "oo", "uh", "uw")
    probabilites = np.zeros((len(training_data_set),1))
    i = 0
    confusion_matrix = np.zeros((12, 12))
    correct = 0
    wrong = 0
    total = 0

    diagonal = 0
    for key, value in training_data_set.items():
        for sample in range(len(value)):
            probabilites = np.zeros((len(training_data_set), 1))
            for j in range(len(Train)):
                gmm = gmm_list[j]
                for n in range(n_components):
                    N = multivariate_normal(mean=gmm.means_[n], cov=gmm.covariances_[n], allow_singular=True)
                    gmgmgmgmg = gmm.weights_[n]
                    sdkjfklsdjafjklsd = value[sample]
                    probabilites[j] += gmm.weights_[n] * N.pdf(value[sample])
            predictions = np.argmax(probabilites)
            pred_vowel = vowel[predictions]

            if(pred_vowel == key): correct += 1
            else: wrong += 1
            total += 1
            confusion_matrix[diagonal][predictions] += 1
        diagonal += 1

    print(confusion_matrix)
    total = 0
    correct_vec = []
    error_rate_vec = []
    for x in range(12):
        individual_set = sum(confusion_matrix[x, :])
        correct_vec.append(confusion_matrix[x][x])
        error_rate_vec.append(round((1 - confusion_matrix[x][x] / individual_set) * 100, 2))
        total += individual_set
    total_error = (1 - (sum(correct_vec) / total)) * 100
    false_matrix = np.empty((12,12))
    for x in range(12):
        for y in range(12):
            if(confusion_matrix[x][y] != 0): false_matrix[x][y] = False
            else: false_matrix[x][y] = True
    print(error_rate_vec, "\n", total_error)
    cols = ["ae", "ah", "aw", "eh", "er", "ei", "ih", "iy", "oa", "oo", "uh", "uw"]
    idx = ["ae", "ah", "aw", "eh", "er", "ei", "ih", "iy", "oa", "oo", "uh", "uw"]
    cMatrix_df = pd.DataFrame(confusion_matrix, index=idx, columns=cols)
    sns.heatmap(cMatrix_df, annot=True, cmap="Blues",cbar=True, mask=false_matrix, linecolor="lightsteelblue",linewidths=0.05)
    plt.show()
    return confusion_matrix

w = test_gmm(Test,train_gmm(Test,2),2)












#gmm_list, probs = train_gmm(Train,2)
#print(gmm_list,"\n",probs)
#test_gmm(Train,train_gmm(Train,2))