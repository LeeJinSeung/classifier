import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random

def main():
    # 파일 오픈
    f1 = open("iris.data", 'r')

    iris = []

    while(True):
        line = f1.readline()
        if not line: 
            break
        if(line == '\n'):
            continue

        # ,를 기준으로 split 
        # sepal length, width, petal length, width, class
        words = line.split(',')
        
        if "Iris-setosa" in words[4]:
            iris.append([float(words[0]), float(words[1]), float(words[2]), float(words[3]), "setosa"])
        elif "Iris-versicolor" in words[4]:
            iris.append([float(words[0]), float(words[1]), float(words[2]), float(words[3]), "versicolor"])
        elif "Iris-virginica" in words[4]:
            iris.append([float(words[0]), float(words[1]), float(words[2]), float(words[3]), "virginica"])
 
            
    f1.close()

    
    df = pd.DataFrame(iris, columns = ['Sepal-Length', 'Sepal-Width','Petal-Length','Petal-Width', 'species'])

    # data shuffle
    df = shuffle(df)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # print(df)


    # dataSet : Sepal-Length, Sepal-Width, Petal-Length, Petal-Width, species
    dataSet = []

    cnt = -1
    
    # data를 5가지 set으로 구분(각 set마다 종류별로 10개씩)
    for i in range (0, df.shape[0]):
        temp = []
        if(i % 30 == 0):
            cnt = cnt + 1
            dataSet.append([])
        for j in range (0, df.shape[1]):
            temp.append(df.ix[i][j])
        dataSet[cnt].append(temp)


    precision = [0.0, 0.0, 0.0]
    recall = [0.0, 0.0, 0.0]
    
    # test set을 어떤 것을 이용할 것인가
    # Petal-Width를 이용하여 pmf로 판단.
    for i in range(5):
        # i가 test set

         # train을 위한 0으로 초기화 된 크기가 100인 배열 생성
        setosa = [[0 for x in range(4)] for y in range(100)]
        versicolor = [[0 for x in range(4)] for y in range(100)]
        virginica = [[0 for x in range(4)] for y in range(100)]

        # 전체 train한 갯수(index를 기준으로)
        train = [[0 for x in range(4)] for y in range(100)]

        result = [[0 for x in range(3)] for y in range(3)]
        

        # train한 iris의 갯수
        cnt = [0] * 3

        # prior
        prior = [0] * 3


        # training
        for j in range(30):
            for k in range(i):
                idx = int(dataSet[k][j][0] * 10)
                idx1 = int(dataSet[k][j][1] * 10)
                idx2 = int(dataSet[k][j][2] * 10)
                idx3 = int(dataSet[k][j][3] * 10)
                if "setosa" in dataSet[k][j][4]:
                    setosa[idx][0] = setosa[idx][0] + 1
                    setosa[idx1][1] = setosa[idx1][1] + 1
                    setosa[idx2][2] = setosa[idx2][2] + 1
                    setosa[idx3][3] = setosa[idx3][3] + 1
                    cnt[0] = cnt[0] + 1
                elif "versicolor" in dataSet[k][j][4]:
                    versicolor[idx][0] = versicolor[idx][0] + 1
                    versicolor[idx1][1] = versicolor[idx1][1] + 1
                    versicolor[idx2][2] = versicolor[idx2][2] + 1
                    versicolor[idx3][3] = versicolor[idx3][3] + 1
                    cnt[1] = cnt[1] + 1
                elif "virginica" in dataSet[k][j][4]:
                    virginica[idx][0] = virginica[idx][0] + 1
                    virginica[idx1][1] = virginica[idx1][1] + 1
                    virginica[idx2][2] = virginica[idx2][2] + 1
                    virginica[idx3][3] = virginica[idx3][3] + 1
                    cnt[2] = cnt[2] + 1

            for k in range(i + 1, 5):
                idx = int(dataSet[k][j][0] * 10)
                idx1 = int(dataSet[k][j][1] * 10)
                idx2 = int(dataSet[k][j][2] * 10)
                idx3 = int(dataSet[k][j][3] * 10)
                if "setosa" in dataSet[k][j][4]:
                    setosa[idx][0] = setosa[idx][0] + 1
                    setosa[idx1][1] = setosa[idx1][1] + 1
                    setosa[idx2][2] = setosa[idx2][2] + 1
                    setosa[idx3][3] = setosa[idx3][3] + 1
                    cnt[0] = cnt[0] + 1
                elif "versicolor" in dataSet[k][j][4]:
                    versicolor[idx][0] = versicolor[idx][0] + 1
                    versicolor[idx1][1] = versicolor[idx1][1] + 1
                    versicolor[idx2][2] = versicolor[idx2][2] + 1
                    versicolor[idx3][3] = versicolor[idx3][3] + 1
                    cnt[1] = cnt[1] + 1
                elif "virginica" in dataSet[k][j][4]:
                    virginica[idx][0] = virginica[idx][0] + 1
                    virginica[idx1][1] = virginica[idx1][1] + 1
                    virginica[idx2][2] = virginica[idx2][2] + 1
                    virginica[idx3][3] = virginica[idx3][3] + 1
                    cnt[2] = cnt[2] + 1

        for j in range(3):
            prior[j] = float(cnt[j] / (cnt[0] + cnt[1] + cnt[2]))


        for j in range(100):
            for k in range(4):
                train[j][k] = setosa[j][k] + versicolor[j][k] + virginica[j][k]


        for j in range(30):
            
            p = [prior[0], prior[1], prior[2]]
            choice = 0

            for k in range(4):
                idx = int(dataSet[i][j][k] * 10)
                
                # train된 정보가 없는 경우 random으로 채택
                if train[idx][k] == 0:
                    p[0] = 0.0
                    p[1] = 0.0
                    p[2] = 0.0
                    break
                else:
                    if setosa[idx][k] != 0:
                        p[0] = p[0] * float((setosa[idx][k] / train[idx][k]))
                    else:
                        p[0] = 0.0

                    if versicolor[idx][k] != 0:
                        p[1] = p[1] * float((versicolor[idx][k] / train[idx][k]))
                    else:
                        p[1] = 0.0

                    if virginica[idx][k] != 0:
                        p[2] = p[2] * float((virginica[idx][k] / train[idx][k]))
                    else:
                        p[2] = 0.0

            if p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0:
                choice = random.randrange(0,3)
            else:
                if p[0] < p[1]:
                    if p[1] < p[2]:
                        choice = 2
                    else:
                        choice = 1
                else:
                    if p[0] < p[2]:
                        choice = 2
                    else:
                        choice = 0

            if "setosa" in dataSet[i][j][4]:
                if choice == 0:
                    result[0][0] = result[0][0] + 1
                elif choice == 1:
                    result[1][0] = result[1][0] + 1
                elif choice == 2:
                    result[2][0] = result[2][0] + 1
            elif "versicolor" in dataSet[i][j][4]:
                if choice == 0:
                    result[0][1] = result[0][1] + 1
                elif choice == 1:
                    result[1][1] = result[1][1] + 1
                elif choice == 2:
                    result[2][1] = result[2][1] + 1
            elif "virginica" in dataSet[i][j][4]:
                if choice == 0:
                    result[0][2] = result[0][2] + 1
                elif choice == 1:
                    result[1][2] = result[1][2] + 1
                elif choice == 2:
                    result[2][2] = result[2][2] + 1

        for j in range(3):
            precision[j] = precision[j] + float(result[j][j] / (result[j][0] + result[j][1] + result[j][2]))
            recall[j] = recall[j] + float(result[j][j] / (result[0][j] + result[1][j] + result[2][j]))


    for i in range(3):
        precision[i] = precision[i] / 5;
        recall[i] = recall[i] / 5;

    print("Setosa precision : ", precision[0], " recall : ", recall[0])
    print("versicolor precision : ", precision[1], " recall : ", recall[1])
    print("virginica precision : ", precision[2], " recall : ", recall[2])


if __name__ == "__main__":
    main()
    