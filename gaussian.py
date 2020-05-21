import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def pdf(x, mean, variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-((x-mean)*(x-mean))/(2*variance))

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



    
    precision = [0.0, 0,0, 0.0];
    recall = [0.0, 0.0, 0.0];

    # test set을 어떤 것을 이용할 것인가
    # Petal-Length, Petal-Width를 이용한다
    for i in range(5):
        # i가 test set

        # train을 위한 0으로 초기화 된 크기가 100인 배열 생성
        setosa = []
        versicolor = []
        virginica = []


        sum = [[0 for x in range(4)] for y in range(3)]
        mean = [[0 for x in range(4)] for y in range(3)]
        variance = [[0 for x in range(4)] for y in range(3)]

        result = [[0 for x in range(3)] for y in range(3)]

        prior = [0] * 3

        # 전체 train한 갯수(index를 기준으로)
        

        # train한 iris의 갯수
        cnt = [0, 0, 0]


        # training
        for j in range(30):
            for k in range(i):
                idx = float(dataSet[k][j][0])
                idx1 = float(dataSet[k][j][1])
                idx2 = float(dataSet[k][j][2])
                idx3 = float(dataSet[k][j][3])
                if "setosa" in dataSet[k][j][4]:
                    setosa.append([idx, idx1, idx2, idx3])
                    sum[0][0] = sum[0][0] + idx
                    sum[0][1] = sum[0][1] + idx1
                    sum[0][2] = sum[0][2] + idx2
                    sum[0][3] = sum[0][3] + idx3
                    cnt[0] = cnt[0] + 1
                elif "versicolor" in dataSet[k][j][4]:
                    versicolor.append([idx, idx1, idx2, idx3])
                    sum[1][0] = sum[1][0] + idx
                    sum[1][1] = sum[1][1] + idx1
                    sum[1][2] = sum[1][2] + idx2
                    sum[1][3] = sum[1][3] + idx3
                    cnt[1] = cnt[1] + 1
                elif "virginica" in dataSet[k][j][4]:
                    virginica.append([idx, idx1, idx2, idx3])
                    sum[2][0] = sum[2][0] + idx
                    sum[2][1] = sum[2][1] + idx1
                    sum[2][2] = sum[2][2] + idx2
                    sum[2][3] = sum[2][3] + idx3
                    cnt[2] = cnt[2] + 1

            for k in range(i + 1, 5):
                idx = float(dataSet[k][j][0])
                idx1 = float(dataSet[k][j][1])
                idx2 = float(dataSet[k][j][2])
                idx3 = float(dataSet[k][j][3])
                if "setosa" in dataSet[k][j][4]:
                    setosa.append([idx, idx1, idx2, idx3])
                    sum[0][0] = sum[0][0] + idx
                    sum[0][1] = sum[0][1] + idx1
                    sum[0][2] = sum[0][2] + idx2
                    sum[0][3] = sum[0][3] + idx3
                    cnt[0] = cnt[0] + 1
                elif "versicolor" in dataSet[k][j][4]:
                    versicolor.append([idx, idx1, idx2, idx3])
                    sum[1][0] = sum[1][0] + idx
                    sum[1][1] = sum[1][1] + idx1
                    sum[1][2] = sum[1][2] + idx2
                    sum[1][3] = sum[1][3] + idx3
                    cnt[1] = cnt[1] + 1
                elif "virginica" in dataSet[k][j][4]:
                    virginica.append([idx, idx1, idx2, idx3])
                    sum[2][0] = sum[2][0] + idx
                    sum[2][1] = sum[2][1] + idx1
                    sum[2][2] = sum[2][2] + idx2
                    sum[2][3] = sum[2][3] + idx3
                    cnt[2] = cnt[2] + 1

        for j in range(3):
            prior[j] = float(cnt[j] / (cnt[0] + cnt[1] + cnt[2]))

        # mean 설정
        for j in range(3):
            for k in range(4):
                mean[j][k] = float(sum[j][k] / cnt[j])
                sum[j][k] = 0

        for j in range(len(setosa)):
            for k in range(4):
                sum[0][k] = sum[0][k] + ((setosa[j][k] - mean[0][k]) * (setosa[j][k] - mean[0][k]))

        for j in range(len(versicolor)):
            for k in range(4):
                sum[1][k] = sum[1][k] + ((versicolor[j][k] - mean[1][k]) * (versicolor[j][k] - mean[1][k]))
        
        for j in range(len(virginica)):
            for k in range(4):
                sum[2][k] = sum[2][k] + ((virginica[j][k] - mean[2][k]) * (virginica[j][k] - mean[2][k]))

        # varience 설정
        for j in range(3):
            for k in range(4):
                variance[j][k] = float(sum[j][k] / cnt[j])


        for j in range(30):
            normal1 = prior[0];
            normal2 = prior[1];
            normal3 = prior[2];
            choice = 0
            for k in range(4):
                idx = float(dataSet[i][j][k])
                normal1 = normal1 * pdf(idx, mean[0][k], variance[0][k])
                normal2 = normal2 * pdf(idx, mean[1][k], variance[1][k])
                normal3 = normal3 * pdf(idx, mean[2][k], variance[2][k])

            if normal1 < normal2:
                if normal2 < normal3:
                    choice = 2
                else:
                    choice = 1
            else:
                if normal1 < normal3:
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
    