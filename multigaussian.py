import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def func(a1, a2, a3, b1, b2, b3, c1, c2, c3):
    return a1*b2*c3 + a2*b3*c1 + a3*b1*c2 - a1*b3*c2 - a2*b1*c3 - a3*b2*c1

def determinant(cov):
    a = cov[0][0]
    b = -cov[0][1]
    c = cov[0][2]
    d = -cov[0][3]

    x = a * func(cov[1][1], cov[1][2], cov[1][3], cov[2][1], cov[2][2], cov[2][3], cov[3][1], cov[3][2], cov[3][3])
    y = b * func(cov[1][0], cov[1][2], cov[1][3], cov[2][0], cov[2][2], cov[2][3], cov[3][0], cov[3][2], cov[3][3])
    z = c * func(cov[1][0], cov[1][1], cov[1][3], cov[2][0], cov[2][1], cov[2][3], cov[3][0], cov[3][1], cov[3][3])
    w = d * func(cov[1][0], cov[1][1], cov[1][2], cov[2][0], cov[2][1], cov[2][2], cov[3][0], cov[3][1], cov[3][2])

    return x+y+z+w

def inverseMatrix(cov, det):
    a11 = cov[0][0]
    a12 = cov[0][1]
    a13 = cov[0][2]
    a14 = cov[0][3]

    a21 = cov[1][0]
    a22 = cov[1][1]
    a23 = cov[1][2]
    a24 = cov[1][3]

    a31 = cov[2][0]
    a32 = cov[2][1]
    a33 = cov[2][2]
    a34 = cov[2][3]

    a41 = cov[3][0]
    a42 = cov[3][1]
    a43 = cov[3][2]
    a44 = cov[3][3]

    b11 = func(a22, a23, a24, a32, a33, a34, a42, a43, a44)
    b12 = -func(a12, a13, a14, a32, a33, a34, a42, a43, a44)
    b13 = func(a12, a13, a14, a22, a23, a24, a42, a43, a44)
    b14 = -func(a12, a13, a14, a22, a23, a24, a32, a33, a34)

    b21 = -func(a21, a23, a24, a31, a33, a34, a41, a43, a44)
    b22 = func(a11, a13, a14, a31, a33, a34, a41, a43, a44)
    b23 = -func(a11, a13, a14, a21, a23, a24, a41, a43, a44)
    b24 = func(a11, a13, a14, a21, a23, a24, a31, a33, a34)
    
    b31 = func(a21, a22, a24, a31, a32, a34, a41, a42, a44)
    b32 = -func(a11, a12, a14, a31, a32, a34, a41, a42, a44)
    b33 = func(a11, a12, a14, a21, a22, a24, a41, a42, a44)
    b34 = -func(a11, a12, a14, a21, a22, a24, a31, a32, a34)

    b41 = -func(a21, a22, a23, a31, a32, a33, a41, a42, a43)
    b42 = func(a11, a12, a13, a31, a32, a33, a41, a42, a43)
    b43 = -func(a11, a12, a13, a21, a22, a23, a41, a42, a43)
    b44 = func(a11, a12, a13, a21, a22, a23, a31, a32, a33)

    b = np.array([[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34], [b41, b42, b43, b44]])
    return 1/det * b



def pdf(x, mean, det, inverse):
    return (1/ (((2 * np.pi)*(2 * np.pi)) * np.sqrt(det))) * np.exp(-1/2 * np.dot(np.dot((x-mean), inverse), (x-mean).T))


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



    
    precision = [0.0, 0,0, 0.0]
    recall = [0.0, 0.0, 0.0]

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
        cov = [[[0 for x in range(4)] for y in range(4)] for z in range(3)]

        sum2 = [[[0 for x in range(4)] for y in range(4)] for z in range(3)]
        result = [[0 for x in range(3)] for y in range(3)]

        # 전체 train한 갯수(index를 기준으로)
        

        # train한 iris의 갯수
        cnt = [0, 0, 0]
        prior = [0, 0, 0]


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
                cov[j][k][k] = variance[j][k]

        for j in range(len(setosa)):
            for k in range(4):
                for q in range(k):
                    sum2[0][k][q] = sum2[0][k][q] + ((setosa[j][k] - mean[0][k]) * (setosa[j][q] - mean[0][q]))

        for j in range(len(versicolor)):
            for k in range(4):
                for q in range(k):
                    sum2[1][k][q] = sum2[1][k][q] + ((versicolor[j][k] - mean[1][k]) * (versicolor[j][q] - mean[1][q]))

        for j in range(len(virginica)):
            for k in range(4):
                for q in range(k):
                    sum2[2][k][q] = sum2[2][k][q] + ((virginica[j][k] - mean[2][k]) * (virginica[j][q] - mean[2][q]))


        for j in range(3):
            for k in range(4):
                for q in range(k):
                    cov[j][k][q] = (sum2[j][k][q] / cnt[j])
                    cov[j][q][k] = cov[j][k][q]

        c1 = np.array(cov[0])
        c2 = np.array(cov[1])
        c3 = np.array(cov[2])

        det1 = determinant(c1)
        det2 = determinant(c2)
        det3 = determinant(c3)

        inverse1 = inverseMatrix(c1, det1)
        inverse2 = inverseMatrix(c2, det2)
        inverse3 = inverseMatrix(c3, det3)

        m1 = np.array(mean[0])
        m2 = np.array(mean[1])
        m3 = np.array(mean[2])

        # if i == 0:
        #     print(c1)
        #     print()
        #     print(det1)
        #     print()
        #     print(inverse1)

        for j in range(30):
            data = []
            for k in range(4):
                idx = float(dataSet[i][j][k])
                data.append(idx)

            d = np.array(data)

           
            normal1 = pdf(d, m1, det1, inverse1) * prior[0]
            normal2 = pdf(d, m2, det2, inverse2) * prior[1]
            normal3 = pdf(d, m3, det3, inverse3) * prior[2]

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
    
