import pandas as pd
import numpy as np

def predict(features, weights):
    predictions = np.dot(features, weights)
    return predictions


def update_weight(features, targets, weights, learingRate):

    x1 = features[:,0]
    x2 = features[:,1]
    x3 = features[:,2]

    for i in range(25000):
        predictions = predict(features, weights)

        d_w1 = -x1*(targets - predictions)
        d_w2 = -x2*(targets - predictions)
        d_w3 = -x3*(targets - predictions)

        weights[0][0] -= (learingRate * np.mean(d_w1))
        weights[1][0] -= (learingRate * np.mean(d_w2))
        weights[2][0] -= (learingRate * np.mean(d_w3))

        # if i & 1000 == 0:
        #     print("weight : ", weights)

    predictions = predict(features, weights)
    # print("predictions : ", predictions)
    cnt = [0] * 3
    
    for i in range(len(predictions)):
        if predictions[i] < 1.5:
            cnt[0] = cnt[0] + 1
        elif predictions[i] < 2.5:
            cnt[1] = cnt[1] + 1
        else:
            cnt[2] = cnt[2] + 1

    print("cnt : ", cnt)

    return weights


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
            # print("Good!")
            # print(words[0], " ", words[1], " ", words[2], " ", words[3])
            iris.append([float(words[0]), float(words[1]), float(words[2]), float(words[3]), "setosa"])
        elif "Iris-versicolor" in words[4]:
            iris.append([float(words[0]), float(words[1]), float(words[2]), float(words[3]), "versicolor"])
        elif "Iris-virginica" in words[4]:
            iris.append([float(words[0]), float(words[1]), float(words[2]), float(words[3]), "virginica"])

        
            
    f1.close()

    # data 분리

    df = pd.DataFrame(iris, columns = ['Sepal-Length', 'Sepal-Width','Petal-Length','Petal-Width', 'species'])
    
    for i in range(4):
        for j in range(i):
            # j < i, j가 x, i가 y
            x_data = []
            y_data = []
            x1_data = []
            y1_data = []
            
            for k in range(df.shape[0]):
                x_data.append([df.ix[k][i], df.ix[k][j], 1])
                if "setosa" in df.ix[k][4]:
                    y_data.append(1)
                elif "versicolor" in df.ix[k][4]:
                    y_data.append(2)
                    y1_data.append(2)
                    x1_data.append([df.ix[k][i], df.ix[k][j], 1])
                elif "virginica" in df.ix[k][4]:
                    y_data.append(2)
                    y1_data.append(3)
                    x1_data.append([df.ix[k][i], df.ix[k][j], 1])
            weights = [[0.0], [0.0], [0.0]]
            weights1 = [[0.0], [0.0], [0.0]]
            X = np.array(x_data)
            X1 = np.array(x1_data)
            targets = np.array(y_data)
            targets1 = np.array(y1_data)
            
            w1 = update_weight(X, targets, weights, 0.0001)
            w2 = update_weight(X1, targets1, weights1, 0.0001)
            print("w1 : ", w1)
            print("w2 : ", w2)
            print()
            
            
    

if __name__ == "__main__":
    main()
    
    