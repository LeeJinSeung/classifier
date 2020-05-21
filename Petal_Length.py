import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_style("white")


def main():
    # 파일 오픈
    f1 = open("iris.data", 'r')

    setosa = []
    versicolor = []
    virginica = []


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
            setosa.append([float(words[0]), float(words[1]), float(words[2]), float(words[3])])
        elif "Iris-versicolor" in words[4]:
            versicolor.append([float(words[0]), float(words[1]), float(words[2]), float(words[3])])
        elif "Iris-virginica" in words[4]:
            virginica.append([float(words[0]), float(words[1]), float(words[2]), float(words[3])])
            
    f1.close()

    # data 분리
    df = pd.DataFrame(setosa, columns = ['Sepal-Length', 'Sepal-Width','Petal-Length','Petal-Width'])
    df1 = pd.DataFrame(versicolor, columns = ['Sepal-Length', 'Sepal-Width','Petal-Length','Petal-Width'])
    df2 = pd.DataFrame(virginica, columns = ['Sepal-Length', 'Sepal-Width','Petal-Length','Petal-Width'])

    
    sns.distplot(df['Petal-Length'], hist = True, kde = False, color='red')
    sns.distplot(df1['Petal-Length'], hist = True, kde = False, color='green')
    sns.distplot(df2['Petal-Length'], hist = True, kde = False, color='blue')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()