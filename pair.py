import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_style("white")


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
    
    sns.pairplot(df, hue="species")
    
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
    