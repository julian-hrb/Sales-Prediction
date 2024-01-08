import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sales = pd.read_csv("sales.csv")
df = pd.DataFrame(sales)
date = sales["Year-Month"]
by_date = df.set_index("Year-Month")
by_sale = df.set_index("Sales")
x = np.array([202101,202102,202103,202104,202105,202106,202107,202108,202109,202110,202111,202112])
y = np.array([45000,50000,52000,58000,55000,62000,59000,65000,70000,65000,72000,68000])

def fx (x1, coef):
    fx = 0
    n = len(coef) - 1
    for p in coef:
        fx = fx + p*x1**n
        n = n - 1
    return fx

anno = 202113
for i in range(0,15):
    coef = np.polyfit(x,y,i)
    p = np.polyval(coef, anno)
    
    print(f"para grado {i} la predicción es {p}")
    x1 = np.linspace(202101, anno + 1, 100000)
    y1 = fx(x1, coef) # funcion
    plt.figure(figsize=[20,10])
    plt.title("Ventas vs año. Para grado: " + str(i))
    
    plt.scatter(x,y,s=120,c='blueviolet')
    plt.plot(x1,y1,"--",linewidth=3,color='orange')
    plt.scatter(anno,p,s=200,c='red')
    plt.yticks(range(45000,80000,5000))
    plt.grid("on")
    ax=plt.gca()
    ax.set_xlabel("$años$")
    ax.set_ylabel("$Ventas$")
    #plt.savefig("img" + str(i)+".jpg", dpi=600)
    plt.show()