# =============================================================================
# 1.A-家书数
# 2.A-父母教育程度
# 3.A-年收入
# 4.B-情感投资
# 5.B-时间投资
# 6.B-经济投资【不分析不确定】
# 7.C-软实力投资【不分析补充】（选项非比重）
# 8.C-软实力投资占比
# 9.D-教育理念【不分析补充】【不分析不确定】（选项非比重）
# 10.E-经济回报
# 11.E-非经济回报
# 12.F-双减X投资--态度
# 13.F-双减X投资--成本
# 14.F-双减X投资--结构
# =============================================================================
import pandas as pd
import numpy as np
import itertools
from sklearn import tree
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

df:pd.DataFrame
df2:pd.DataFrame
df3:pd.DataFrame
comb:itertools.combinations
tmp_df:pd.DataFrame
fig:mpl.figure.Figure
ax:mpl.axes._axes.Axes
# =============================================================================
####PRESETS####
selections = pd.Series([4,5,4,3,4,3,3,3,3,2,3,3,3,3])
relation =   pd.Series([1,1,1,1,1,1,0,1,0,1,1,1,1,1])
names = pd.Series(["A-家书数",'A-父母教育程度',"A-年收入",'B-情感投资',"B-时间投资",'B-经济投资',"C-软实力投资",'8.C-软实力投资占比',"D-教育理念",'E-经济回报',"E-非经济回报",'F-双减X投资--态度',"F-双减X投资--成本",'F-双减X投资--结构'])
plt.rcParams['font.sans-serif'] = ['SimHei']
# =============================================================================
####RANDOM DATA GENERATION####
##THIS IS NOT USED IN THE REAL ANALYSIS PROCESS##
np.random.seed(6400)

a = np.random.randint(1, 4, (5, 10))

random_array = np.array([np.random.randint(1, value+1, (100, )) for value in selections]).T

df = pd.DataFrame(random_array)
df = df.replace({1:'A', 2:'B', 3:'C', 4:'D', 5:'E'})
df.columns = names
df.to_excel("generated.xlsx")
# =============================================================================
####SORT AND MODIFICATION####
def choiceToInt(c: str, reverse: bool, biggest: int) -> int:
    if reverse:
        i = biggest - (ord(c)-ord('A'))
    else:
        i = ord(c) - ord('A') + 1
    return i
df2 = pd.read_excel("generated.xlsx", index_col=0)
df3 = df2
for i in range(0,df2.shape[1]):
    df3.iloc[:,i] = df2.iloc[:,i].map(lambda x: choiceToInt(x, relation[i], selections[i]))
    # df3 = pd.concat((df2.iloc[:,i].map(lambda x: choiceToInt(x, relation[i], selections[i])),df3), axis=1)
df3_relate = df3.loc[:, relation.astype(bool).values]
df3_unrelate = df3.loc[:, ~relation.astype(bool).values]
# =============================================================================
####DATA ANYLSIS####
# =============================================================================
####VISUAL REPORT####
