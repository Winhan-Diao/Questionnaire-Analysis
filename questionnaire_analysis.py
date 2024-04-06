# =============================================================================
# 1.A-情感投资
# 2.A-时间投资
# 3.A-经济投资
# *4.B-软实力投资
# 5.B-软实力投资--占比
# *6.C-教育理念
# 7.D-经济回报
# 8.D-非经济回报
# 9.E-双减X投资--态度
# 10.E-双减X投资--成本
# 11.E-双减X投资--结构
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
comb:itertools.combinations
tmp_df:pd.DataFrame
fig:mpl.figure.Figure
ax:mpl.axes._axes.Axes
# =============================================================================
####PRESETS####
selections = pd.Series([3, 4, 4, 4, 5, 4, 5, 3, 3, 3, 3])
multiple = pd.Series([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
plt.rcParams['font.sans-serif'] = ['SimHei']
# =============================================================================
####RANDOM DATA GENERATION####
##THIS IS NOT USED IN THE REAL ANALYSIS PROCESS##
np.random.seed(60)

a = np.random.randint(1, 4, (5, 10))

random_array = np.array([np.random.randint(1, value+1, (100, )) for value in selections]).T

df = pd.DataFrame(random_array)
df = df.replace({1:'A', 2:'B', 3:'C', 4:'D', 5:'E'})

for i in range(0, multiple.size):
    if multiple.get(i) == 1:
        comb = itertools.combinations(range(1, selections.get(i)+1), 2)
        comb_S = pd.Series([t for k in range(1, selections.get(i)+1) for t in itertools.combinations(range(1, selections.get(i)+1), k)])
        comb_S = comb_S.apply(lambda x: ''.join(chr(ord('A')-1+j) for j in x))
        df[i] = df[i].map(lambda x: comb_S.sample(1).iloc[0])

df.to_excel("result.xlsx")
# =============================================================================
####SORT AND MODIFICATION####
df2 = pd.read_excel("result.xlsx", index_col=0)
df2.columns = ("A-情感投资","A-时间投资","A-经济投资","B-软实力投资","B-软实力投资--占比","C-教育理念","D-经济回报","D-非经济回报","E-双减X投资--态度","E-双减X投资--成本","E-双减X投资--结构")

for i in range(0, multiple.size):
    if multiple.get(i) == 1:
        tmp_df = df2.iloc[:,i].str.get_dummies('')
        tmp_df.rename(columns=lambda x: df2.columns[i] + '_' + x, inplace=True)
        df2 = pd.concat((df2, tmp_df), axis=1)
    else:
        df2.iloc[:,i] = df2.iloc[:,i].map({'A':1, 'B':2, 'C':3, 'D':4, 'E':5})
for i in range(multiple.size-1, -1, -1):
    if multiple.get(i) == 1:
        df2.drop(df2.columns[i], axis=1, inplace=True)
df2 = df2.astype(int)

# =============================================================================
####DATA ANYLSIS####
df2_corr = df2.corr()
"""[A]相关系数分析：时间投资VS经济投资VS情感投资"""
print("-"*20+'\n'+"""[A]相关系数分析：时间投资VS经济投资VS情感投资""")
print(df2_corr.iloc[0:3,0:3])

"""相关系数分析：投资教育不同的人受双减影响的区别"""
'See in df2_corr'

"""决策树：各方面->经济回报"""#这决策树没啥用(ˉ▽ˉ；)...
inv_eco_clf = tree.DecisionTreeClassifier()
inv_X = df2.drop(columns=["D-经济回报","D-非经济回报"])
inv_eco_y = df2["D-经济回报"]
inv_oth_y = df2["D-非经济回报"]
inv_eco_clf = inv_eco_clf.fit(inv_X, inv_eco_y)
inv_eco_text = tree.export_text(inv_eco_clf)
print(inv_eco_clf.score(inv_X, inv_eco_y))

"""线性回归：各方面->经济回报"""
print("-"*20+'\n'+"""线性回归：各方面->经济回报（权重系数）""")
inv_eco_reg = LinearRegression()
inv_eco_reg.fit(inv_X, inv_eco_y)
for i in range(0, inv_X.columns.size):
    print(inv_X.columns[i]+":",inv_eco_reg.coef_[i])
    
"""线性回归：各方面->其他回报"""
print("-"*20+'\n'+"""线性回归：各方面->其他回报（权重系数）""")
inv_eco_reg = LinearRegression()
inv_eco_reg.fit(inv_X, inv_oth_y)
for i in range(0, inv_X.columns.size):
    print(inv_X.columns[i]+":",inv_eco_reg.coef_[i])
    
# =============================================================================
####VISUAL REPORT####
"""软实力投资差异与教育理念的饼图"""
df2_soft = df2.loc[..., "B-软实力投资_A":"B-软实力投资_D"].agg('sum')
df2_val = df2.loc[..., "C-教育理念_A":"C-教育理念_D"].agg('sum')
fig = plt.figure(figsize=(10, 10), dpi=500)
ax = fig.add_subplot(221)
ax.pie(df2_soft, labels=df2_soft.index, colors=sns.light_palette("#a275ac"))
ax2 = fig.add_subplot(222)
ax2.pie(df2_val, labels=df2_soft.index, colors=sns.light_palette("seagreen"))
fig.savefig("pies.png")
"""双减影响"""
