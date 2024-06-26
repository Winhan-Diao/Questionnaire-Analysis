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
np.random.seed(6400)

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
        df2.iloc[:,i] = df2.iloc[:,i].map(lambda x:selections[i]-(ord(x)-ord('A')))
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

"""[D]相关系数分析：经济回报与非经济回报"""
print(df2_corr.iloc[4:6,4:6])

"""[ABCE-D]决策树：各方面->经济回报"""#这决策树没啥用(ˉ▽ˉ；)...
inv_eco_clf = tree.DecisionTreeClassifier()
inv_X = df2.drop(columns=["D-经济回报","D-非经济回报"])
inv_eco_y = df2["D-经济回报"]
inv_oth_y = df2["D-非经济回报"]
inv_eco_clf = inv_eco_clf.fit(inv_X, inv_eco_y)
inv_eco_text = tree.export_text(inv_eco_clf)
print(inv_eco_clf.score(inv_X, inv_eco_y))

"""[ABCE-D]线性回归：各方面->经济回报"""
print("-"*20+'\n'+"""线性回归：各方面->经济回报（权重系数）""")
inv_eco_reg = LinearRegression()
inv_eco_reg.fit(inv_X, inv_eco_y)
for i in range(0, inv_X.columns.size):
    print(inv_X.columns[i]+":",inv_eco_reg.coef_[i])
    
"""[ABCE-D]线性回归：各方面->其他回报"""
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
fig = plt.figure(figsize=(15, 30), dpi=500)
fig.suptitle('上海市居民家庭教育投资状况调查',verticalalignment='center',fontsize=30,y=.93)
ax = fig.add_subplot(621)
ax.set_title('【多选】软实力投资比重')
ax.pie(df2_soft, labels=df2_soft.index, colors=sns.light_palette("#a275ac"))

ax2 = fig.add_subplot(622)
ax2.set_title('【多选】教育理念比重')
ax2.pie(df2_val, labels=df2_soft.index, colors=sns.light_palette("seagreen"))
"""[D]密度二维图：经济回报与非经济回报"""
ax3 = fig.add_subplot(625)
ax3.set_title('经济回报与非经济回报比重及相关性比较')
df_ben=pd.crosstab(index=df2["D-经济回报"], columns=df2["D-非经济回报"])
sns.heatmap(data=df_ben,linewidth=.5,ax=ax3,cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True))
"""[ABCDE]提琴图：各单选题选择比重"""
ax4 = fig.add_subplot(6,2,(3,4))
ax4.set_title('【单选】各题选择比重')
sns.violinplot(df2.loc[:,:'E-双减X投资--结构'].div([3,4,4,5,5,3,3,3,3]),ax=ax4,palette=sns.color_palette("pastel"),width=.5)
ax4.grid(True)

'''[A]提琴图：情感投资在选则比重的差异'''
df2_soft_melt = df2.loc[:,'B-软实力投资_A':'B-软实力投资_D'].melt(var_name='B-软实力投资',value_name='Density')
df2_soft_com = df2_soft_melt[df2_soft_melt['Density'] != 0].loc[:,'B-软实力投资'].replace({"B-软实力投资_A":1,"B-软实力投资_B":2,"B-软实力投资_C":3,"B-软实力投资_D":4})
df2_val_melt = df2.loc[:,"C-教育理念_A":"C-教育理念_D"].melt(var_name='C-教育理念',value_name='Density')
df2_val_com = df2_val_melt[df2_val_melt['Density'] != 0].loc[:,'C-教育理念'].replace({"C-教育理念_A":1,"C-教育理念_B":2,"C-教育理念_C":3,"C-教育理念_D":4})

df2_com=pd.concat((df2.loc[:,:'E-双减X投资--结构'],df2_soft_com,df2_val_com),axis=1)
ax5 = fig.add_subplot(6,2,(7,8))
ax5 = sns.violinplot(df2_com,palette=sns.color_palette("Set2"),fill=False)




fig.savefig("pies.png")