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
selections  = pd.Series([4,5,4,3,4,3,3,3,3,2,3,3,3,3])
needReverse = pd.Series([0,0,0,1,1,1,0,1,0,1,1,1,1,1])
related     = pd.Series([1,1,1,1,1,1,0,1,0,1,1,1,1,1])
names = pd.Series(["A-家书数",'A-父母教育程度',"A-年收入",'B-情感投资',"B-时间投资",'B-经济投资',"C-软实力投资",'C-软实力投资占比',"D-教育理念",'E-经济回报',"E-非经济回报",'F-双减X投资--态度',"F-双减X投资--成本",'F-双减X投资--结构'])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# import matplotlib.font_manager
# flist = matplotlib.font_manager.findSystemFonts()
# names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]
# print(names)
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
    df3.iloc[:,i] = df2.iloc[:,i].map(lambda x: choiceToInt(x, needReverse[i], selections[i]))
    # df3 = pd.concat((df2.iloc[:,i].map(lambda x: choiceToInt(x, relation[i], selections[i])),df3), axis=1)
df3_related = df3.loc[:, related.astype(bool).values].astype(int)
df3_unrelated = df3.loc[:, ~related.astype(bool).values].astype(int)
df3 = df3.astype(int)
df3_melt = df3.melt(var_name='k',value_name='程度')
# =============================================================================
####DATA ANYLSIS####
'''所有相关系数(spearman method)'''
df3_corr = df3_related.corr(method='spearman')
df3_corr.to_csv("correlation.csv")

'''Linear regression on non-eco returns'''
X = df3_related.drop(['E-经济回报',"E-非经济回报"],axis=1)
y = df3["E-非经济回报"]
reg_noneco = LinearRegression()
reg_noneco.fit(X, y)
print(reg_noneco.coef_,reg_noneco.intercept_)

'''Tree dicision for non-eco returns'''
X_tree = df3.drop(['E-经济回报',"E-非经济回报"],axis=1)
clf_noneco = tree.DecisionTreeClassifier()
clf_noneco = clf_noneco.fit(X_tree, y)

# =============================================================================
####VISUAL REPORT####
T = (25,20)
fig = plt.figure(figsize=T,dpi=300)
fig.suptitle('上海市居民家庭教育投资状况调查',fontsize=60,verticalalignment='top')
gs = fig.add_gridspec(*T)

'''HeatMap for all correlations'''
ax = fig.add_subplot(gs[:5,2:])
ax = sns.heatmap(df3_corr,annot=True,cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True),linewidths=.1)
ax.tick_params(axis='x',labelsize='x-small',labelrotation=0)
ax.set_xlabel('所有可量化题目的线性相关系数',fontsize='large')

'''Violin plot for all choices'''
ax2 = fig.add_subplot(gs[6:10,:])
ax2.grid(True)
ax2 = sns.violinplot(df3_melt, x='k', y='程度',palette='Pastel2',width=.8,fill=True,gap=.1,bw_adjust=1)
ax2.set_xlabel('所有选项的选择比重',fontsize='large')

'''提琴图：家庭教育素质对投资、回报、双减态度。。。的影响'''
'>=200书 && >=12年'
df_up = df3[(df3["A-家书数"]>=3) & (df3["A-父母教育程度"]>=4)]
df_low = df3[~((df3["A-家书数"]>=3) & (df3["A-父母教育程度"]>=4))]
L_up = ['up' for _ in range(df_up.size)]
L_low = ['low' for _ in range(df_low.size)]
df_up_melt = df_up.melt(var_name='k',value_name='程度')
df_up_melt.insert(2,'education',L_up)
df_low_melt = df_low.melt(var_name='k',value_name='程度')
df_low_melt.insert(2,'education',L_low)
df_edu_clsf = df_up_melt._append(df_low_melt)
ax3 = fig.add_subplot(gs[11:15,:])
ax3.grid(True)
ax3 = sns.violinplot(df_edu_clsf, x='k', y='程度', hue='education',palette='Set2',width=1,fill=False,gap=.1,split=True,bw_adjust=1)
ax3.set_xlabel('选项选择比重的差异--家庭教育素质高VS普通',fontsize='large')

'''提琴图：家庭年收入对投资、回报、双减态度。。。的影响'''
df_wea = df3[df3["A-年收入"]>=3]
df_poo = df3[~(df3["A-年收入"]>=3)]
L_wea = ['wealthy' for _ in range(df_wea.size)]
L_poo = ['poor' for _ in range(df_poo.size)]
df_wea_melt = df_wea.melt(var_name='k',value_name='程度')
df_wea_melt.insert(2,'income',L_wea)
df_poo_melt = df_poo.melt(var_name='k',value_name='程度')
df_poo_melt.insert(2,'income',L_poo)
df_inc_clsf = df_wea_melt._append(df_poo_melt)
ax4 = fig.add_subplot(gs[16:20,:])
ax4.grid(True)
ax4 = sns.violinplot(df_inc_clsf, x='k', y='程度', hue='income',palette='husl',width=1,fill=False,gap=.1,split=True)
ax4.set_xlabel('选项选择比重的差异--家庭收入高VS普通',fontsize='large')

'''条形图：各可量化选项对非经济回报的线性相关权重'''
df_fimp = pd.DataFrame(clf_noneco.feature_importances_,index=X_tree.columns,columns=['决策树权重'])
df_fimp_melt = df_fimp.melt(var_name='方法',value_name='权重')
df_fimp_melt['题目'] = df_fimp.index
df_fcoe = pd.DataFrame(reg_noneco.coef_,index=X.columns,columns=['线性回归系数权重'])
df_fcoe_melt = df_fcoe.melt(var_name='方法',value_name='权重')
df_fcoe_melt['题目'] = df_fcoe.index
df_coim_melt = pd.concat((df_fimp_melt,df_fcoe_melt))

ax5 = fig.add_subplot(gs[21:,:])
ax5.set_axisbelow(True)
ax5.grid(True,color='silver')
ax5.axhline(0, color="k", clip_on=False)
ax5 = sns.barplot(df_coim_melt, x='题目', y='权重', hue='方法', palette="pastel")
ax5.bar_label(ax5.containers[0], fmt="{:.2f}".format, fontsize=10)
ax5.bar_label(ax5.containers[1], fmt="{:.2f}".format, fontsize=10)
# ax5.set_xticklabels(X.columns)
ax5.set_xlabel('各可量化选项对非经济回报的线性相关权重',fontsize='large')
fig.savefig('result.png')
