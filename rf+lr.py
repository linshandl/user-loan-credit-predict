# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:42:08 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('all/cs-training.csv')
df_test = pd.read_csv('all/cs-test.csv')
columns = ({'SeriousDlqin2yrs':'好坏客户','RevolvingUtilizationOfUnsecuredLines':'可用额度比值',
            'age':'年龄','NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天笔数',
            'NumberOfOpenCreditLinesAndLoans':'信贷数量','DebtRatio':'负债率','MonthlyIncome':'月收入',
            'NumberOfTimes90DaysLate':'逾期90天笔数','NumberRealEstateLoansOrLines':'固定资产贷款数量',
            'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天笔数','NumberOfDependents':'家属数量'})
df_train.rename(columns=columns,inplace=True)
df_test.rename(columns=columns,inplace=True)
df_train = df_train.drop(df_train.columns[0],axis=1)
df_test = df_test.drop(df_test.iloc[:,:2],axis=1)

print('训练集缺失值情况：','\n',df_train.isnull().sum())
print('测试集缺失值情况：','\n',df_test.isnull().sum())

print('训练集月收入缺失比:{:.2%}'.format(df_train['月收入'].isnull().sum()/df_train.shape[0]))
print('训练集家属数量缺失比:{:.2%}'.format(df_train['家属数量'].isnull().sum()/df_train.shape[0]))
print('测试集月收入缺失比:{:.2%}'.format(df_test['月收入'].isnull().sum()/df_test.shape[0]))
print('测试集家属数量缺失比:{:.2%}'.format(df_test['家属数量'].isnull().sum()/df_test.shape[0]))

from sklearn.ensemble import RandomForestRegressor as RFR
def fill_missing(df):
    all_df = df.iloc[:,[5,0,1,2,3,4,6,7,8,9]]    #第5列表示月收入，去除家属数量
    # df.head()
    known = all_df[all_df.月收入.notnull()].as_matrix()
    unknown = all_df[all_df.月收入.isnull()].as_matrix()
    X = known[:,1:]
    Y = known[:,0]
    rfr = RFR(random_state=0,n_estimators=200,max_depth=3)
    rfr.fit(X,Y)
    predict = rfr.predict(unknown[:,1:]).round(0)
    df.loc[(df.月收入.isnull()),'月收入'] = predict
    return df
df_train = fill_missing(df_train)

from sklearn.ensemble import RandomForestRegressor as RFR
def fill_missing(df):
    all_df = df.iloc[:,[4,0,1,2,3,5,6,7,8]]    #第4列表示月收入，去除家属数量
    # df.head()
    known = all_df[all_df.月收入.notnull()].as_matrix()
    unknown = all_df[all_df.月收入.isnull()].as_matrix()
    X = known[:,1:]
    Y = known[:,0]
    rfr = RFR(random_state=0,n_estimators=200,max_depth=3)
    rfr.fit(X,Y)
    predict = rfr.predict(unknown[:,1:]).round(0)
    df.loc[(df.月收入.isnull()),'月收入'] = predict
    return df
df_test = fill_missing(df_test)

from sklearn.preprocessing import Imputer
imp1 = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp2 = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp_train = imp1.fit(df_train)
imp_test = imp2.fit(df_test)
df_train = pd.DataFrame(imp_train.transform(df_train))
df_test = pd.DataFrame(imp_test.transform(df_test))

col_train = ['好坏客户','可用额度比值','年龄','逾期30-59天笔数','信贷数量','负债率','月收入','逾期90天笔数',
       '固定资产贷款数量','逾期60-89天笔数','家属数量']
col_test = ['可用额度比值','年龄','逾期30-59天笔数','信贷数量','负债率','月收入','逾期90天笔数',
       '固定资产贷款数量','逾期60-89天笔数','家属数量']
df_train.columns=col_train
df_test.columns=col_test

df_train = df_train[df_train['年龄'] > 0]   #只有一个异常值

#求各组年龄的总人数
age_cut = pd.cut(df_train['年龄'],5)
age_cut_group_all = df_train['好坏客户'].groupby(age_cut).count()
age_cut_group_all

age_cut_group_bad = df_train['好坏客户'].groupby(age_cut).sum()
age_cut_group_bad

#连接
after_join = pd.merge(pd.DataFrame(age_cut_group_all),pd.DataFrame(age_cut_group_bad),left_index=True,right_index=True)
after_join.rename(columns={'好坏客户_x':'总客户数','好坏客户_y':'坏客户数'},inplace=True)
after_join

#增加好客户数和坏客户占比
after_join.insert(2,'好客户数',after_join['总客户数']-after_join['坏客户数'])
after_join.insert(3,'坏客户占比',after_join['坏客户数']/after_join['总客户数'])
after_join

# fig = plt.figure(figsize=(10,5))
ax1 = after_join[['好客户数','坏客户数']].plot.bar()
ax1.set_xticklabels(after_join.index,rotation=15)
ax1.set_ylabel('客户数')
ax1.set_title('年龄与好坏客户数分布图')

ax2 = after_join['坏客户占比'].plot()
ax2.set_xticklabels([0,20,29,38,47,55,64,72,81,89,98,107])
ax2.set_ylabel('坏客户率')
ax2.set_title('坏客户率随着年龄的变化曲线图')


corr = df_train.corr()
fig = plt.figure(figsize=(10,8))
ax3 = fig.add_subplot(1,1,1)
sns.heatmap(corr,annot=True,fmt='.2f',annot_kws={'size':10})
xticks = list(corr.index)  #x轴标签
yticks = list(corr.index)  #y轴标签
ax3.set_xticklabels(xticks,rotation=30,fontsize=8)
ax3.set_yticklabels(yticks,fontsize=8)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

x = df_train.iloc[:,1:]
y = df_train.iloc[:,:1]
x_train_rf,x_val_rf,y_train_rf,y_val_rf = train_test_split(x,y,test_size=0.2,random_state=123)

#rf = RandomForestClassifier(n_estimators=200,
#                                 oob_score= True,
#                                 min_samples_split=2,
#                                 min_samples_leaf=50,
#                                 n_jobs=-1,
#                                 class_weight='balanced_subsample',
#                                 bootstrap=True)
# param_grid = {"max_features": [2, 3, 4]}
# grid_search = GridSearchCV(rf, cv=10, scoring='roc_auc', param_grid=param_grid, iid=False)

# grid_search.fit(x_train_rf, y_train_rf)
# print("the best parameter:", grid_search.best_params_)
# print("the best score:", grid_search.best_score_)     #训练集上的auc值

clf = RandomForestClassifier(n_estimators=200,
                                oob_score= True,
                                min_samples_split=2,
                                min_samples_leaf=50,
                                max_features=2,
                                n_jobs=-1,
                                class_weight='balanced_subsample',
                                bootstrap=True)
clf.fit(x_train_rf,y_train_rf)
pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), x_train_rf.columns),reverse=True))

def computeAUC(y_true,y_score):
    auc = roc_auc_score(y_true,y_score)
    print("auc=",auc)
    return auc
    
print('训练集上的各类预测概率')
# predicted_probs_train = grid_search.predict_proba(x_train_rf)
predicted_probs_train = clf.predict_proba(x_train_rf)
predicted_probs_train = [x[1] for x in predicted_probs_train]
computeAUC(y_train_rf, predicted_probs_train)  

# predicted_probs_val = grid_search.predict_proba(x_val_rf)
predicted_probs_val = clf.predict_proba(x_val_rf)
predicted_probs_val = [x[1] for x in predicted_probs_val]
computeAUC(y_val_rf, predicted_probs_val)     #验证集上的auc

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_rf, y_train_rf)
predicted_probs_train = lr.predict_proba(x_train_rf)
predicted_probs_train = [x[1] for  x in predicted_probs_train]
computeAUC(y_train_rf, predicted_probs_train)

predicted_probs_test = lr.predict_proba(x_val_rf)
predicted_probs_test = [x[1] for x in predicted_probs_test]
computeAUC(y_val_rf, predicted_probs_test)

predicted_probs_test = grid_search.predict_proba(df_test)
predicted_probs_test = ["%.9f" % x[1] for x in predicted_probs_test]
submission = pd.DataFrame({'Probability':predicted_probs_test})
submission.to_csv("rf_predict.csv", index=False)