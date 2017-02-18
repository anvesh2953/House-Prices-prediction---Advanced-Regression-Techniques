#import packages which are necessary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, skewtest
import matplotlib

train = 'C:/Materials/Classes/Big Data/House Prices/Data/train.csv'
test = 'C:/Materials/Classes/Big Data/House Prices/Data/test.csv'
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)
df_full =df_train.append(df_test)


fullCat = df_full.select_dtypes(include=['object']).index
fullCont = df_full.dtypes[df_full.dtypes !='object'].index

skewed_data = df_full[fullCont].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_data = skewed_data[skewed_data > 0.75]
skewed_data = skewed_data.index
df_full[skewed_data] = np.log(df_full[skewed_data] + 1)

df_full = pd.get_dummies(df_full)
df_full = df_full.fillna(df_full[:df_train.shape[0]].mean())
def heatmap(df,labels):
    cm = np.corrcoef(df[labels].dropna().values.T)
    sns.set(font_scale=1)
    hm = sns.heatmap(cm,
                     cbar= False,
                     annot =False,
                     square= True,
                     vmax=1
                     )

    #heatmap = ax.pcolor(nba_sort, cmap=plt.cm.Blues, alpha=0.8
    color_map = plt.cm.Blues
    plt.pcolor(cm,cmap=color_map)
    plt.colorbar().set_label("Features", rotation=270)
    hm.set_xticklabels(labels, rotation=90)
    hm.set_yticklabels(labels[::-1], rotation=0)

    return hm,cm

Features = list(df_full[fullCont].columns.values)
plt.figure(figsize = (20,10))
htmp,corrm = heatmap(df_full[fullCont],Features)
plt.show()

#Feature Selection
df_train['SalePrice'] = np.log(df_train['SalePrice'])
del df_full['SalePrice']
trainData = df_full[:df_train.shape[0]]
testData = df_full[:df_test.shape[0]]
YData = df_train.SalePrice

#Applying Lasso Model:
#Lasso model :
class sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic) ¶
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LassoCV, LassoLarsCV, LinearRegression
#calculating root mean square error
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, trainData, YData, scoring="neg_mean_squared_error", cv=5))

    return(rmse)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001,0.0005], selection='random', max_iter=15000).fit(trainData, YData)
res = rmse_cv(model_lasso)
#print(res)
print("Lasso Mean:",res.mean())
print("Lasso Min: ",res.min())


#Finding out important features by Lasso Model
coef = pd.Series(model_lasso.coef_, index = trainData.columns)
#print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh",color = 'r')
plt.title("Coefficients used in the Lasso Model")
plt.show()

# Predicting the final Sale Price of the house using Lasso:
test_preds = np.expm1(model_lasso.predict(testData))
result = pd.DataFrame()
result['Id'] = df_test['Id']
result["SalePrice"] = test_preds
result.to_csv("C:/Materials/Classes/Big Data/House Prices/Data/lasso.csv", index=False)


#Applying Linear Regression:

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression = linear_regression.fit(trainData,YData)
res1 = rmse_cv(linear_regression)
test_preds = (linear_regression.predict(testData))
result2 = pd.DataFrame()
result2['Id'] = df_test['Id']
result2["SalePrice"] = np.exp(test_preds)
result2.to_csv("C:/Materials/Classes/Big Data/House Prices/Data/linear.csv", index=False)
print("Linear Regression Mean",res1.mean())
print("Linear Regression Min:",res1.min())

#Applying XGBoost Model
import xgboost as xgb
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200)
regr.fit(trainData,YData)

print("XGBoost Mean ", rmse_cv(regr).mean())
res2 = rmse_cv(regr).mean()
pred_xgb=  regr.predict(testData)
 result1 = pd.DataFrame()
result1['Id'] = df_test['Id']
result1["SalePrice"] = np.exp(pred_xgb)
result1.to_csv("C:/Materials/Classes/Big Data/House Prices/Data/xgb.csv", index=False)

#Comparing the Root mean square error for 3 models:

labels = ['LinearRegression','Laaso Model','XGBoost']
meanScores = [0.1655,0.1229,0.119]
ind_scrs = np.arange(0,len(meanScores))
width = 0.3
fig, ax = plt.subplots()
rects = ax.bar(ind_scrs, meanScores, width, color = 'b')
ax.set_xticks(np.array(ind_scrs) + width/2)
ax.set_xticklabels(labels)
ax.set_ylabel('RMSE on test set')
ax.set_ylim([0,0.2])
plt.show()





