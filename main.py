import numpy as np
import pandas as pd
import xgboost as xgb
training = pd.read_csv('/Users/apple/Desktop/携程流失率预测比赛/data/train.csv',parse_dates=['d','arrival'])
testing = pd.read_csv('/Users/apple/Desktop/携程流失率预测比赛/data/yanzhengji_final.csv',parse_dates=['d','arrival'])
training['label'] = 1-training.label
merges=[training,testing]
merge_data = pd.concat(merges)

#merge_data.drop('historyvisit_7ordernum',axis=1,inplace=True)
#merge_data.drop('ordercanceledprecent',axis=1,inplace=True)
#merge_data.drop('ordercanncelednum',axis=1,inplace=True)
#merge_data.drop('historyvisit_visit_detailpagenum',axis=1,inplace=True)
#merge_data.drop('delta_price1',axis=1,inplace=True)
#merge_data.drop('ordernum_oneyear',axis=1,inplace=True)
#merge_data.drop('avgprice',axis=1,inplace=True)
#merge_data.drop('firstorder_bu',axis=1,inplace=True)
#merge_data.drop('delta_price2',axis=1,inplace=True)
#merge_data.drop('customer_value_profit',axis=1,inplace=True)
#merge_data.drop('ctrip_profits',axis=1,inplace=True)
#merge_data.drop('lasthtlordergap',axis=1,inplace=True)
#merge_data=merge_data.fillna({'decisionhabit_user':-999,'historyvisit_totalordernum':-999,'starprefer':-999,'consuming_capacity':-999,'historyvisit_avghotelnum':-999,'price_sensitive':-999,'businessrate_pre':-999,'deltaprice_pre2_t1':-999,'lastpvgap':-999,'cr':-999,'visitnum_oneyear':-999})


merge_data['timediff'] = (merge_data['arrival']-merge_data['d']).astype('timedelta64[D]')
merge_data['dayofweek'] = merge_data['arrival'].dt.dayofweek

merge_data.ix[merge_data.dayofweek==0, 'isweekday'] = 0
merge_data.ix[merge_data.dayofweek==6, 'isweekday'] = 0
merge_data.ix[merge_data.dayofweek==1, 'isweekday'] = 1
merge_data.ix[merge_data.dayofweek==2, 'isweekday'] = 1
merge_data.ix[merge_data.dayofweek==3, 'isweekday'] = 1
merge_data.ix[merge_data.dayofweek==4, 'isweekday'] = 1
merge_data.ix[merge_data.dayofweek==5, 'isweekday'] = 1

dummies_df = pd.get_dummies(merge_data['dayofweek'])  
dummies_df = dummies_df.rename(columns=lambda x:'dayofweek'+str(x))
merge_data = pd.concat([merge_data,dummies_df],axis=1)

dummies_df = pd.get_dummies(merge_data['h'])  
dummies_df = dummies_df.rename(columns=lambda x:'h'+str(x))
merge_data = pd.concat([merge_data,dummies_df],axis=1)

def num_missing(x):    
    return sum(x.isnull())  

merge_data['num_missing'] = merge_data.apply(num_missing, axis=1) 
merge_data['cr_jet_num'] = merge_data['cr'].map(lambda x: 1 if x>1.39 else 0)
merge_data['businessrate_pre_jet_num'] = merge_data['businessrate_pre'].map(lambda x: 1 if x>0.5 else 0)
merge_data['businessrate_pre2_jet_num'] = merge_data['businessrate_pre2'].map(lambda x: 1 if x>0.5 else 0)


merge_data = merge_data.fillna(0)

data_train=merge_data.iloc[0:689945,]
data_test=merge_data.iloc[689945:,]

data_train.drop(['d'],axis=1,inplace=True)
data_train.drop(['arrival'],axis=1,inplace=True)
data_test.drop(['label'],axis=1,inplace=True)
data_test.drop(['d'],axis=1,inplace=True)
data_test.drop(['arrival'],axis=1,inplace=True)

from sklearn.cross_validation import train_test_split
random_seed = 1024
#split train set,generate train,val,test set
train_xy = data_train.drop(['sampleid'],axis=1)
train,val = train_test_split(train_xy, test_size = 0.3,random_state=1)#random_state is of big influence for val-auc
#train1 = pd.merge(train_xy[train_xy.dayofweek0==1],train_xy[train_xy.dayofweek1==1],how='outer')
#train2 = pd.merge(train_xy[train_xy.dayofweek2==1],train_xy[train_xy.dayofweek3==1],how='outer')
#train3 = pd.merge(train2,train_xy[train_xy.dayofweek4==1],how='outer')
#train = pd.merge(train1,train3,how='outer')
#val = pd.merge(train_xy[train_xy.dayofweek5==1],train_xy[train_xy.dayofweek6==1],how='outer')
y = train.label
X = train.drop(['label'],axis=1)
val_y = val.label
val_X = val.drop(['label'],axis=1)
test_X = data_test.drop(['sampleid'],axis=1)
test_no = data_test.sampleid
#xgboost start here
dtest = xgb.DMatrix(test_X)
dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params={
	'booster':'gbtree',
	'objective': 'binary:logistic',
	'early_stopping_rounds':100,
        'eval_metric': 'auc',
	'gamma':0.1,
	'max_depth':8,
	'lambda':10,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2, 
        'eta': 0.025,
	'seed':0,
    }

params['eval_metric'] = ['auc', 'error']

watchlist  = [(dtrain,'train'),(dval,'val')]#The early stopping is based on last set in the evallist
model = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)
#model.save_model('/Users/apple/Downloads/xgb8_23_3.model')
print "best best_ntree_limit",model.best_ntree_limit   #did not save the best,why?

#predict test set (from the best iteration)
test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["sampleid","prob"])
test_result.sampleid = data_test.sampleid
test_result.prob = test_y
