import csv
import sys
import datetime
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from operator import sub
from sklearn import preprocessing, ensemble
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def runXGB(train_X, train_y, seed_val=123):
    # 0.87 0.87 140 total log loss is 0.976276401146
    # 0.87 0.87 120 total log loss is 0.978276401146
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.085
	param['max_depth'] = 4
	param['silent'] = 1
	param['num_class'] = 22
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 5
	param['subsample'] = 0.875
	param['colsample_bytree'] = 0.875
	param['seed'] = seed_val
	num_rounds = 140  #total log loss is 0.964639614358

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model

def score(Xtrain,y,random_state = 0):
	print Xtrain.shape
	print y.shape
	kf = StratifiedKFold(y,n_folds = 5, shuffle = True, random_state = random_state)
	pred = np.zeros((y.shape[0],22))
	for i,(itrain,itest) in enumerate(kf):
		print ("fold : ",i)
		Xtr,Xte = Xtrain[itrain,:],Xtrain[itest,:]
		ytr,yte = y[itrain],y[itest]
		print "training"
		model = runXGB(Xtr, ytr, seed_val=0)
		xgtest = xgb.DMatrix(Xte)
		print "testing"
		preds = model.predict(xgtest)
		print pred.shape
		print preds.shape
		pred[itest,:] = preds
		print ("{:.5f}".format(log_loss(yte, preds)))
	print "total log loss is "+ str(log_loss(y, pred))

def treeScore(clf,X,y,random_state = 0):
	kf = StratifiedKFold(y,n_folds = 3, shuffle = True,random_state = random_state)
	pred = np.zeros((y.shape[0],22))
	for itrain,itest in kf:
		Xtr,Xte = X[itrain,:],X[itest,:]
		ytr,yte = y[itrain],y[itest]
		model = clf.fit(Xtr,ytr)
		preds = model.predict_proba(Xte)
		pred[itest,:] = preds
		print ("{:.5f}".format(log_loss(yte, preds)))
	print "total log loss is "+ str(log_loss(y, pred))

def mean(test):
	avgList = np.zeros((test.shape[0],22))
	for i in range(22):
		avgList[:,i] = (test[:,i]+test[:,i+22])/2
	return avgList

def stacking(clfs,Xtrain,ytrain,Xtest):
	n_folds = 4
	nclasses = 22
	skf = list(StratifiedKFold(ytrain, n_folds))
	dataset_stack_train = np.zeros((Xtrain.shape[0], nclasses*len(clfs)))
	dataset_stack_test = np.zeros((Xtest.shape[0],nclasses*len(clfs)))

	for j,clf in enumerate(clfs):
		print("============  ",j)
		dataset_stack_test_j = np.zeros((Xtest.shape[0],nclasses*len(skf)))
		if j<2:
			for i,(itrain,itest) in enumerate(skf):
				print i
				Xtr = Xtrain[itrain,:]
				ytr = ytrain[itrain]
				Xte = Xtrain[itest,:]
				yte = ytrain[itest]
				clf.fit(Xtr,ytr)
				preds = clf.predict_proba(Xte)
				test_pred = clf.predict_proba(Xtest)
				dataset_stack_train[itest,j*nclasses:(j+1)*nclasses] = preds
				dataset_stack_test_j[:, i*nclasses:(i+1)*nclasses] = test_pred
			dataset_stack_test[:,j*nclasses:(j+1)*nclasses] = mean(dataset_stack_test_j)
		
		if j == 2:
			for i,(itrain,itest) in enumerate(skf):
				print i
				Xtr = Xtrain[itrain,:]
				ytr = ytrain[itrain]
				Xte = Xtrain[itest,:]
				yte = ytrain[itest]
				model = runXGB(Xtr, ytr, seed_val=0)
				xgte = xgb.DMatrix(Xte)
				preds = model.predict(xgte)			
				xgtest = xgb.DMatrix(Xtest)
				test_pred = model.predict(xgtest)
				dataset_stack_train[itest,j*nclasses:(j+1)*nclasses] = preds
				dataset_stack_test_j[:, i*nclasses:(i+1)*nclasses] = test_pred
			dataset_stack_test[:,j*nclasses:(j+1)*nclasses] = mean(dataset_stack_test_j)
	return dataset_stack_train,dataset_stack_test

if __name__ == "__main__":
	start_time = datetime.datetime.now()
	print start_time
	f = open('train1220_X.pkl','r')
	train_X = pickle.load(f)
	f.close()
	f = open('train1220_y.pkl','r')
	train_y=pickle.load(f)
	f.close()
	score(train_X,train_y,random_state = 0)
	print datetime.datetime.now()-start_time