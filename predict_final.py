import csv
import sys
import datetime
import pickle
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

mapping_dict = {
'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo'          : {'V':0, 'H':1, -99:2},
'ind_nuevo'     : {'0':0, '1':1, -99:2},
'indrel'        : {'1':0, '99':1, -99:2},
'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},
'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'indresi'       : {-99:0, 'S':1, 'N':2},
'indext'        : {-99:0, 'S':1, 'N':2},
'conyuemp'      : {-99:0, 'S':1, 'N':2}, #the null number is 1005073
'indfall'       : {-99:0, 'S':1, 'N':2},
'tipodom'       : {-99:0, '1':1},
'ind_actividad_cliente' : {'0':0, '1':1, -99:2},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 
					'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 
					'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 
					'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 
					'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 
					'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 
					'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 
					'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 
					'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105,
 					-99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 
					'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 
					'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 
					'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 
					'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 
					'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 
					'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 
					'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 
					'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 
					'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 
					'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 
					'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 
					'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 
					'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 
					'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 
					'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 
					'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}
cat_cols = list(mapping_dict.keys())

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
				'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
				'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
				'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]

def getTarget(row):
	tlist = []
	for col in target_cols:
		if row[col].strip() in ['', 'NA']:
			target = 0
		else:
			target = int(float(row[col]))
		tlist.append(target)
	return tlist

def getIndex(row, col):
	val = row[col].strip()
	if val not in ['','NA']:
		ind = mapping_dict[col][val]
	else:
		ind = mapping_dict[col][-99]
	return ind

def getAge(row):
	mean_age = 40.
	min_age = 18.
	max_age = 100.
	range_age = max_age - min_age
	age = row['age'].strip()
	if age == 'NA' or age == '':
		age = mean_age
	else:
		age = float(age)
		if age < min_age:
			age = min_age
		elif age > max_age:
			age = max_age
	return round( (age - min_age) / range_age, 4)

def getCustSeniority(row):
	min_value = 0.
	max_value = 256.
	range_value = max_value - min_value
	missing_value = 0.
	cust_seniority = row['antiguedad'].strip()
	if cust_seniority == 'NA' or cust_seniority == '':
		cust_seniority = missing_value
	else:
		cust_seniority = float(cust_seniority)
		if cust_seniority < min_value:
			cust_seniority = min_value
		elif cust_seniority > max_value:
			cust_seniority = max_value
	return round((cust_seniority-min_value) / range_value, 4)

def getRent(row):
	min_value = 0.
	max_value = 1500000.
	range_value = max_value - min_value
	renta_dict = {'ALBACETE': 76895,  'ALICANTE': 60562,  'ALMERIA': 77815,  'ASTURIAS': 83995,  'AVILA': 78525,  
				'BADAJOZ': 60155,  'BALEARS, ILLES': 114223,  'BARCELONA': 135149,  'BURGOS': 87410, 'NAVARRA' : 101850,
    			'CACERES': 78691,  'CADIZ': 75397,  'CANTABRIA': 87142,  'CASTELLON': 70359,  'CEUTA': 333283, 'CIUDAD REAL': 61962,  
    			'CORDOBA': 63260,  'CUENCA': 70751,  'GIRONA': 100208,  'GRANADA': 80489,'GUADALAJARA': 100635,  'HUELVA': 75534,  
    			'HUESCA': 80324,  'JAEN': 67016,  'LEON': 76339,  'LERIDA': 59191,  'LUGO': 68219,  'MADRID': 141381,  'MALAGA': 89534,  
    			'MELILLA': 116469, 'GIPUZKOA': 101850,'MURCIA': 68713,  'OURENSE': 78776,  'PALENCIA': 90843,  'PALMAS, LAS': 78168,  
    			'PONTEVEDRA': 94328,  'RIOJA, LA': 91545,  'SALAMANCA': 88738,  'SANTA CRUZ DE TENERIFE': 83383, 'ALAVA': 101850, 'BIZKAIA' : 101850,
    			'SEGOVIA': 81287,  'SEVILLA': 94814,  'SORIA': 71615,  'TARRAGONA': 81330,  'TERUEL': 64053,  'TOLEDO': 65242,  'UNKNOWN': 103689,  
    			'VALENCIA': 73463,  'VALLADOLID': 92032,  'ZAMORA': 73727,  'ZARAGOZA': 98827}

	rent = row['renta'].strip()
	if rent == 'NA' or rent == '':
	    if row['nomprov'] == 'NA' or row['nomprov'] == '':
	        rent = float(renta_dict['UNKNOWN'])
	    else:
	    	if(row['nomprov'] not in renta_dict.keys()):
	    		rent = 103567
	    	else:
	        	rent = float(renta_dict[row['nomprov']])
	else:
		rent = float(rent)
		if rent < min_value:
			rent = min_value
		elif rent > max_value:
			rent = max_value

	return round((rent-min_value) / range_value, 6)

def getMarriageIndex(age, sex, income):
    marriage_age = 28
    modifier = 0
    if sex == 'V':
        modifier += -2
    if income <= 101850:
        modifier += -1
    
    marriage_age_mod = marriage_age + modifier
    
    if age <= marriage_age_mod:
        return 0
    else:
        return 1

def listDotMultiply(target_months_list):
	n = len(target_months_list[0])
	newList = [0]*n
	for i in range(n):
		for j,tempList in enumerate(target_months_list):
			newList[i] = newList[i] | tempList[i]
	return newList


def processData(in_file_name, cust_dict):
	x_vars_list = []
	y_vars_list = []
	for row in csv.DictReader(in_file_name):
		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28', '2015-06-28',
									'2015-07-28','2015-08-28','2015-09-28','2015-10-28','2015-11-28','2015-12-28'
									'2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28', '2016-06-28']:
			continue

		cust_id = int(row['ncodpers'])
		month = row['fecha_dato'][5:7]
		if(month!= "06" and month!= "12"):
			target_list = getTarget(row)
			cust_dict[month][cust_id] = target_list
			continue
		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col))
		sex = getIndex(row,'sexo')
		age = getAge(row)
		income = getRent(row)
		x_vars.append( age )
		x_vars.append( getCustSeniority(row))
		x_vars.append( income )
		x_vars.append(getMarriageIndex(age,sex,income))

		if row['fecha_dato'] == '2016-06-28':
			prev_target_list1 = cust_dict["05"].get(cust_id, [0]*22)
			prev_target_list2 = cust_dict["04"].get(cust_id, [0]*22)
			prev_target_list3 = cust_dict["03"].get(cust_id, [0]*22)
			prev_target_list4 = cust_dict["02"].get(cust_id, [0]*22)
			prev_target_list5 = cust_dict["01"].get(cust_id, [0]*22)

			diff = sum(prev_target_list1)-sum(prev_target_list2)
			if diff < 0:diff = 0
			prev_months_targets = [prev_target_list1,prev_target_list2,prev_target_list3,prev_target_list4,prev_target_list5]
			prev_target_list_5months=  listDotMultiply(prev_months_targets)
			x_vars_list.append(x_vars + prev_target_list1+ prev_target_list_5months +[sum(prev_target_list1),diff])  #prev_target_list[2]*prev_target_list[21]  
		elif row['fecha_dato'] == '2015-06-28':
			prev_target_list1 = cust_dict["05"].get(cust_id, [0]*22)
			prev_target_list2 = cust_dict["04"].get(cust_id, [0]*22)
			prev_target_list3 = cust_dict["03"].get(cust_id, [0]*22)
			prev_target_list4 = cust_dict["02"].get(cust_id, [0]*22)
			prev_target_list5 = cust_dict["01"].get(cust_id, [0]*22)

			prev_months_targets = [prev_target_list1,prev_target_list2,prev_target_list3,prev_target_list4,prev_target_list5]
			prev_target_list_5months=  listDotMultiply(prev_months_targets)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list1)] #purchase new product compared with last month
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list1) == 22
						# the additive quantity in 2015/05 compared to 2015/04
						diff = sum(prev_target_list1)-sum(prev_target_list2)
						if diff < 0:diff = 0
						x_vars_list.append(x_vars+prev_target_list1+ prev_target_list_5months+[sum(prev_target_list1),diff]) #prev_target_list[2]*prev_target_list[21]  + prev_target_04_list
						y_vars_list.append(ind)
		elif row['fecha_dato'] == '2015-12-28':
			prev_target_list1 = cust_dict["11"].get(cust_id, [0]*22)
			prev_target_list2 = cust_dict["10"].get(cust_id, [0]*22)
			prev_target_list3 = cust_dict["09"].get(cust_id, [0]*22)
			prev_target_list4 = cust_dict["08"].get(cust_id, [0]*22)
			prev_target_list5 = cust_dict["07"].get(cust_id, [0]*22)

			prev_months_targets = [prev_target_list1,prev_target_list2,prev_target_list3,prev_target_list4,prev_target_list5]
			prev_target_list_5months=  listDotMultiply(prev_months_targets)
			target_list = getTarget(row)

			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list1)] #purchase new product compared with last month
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list1) == 22
						diff = sum(prev_target_list1)-sum(prev_target_list2)
						if diff < 0:diff = 0
						x_vars_list.append(x_vars+prev_target_list1+ prev_target_list_5months+[sum(prev_target_list1),diff]) #prev_target_list[2]*prev_target_list[21]  + prev_target_04_list
						y_vars_list.append(ind)

	return x_vars_list, y_vars_list,cust_dict

			
def runXGB(train_X, train_y, seed_val=123):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.085
	param['max_depth'] = 4
	param['silent'] = 1
	param['num_class'] = 22
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 1
	param['subsample'] = 0.85
	param['colsample_bytree'] = 0.85
	param['seed'] = seed_val
	num_rounds = 140

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model

def score(Xtrain,y,random_state = 0):
	# score of 3-fold cross validation 
	kf = StratifiedKFold(y,n_folds =3, shuffle = True, random_state = random_state)
	pred = np.zeros((y.shape[0],22))
	num = 0;
	for itrain,itest in kf:
		Xtr,Xte = Xtrain[itrain,:],Xtrain[itest,:]
		ytr,yte = y[itrain],y[itest]
		model = runXGB(Xtr, ytr, seed_val=0)
		xgtest = xgb.DMatrix(Xte)
		preds = model.predict(xgtest)
		pred[itest,:] = preds
		print ("{:.5f}".format(log_loss(yte, preds)))
	print "total log loss is "+ str(log_loss(y, pred))

def mean(test):
	avgList = np.zeros((test.shape[0],22))
	for i in range(22):
		avgList[:,i] = (test[:,i]+test[:,i+22])/2
	return avgList

if __name__ == "__main__":
	start_time = datetime.datetime.now()
	print start_time
	data_path = ""
	train_file =  open("train_ver2.csv")
	cust_dict = {"01":{},"02":{},"03":{},"04":{},"05":{},"06":{},"07":{},"08":{},"09":{},"10":{},"11":{},"12":{}}
	x_vars_list, y_vars_list, cust_dict= processData(train_file, cust_dict)
	print(datetime.datetime.now()-start_time)

	train_X = np.array(x_vars_list)
	train_y = np.array(y_vars_list)
	print(np.unique(train_y))
	del x_vars_list, y_vars_list
	train_file.close()
	print(train_X.shape, train_y.shape)

	f = open('train1220_X.pkl','w')
	pickle.dump(train_X,f)
	f.close()
	f = open('train1220_y.pkl','w')
	pickle.dump(train_y,f)
	f.close()
	#cross validation
	score(train_X,train_y,random_state = 0)

	test_file = open(data_path + "test_ver2.csv")
	x_vars_list, y_vars_list, cust_dict = processData(test_file, cust_dict)
	test_X = np.array(x_vars_list)
	del x_vars_list
	test_file.close()
	print(test_X.shape)
	print(datetime.datetime.now()-start_time)

	print("Building model..")
	model = runXGB(train_X, train_y, seed_val=0)
	del train_X, train_y
	print("Predicting..")
	xgtest = xgb.DMatrix(test_X)
	preds = model.predict(xgtest)

	del test_X, xgtest
	print(datetime.datetime.now()-start_time)

	print("Getting the top products..")
	target_cols = np.array(target_cols)
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)[:,:8]
	test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
	out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
	out_df.to_csv('sub_xgb_new.csv', index=False)
	print(datetime.datetime.now()-start_time)