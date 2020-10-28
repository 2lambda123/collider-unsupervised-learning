from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from phenom_read_in_data import *

def fpr_tpr_calc(path, model, bkg_test, df_in):
    df = single_csv_to_df(path)
    if (len(df)-5)%2 != 0:
        num = str(int(((len(df.columns) - 5)/5) + 1))
        df['obj'+num]= -999
        df['E'+num] = -999
        df['pt'+num] = -999
        df['eta'+num] = -999
        df['phi'+num] = -999

    if (len(df.columns)< len(df_in.columns)):
        diff = int(((len(df_in.columns)-5)/5))-int(((len(df.columns)-5)/5))
        num = int(((len(df.columns) - 5)/5) + 1)
        for i in range(diff):
            df['obj'+str(num)]= -999
            df['E'+str(num)] = -999
            df['pt'+str(num)] = -999
            df['eta'+str(num)] = -999
            df['phi'+str(num)] = -999
            num+=1

        

    df = df.fillna(-999)
    variables = [entry for entry in df.columns if entry[0] == 'E'] + [entry for entry in df.columns if entry[0:2] == 'pt'] + [entry for entry in df.columns if entry[0:2] == 'et'] + [entry for entry in df.columns if entry[0:2] == 'ph']

    df = df[['process_ID']+variables]
    one_hot = pd.get_dummies(df['process_ID'])
    processes = one_hot.columns
    df.drop('process_ID', axis = 'columns', inplace = True)
    df = pd.concat([df, one_hot], sort = False, axis = 1)


    x = df[variables].values
    x_scaled = StandardScaler().fit_transform(x)
    df[variables] = x_scaled

    x_train, x_test, y_train, y_test = train_test_split(df[variables].values,
                                                    df[processes].values, 
                                                    shuffle = True,
                                                    random_state = 42,
                                                    test_size = 0.1)

    x_train = x_train.reshape((x_train.shape[0], 1, -1, 4))
    x_test = x_test.reshape((x_test.shape[0], 1, -1, 4))

    y_pred_sm = [model.predict(bkg_test.reshape(bkg_test.shape[0], 1, 1, -1, 4)[i]) for i in range(bkg_test.shape[0])]
    y_pred_bsm = [model.predict(x_test.reshape(x_test.shape[0], 1, 1, -1, 4)[i]) for i in range(x_test.shape[0])]
    y_pred_total = np.append(y_pred_sm, y_pred_bsm)

    y_pred_sm = np.array(y_pred_sm)
    y_pred_bsm = np.array(y_pred_bsm)
    bkg_test_flatten = np.array(bkg_test)
    x_test_flatten = np.array(x_test)

    y_pred_sm = y_pred_sm.flatten()
    y_pred_bsm = y_pred_bsm.flatten()
    bkg_test_flatten = bkg_test_flatten.flatten()
    x_test_flatten = x_test_flatten.flatten()

    y_true = np.append(np.zeros(len(y_pred_sm)), np.ones(len(y_pred_bsm)))
    y_pred_total_norm = y_pred_total/np.linalg.norm(y_pred_total)

    fpr, tpr, _ = roc_curve(y_true, y_pred_total_norm)

    return fpr, tpr


