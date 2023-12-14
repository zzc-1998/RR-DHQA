import pandas as pd
import numpy as np
from scipy import stats
import scipy
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

def performance(database = 'SJTU-H3D'):
    # define database information
    print(database)
    if database == 'SJTU-H3D':
        ref_csv = 'features/SJTU-H3D_3d_nss_ref.csv'
        dis_csv = 'features/SJTU-H3D_3d_nss_dis.csv'
        mos_csv = 'data_info/SJTU-H3D_datainfo.csv'
        k_fold = 5
        mos_scale = 5
    if database == 'DHHQA':
        ref_csv = 'features//DHHQA_3d_nss_ref.csv'
        dis_csv = 'features/DHHQA_3d_nss_dis.csv'
        mos_csv = 'data_info/DHHQA_datainfo.csv'
        k_fold = 4
        mos_scale = 100

    # define using parameterss
    start = 0
    end = 9
    mean_curvature_columns = [f'mean_curvature{i}' for i in range(start, end)]
    dihedral_angle_columns = [f'dihedral_angle{i}' for i in range(start, end)]
    defect_columns = [f'defect{i}' for i in range(start, end)]
    gray_columns = [f'gray{i}' for i in range(start,end)]  
    gradient_columns = [f'gradient{i}' for i in range(start, end)]
    gaussian_curvature_columns = [f'gaussian_curvature{i}' for i in range(start, end)]
    selected_columns = mean_curvature_columns + gaussian_curvature_columns + dihedral_angle_columns + defect_columns +\
        gradient_columns + gray_columns 


    ref_data = pd.read_csv(ref_csv)
    dis_data = pd.read_csv(dis_csv)
    error = np.array(ref_data[selected_columns]-dis_data[selected_columns])
    error = -1 / (1 + np.exp(error))
    mos = np.array(pd.read_csv(mos_csv)['MOS'])/mos_scale
    


    total_num = len(mos)
    fold_num = int(total_num/k_fold)
    best_all = np.zeros([k_fold, 4])

    # do k-fold cross validation
    for i in range(k_fold): 
        # do SVR training
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(np.concatenate((error[:i*fold_num,:], error[(i+1)*fold_num:,:]), axis=0))
        y_train = np.concatenate([mos[:i*fold_num], mos[(i+1)*fold_num:]])
        X_test = scaler.transform(error[i*fold_num:(i+1)*fold_num,:])
        svr = SVR(kernel='rbf', C = 1, epsilon=0.05)
        svr = svr.fit(X_train, y_train)
        y_output = svr.predict(X_test)

        y_test = mos[i*fold_num:(i+1)*fold_num]
        y_output_logistic = y_output
        test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
        test_SROCC = stats.spearmanr(y_output, y_test)[0]
        test_RMSE = np.sqrt(((y_output_logistic*mos_scale-y_test*mos_scale) ** 2).mean())
        test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
        best_all[i, :] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
        print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))
    best_mean = np.mean(best_all, 0)
    print('*************************************************************************************************************************')
    print("The mean performance: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1], best_mean[2], best_mean[3]))
    print('*************************************************************************************************************************')


performance('SJTU-H3D')
performance('DHHQA')