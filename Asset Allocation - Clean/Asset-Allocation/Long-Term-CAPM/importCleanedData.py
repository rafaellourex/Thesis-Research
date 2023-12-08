import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns


def analyzeQuantiles (data,):
    
    """"""""""""""""
    
    Receives:
            pandas dataframe
            
    Creates a df that analyzes outliers by creating 2 disparity features that measure the level of the outliers.
    The fist measure divides de 95% quantile by the 90% and the second divides the mean by the median
    
    Note: thresholds can be fyrther adjusted
    
    """""""""""""""""
    df_out = data
    #calculate the following quantiles [.25, .5, .75,0.9,0.95]
    quantile_disp = np.round(df_out.describe([.25, .5, .75,0.9,0.95]).T).iloc[:,:-1]
    #divide quantile 95 by 90  
    quantile_disp['disparity'] = (quantile_disp['95%'] / quantile_disp['90%']).replace(np.inf,0)
    #divide mean by median
    quantile_disp['mean_disp'] = (quantile_disp['mean'] / quantile_disp['50%']).replace(np.inf,0)

    quantile_disp = quantile_disp.sort_values(ascending=False,by = 'mean_disp')
    display(quantile_disp.iloc[:10])
    
    return(quantile_disp)

def replaceOutliers (data,upper_thresh, lower_thresh):
    """"""""""""""""
    
    Receives:
            pandas dataframe
            
    Replaces outliers with the nearest quantile, per example an observation of quantile 99 will be replaced by the quantile 90
    
    Note: thresholds can be fyrther adjusted
    
    """""""""""""""""
    df_out = data.copy()
    
    cols = df_out.select_dtypes(include = np.number).columns
    rem_cols = df_out.select_dtypes(exclude = np.number).columns
    #for each column
    for col in cols:
        #get quantile values
        upper_quant = df_out[col].quantile(upper_thresh)
        lower_quant = df_out[col].quantile(lower_thresh)
        #replace values higher than the quantile for the quantile 
        df_out.loc[df_out[col]>upper_quant, col] = upper_quant
        df_out.loc[df_out[col]<lower_quant, col] = lower_quant
        
        
    df_out = pd.concat([df_out[cols],
                        df_out[rem_cols]],
                      axis=1)
        
    return(df_out)

def splitData (df_model, date):
    
    # df_sector = pd.read_csv(f'{input_path}\companyInfo_sector.csv').drop(columns = 'Unnamed: 0')
    print('Nr of rows:')
    print(df_model.shape[0])

    #drop columns related to the target - avoid leakage
    leakage_cols = ['CAPM','adjCAPM','Target_0.15','Target_0']
    x = df_model.drop(columns = leakage_cols).copy()

    #set y
    y = df_model['Target_0.15'].copy()
    
    date_ = date
    #split the data into training and testing 
    x_train = x.loc[x.index.get_level_values(1)<date_].copy()
    y_train = y.loc[y.index.get_level_values(1)<date_,:].copy()
    x_test = x.loc[x.index.get_level_values(1)>=date_].copy()
    y_test = y.loc[y.index.get_level_values(1)>=date_,:].copy()

    #for each target column
    #create a dict containing each target 
    y_train_dict = dict()
    y_test_dict = dict()
    target_cols = ['Target_0','Target_0.15',]
    for col in target_cols:
        y_train_dict[col] = df_model.loc[df_model.index.isin(x_train.index),col].copy()
        y_test_dict[col] = df_model.loc[df_model.index.isin(x_test.index),col].copy()

    #perform outliers treatment
    #get x_train out
    low_thresh = 0.01
    high_thresh = 0.99
    
    df_out = x_train.copy()
    df_out_ = replaceOutliers(df_out,
                              high_thresh,
                              low_thresh)
    x_train_out = df_out_.copy()

    #get x_test out
    df_out_test = x_test.copy()
    df_out_test_ = replaceOutliers(df_out_test,
                                   high_thresh,
                                   low_thresh)
    x_test_out = df_out_test_.copy()
    return(x_train,x_train_out,x_test, x_test_out,y_train_dict ,y_test_dict, y_train,y_test)


def splitData_ (df_model, date):
    
    # df_sector = pd.read_csv(f'{input_path}\companyInfo_sector.csv').drop(columns = 'Unnamed: 0')
    print('Nr of rows:')
    print(df_model.shape[0])

    #drop columns related to the target - avoid leakage
    leakage_cols = ['CAPM','Target_0.15','Target_0']
    x = df_model.drop(columns = leakage_cols).copy()

    #set y
    y = df_model['Target_0.15'].copy()
    
    date_ = date
    #split the data into training and testing 
    x_train = x.loc[x.index.get_level_values(1)<date_].copy()
    y_train = y.loc[y.index.get_level_values(1)<date_,:].copy()
    x_test = x.loc[x.index.get_level_values(1)>=date_].copy()
    y_test = y.loc[y.index.get_level_values(1)>=date_,:].copy()

    #for each target column
    #create a dict containing each target 
    y_train_dict = dict()
    y_test_dict = dict()
    target_cols = ['Target_0','Target_0.15',]
    for col in target_cols:
        y_train_dict[col] = df_model.loc[df_model.index.isin(x_train.index),col].copy()
        y_test_dict[col] = df_model.loc[df_model.index.isin(x_test.index),col].copy()

    #perform outliers treatment
    #get x_train out
    low_thresh = 0.01
    high_thresh = 0.99
    
    df_out = x_train.copy()
    df_out_ = replaceOutliers(df_out,
                              high_thresh,
                              low_thresh)
    x_train_out = df_out_.copy()

    #get x_test out
    df_out_test = x_test.copy()
    df_out_test_ = replaceOutliers(df_out_test,
                                   high_thresh,
                                   low_thresh)
    x_test_out = df_out_test_.copy()
    return(x_train,x_train_out,x_test, x_test_out,y_train_dict ,y_test_dict, y_train,y_test)

def importCleanedData (dir,target='CAPM',):
    print(target)
    
    df_full_path_skew = f'{dir}\\LongTerm-DataPreparation_Skew.csv'
    target_path = f'{dir}\\Target_CAPM.csv'
    
    print(f'Import data from: {df_full_path_skew}')
    df_model = pd.read_csv(df_full_path_skew,)
    
    #rename  columns containing risk adjusted excess return (futAlpha is the original CAPM and excessReturns is the adjustedCAPM)
    if 'futAlpha' in df_model.columns:
        df_model = df_model.rename(columns = {'futAlpha':'CAPM'})
        
    if 'excessReturns' in df_model.columns:
        df_model = df_model.rename(columns = { 'excessReturns':'adjCAPM'})
        
    #import target data
    df_target = pd.read_csv(target_path)
    df_target['date'] = pd.to_datetime(df_target['date'],).dt.tz_localize(None)
    df_target['fillingDate'] = pd.to_datetime(df_target['fillingDate']).dt.tz_localize(None)
    df_target['excessReturns'] = df_target['newAlpha']
    
    #set index
    df_model = df_model.set_index(['symbol','date','fillingDate','year','quarter'])
    
    #set columns to drop
    to_drop = ['stockQuarterlyReturns',
       'indexQuarterlyReturns', 'stdStock', 'stdIndex', 'correlation', 'beta',
       'excessRet', 'newAlpha','calendarYear','stockPrice' ,'indexPrice','date_diff']

    #drop columns if they exist
    for col in to_drop:
        if col in df_model.columns:
            df_model = df_model.drop(columns = col)
            
    #if target == CAPM set the original target (CAPM)
    if target == 'CAPM':
        print('Original CAPM is being used as Target')
        to_drop = ['Target_0','Target_0.15','Target_0.05']
        for col in to_drop:
            if col in df_model.columns:
                df_model = df_model.drop(columns = col)
        thresh_list = [0,0.15]
        for thresh in thresh_list:
            df_model.loc[df_model['CAPM']>=thresh,f'Target_{thresh}'] = 1
            df_model[f'Target_{thresh}'] = df_model[f'Target_{thresh}'].fillna(0).astype(bool)
    return(df_model,df_target)



def importCleanedData_Normal(dir,target='CAPM',):
    """""""""""""""""""""
    Similar to importCleanedData but in this case, Data without Variable Transformtion is imported
    Used to run notebook: 
        Outliers_and_Skew_Anaysis.ipynb
    
    """""""""""""""""""""
    print(target)
    
    df_full_path_skew = f'{dir}\\LongTerm-DataPreparation.csv'
    target_path = f'{dir}\\Target_CAPM.csv'
    
    print(f'Import data from: {df_full_path_skew}')
    df_model = pd.read_csv(df_full_path_skew,)
    
    #rename  columns containing risk adjusted excess return (futAlpha is the original CAPM and excessReturns is the adjustedCAPM)
    if 'futAlpha' in df_model.columns:
        df_model = df_model.rename(columns = {'futAlpha':'CAPM'})
        
    if 'excessReturns' in df_model.columns:
        df_model = df_model.rename(columns = { 'excessReturns':'adjCAPM'})
        
    #import target data
    df_target = pd.read_csv(target_path)
    df_target['date'] = pd.to_datetime(df_target['date'],).dt.tz_localize(None)
    df_target['fillingDate'] = pd.to_datetime(df_target['fillingDate']).dt.tz_localize(None)
    df_target['excessReturns'] = df_target['newAlpha']
    
    #set index
    df_model = df_model.set_index(['symbol','date','fillingDate','year','quarter'])
    
    #set columns to drop
    to_drop = ['stockQuarterlyReturns',
       'indexQuarterlyReturns', 'stdStock', 'stdIndex', 'correlation', 'beta',
       'excessRet', 'newAlpha','calendarYear','stockPrice' ,'indexPrice','date_diff']

    #drop columns if they exist
    for col in to_drop:
        if col in df_model.columns:
            df_model = df_model.drop(columns = col)
            
    #if target == CAPM set the original target (CAPM)
    if target == 'CAPM':
        print('Original CAPM is being used as Target')
        to_drop = ['Target_0','Target_0.15','Target_0.05']
        for col in to_drop:
            if col in df_model.columns:
                df_model = df_model.drop(columns = col)
        thresh_list = [0,0.15]
        for thresh in thresh_list:
            df_model.loc[df_model['CAPM']>=thresh,f'Target_{thresh}'] = 1
            df_model[f'Target_{thresh}'] = df_model[f'Target_{thresh}'].fillna(0).astype(bool)
    return(df_model,df_target)

def importCleanedData_assess (dir,target='CAPM', cloud=False,):
    
    """""""""""""""""""""
    Similar to importCleanedData
    Used to run notebook: ModelAssess.ipynb
    
    """""""""""""""""""""
    print(target)
    
    df_full_path_skew = f'{dir}\\LongTerm-DataPreparation_Skew.csv'
    target_path = f'{dir}\\Target_CAPM.csv'
    
    print(f'Import data from: {df_full_path_skew}')
    df_model = pd.read_csv(df_full_path_skew,)
    
    #rename  columns containing risk adjusted excess return (futAlpha is the original CAPM and excessReturns is the adjustedCAPM)
    if 'futAlpha' in df_model.columns:
        df_model = df_model.rename(columns = {'futAlpha':'CAPM'})
        
    if 'excessReturns' in df_model.columns:
        df_model = df_model.rename(columns = { 'excessReturns':'adjCAPM'})
    #import target data
    
    df_target = pd.read_csv(target_path)
    df_target['date'] = pd.to_datetime(df_target['date'],).dt.tz_localize(None)
    df_target['fillingDate'] = pd.to_datetime(df_target['fillingDate']).dt.tz_localize(None)
    df_target['excessReturns'] = df_target['newAlpha']
    
    #set index
    df_model = df_model.set_index(['symbol','date','fillingDate','year','quarter'])
    
    #set columns to drop
    to_drop = ['stockQuarterlyReturns',
       'indexQuarterlyReturns', 'stdStock', 'stdIndex', 'correlation', 'beta',
       'excessRet', 'newAlpha','calendarYear','stockPrice' ,'indexPrice','date_diff']

    #drop columns if they exist
    for col in to_drop:
        if col in df_model.columns:
            df_model = df_model.drop(columns = col)
            
    #if target == CAPM set the original target (CAPM)
    if target == 'CAPM':
        print('Original CAPM is being used as Target')
        to_drop = ['Target_0','Target_0.15','Target_0.05']
        for col in to_drop:
            if col in df_model.columns:
                df_model = df_model.drop(columns = col)
        thresh_list = [0,0.15]
        for thresh in thresh_list:
            df_model.loc[df_model['CAPM']>=thresh,f'Target_{thresh}'] = 1
            df_model[f'Target_{thresh}'] = df_model[f'Target_{thresh}'].fillna(0).astype(bool)
            
    #if target == 'Adj_CAPM' set the adjusted CAPM target        
    if target == 'Adj_CAPM':
        print('Adjusted CAPM is being used as Target')
        to_drop = ['Target_0','Target_0.15','Target_0.05']
        for col in to_drop:
            if col in df_model.columns:
                df_model = df_model.drop(columns = col)
        thresh_list = [0,0.15]
        for thresh in thresh_list:
            df_model.loc[df_model['adjCAPM']>=thresh,f'Target_{thresh}'] = 1
            df_model[f'Target_{thresh}'] = df_model[f'Target_{thresh}'].fillna(0).astype(bool)      
    return(df_model,df_target)



def importCleanedData_skew (dir,):
    df_full_path_skew = f'{dir}\\df_clean_skew.csv'
    target_path = f'{dir}\\df_target.csv'
    
    #import target data
    df_model = df_model = pd.read_csv(df_full_path_skew,)
    
    df_target = pd.read_csv(target_path)
    df_target['date'] = pd.to_datetime(df_target['date'],).dt.tz_localize(None)
    df_target['fillingDate'] = pd.to_datetime(df_target['fillingDate']).dt.tz_localize(None)
    
    #set index
    df_model = df_model.set_index(['symbol','date','fillingDate','year','quarter'])
    
    return(df_model,df_target)

def importCleanedData_ (dir,):
    df_full_path_skew = f'{dir}\\df_clean.csv'
    target_path = f'{dir}\\df_target.csv'
    
    #import target data
    df_model = df_model = pd.read_csv(df_full_path_skew,)
    
    df_target = pd.read_csv(target_path)
    df_target['date'] = pd.to_datetime(df_target['date'],).dt.tz_localize(None)
    df_target['fillingDate'] = pd.to_datetime(df_target['fillingDate']).dt.tz_localize(None)
    
    #set index
    df_model = df_model.set_index(['symbol','date','fillingDate','year','quarter'])
    
    return(df_model,df_target)

def PCA (data,indexes, n_components):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as PCA
    
    
    data.reset_index()
    metric_features = list(data.select_dtypes(include=np.number).set_index(data.index).columns)
    x = data.loc[:,metric_features]
    
    stand_model = StandardScaler()
    stand_model = stand_model.fit(x)
    stand = stand_model.transform(x)
    s = pd.DataFrame(stand, columns= x.columns)
    s.set_index(data.index)
    
    pca = PCA(n_components=n_components,random_state=0)
    components = pd.DataFrame(pca.fit_transform(s), index = data.index)
    
    df = pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1))
    
    PC = pca
    plt.figure()

    cum_sum = PC.explained_variance_ratio_.cumsum()
    exp_var = PC.explained_variance_
    cov = PC.get_covariance()
    
    plt.figure(figsize=(15,10))
    sns.lineplot(data = df.loc[:,['Eigenvalue']])
    plt.axhline(1,ls='--')
    plt.title('Eigenvalues of each component', size = 15)
    plt.ylabel('Eigenvalue',size = 15)
    plt.xlabel('Nr of pricipal components',size = 15)
    plt.show()
    
    plt.figure(figsize=(10,10))
    sns.lineplot(data = df.loc[:,['Cumulative','Proportion']])
    plt.title('Cumulative % of total variance explained by the components', size = 15)
    plt.ylabel('Cumulative %',size = 15)
    plt.xlabel('Nr of pricipal components',size = 15)
    plt.show()
    print('The variance explained by each component is: ' + str(exp_var))
    print('The total variance explained by the components is: '+ str(sum(PC.explained_variance_ratio_)))
    
    for i in components.columns:
        components.rename(columns={i: f'PC_{i}'},inplace=True)
    
    return(df, components,PC,stand_model)


def runPCA (x_train_out,x_test):
    df_pca, components_pca, pca_model,scaler = PCA(x_train_out.fillna(0),x_train_out.index,53)

    test_stand = scaler.transform(x_test.fillna(0).select_dtypes(include=np.number))

    pca_components_test = pca_model.transform(test_stand)
    pca_components_test = pd.DataFrame(pca_components_test, index = x_test.index, columns = components_pca.columns)
    
    return(components_pca,pca_components_test )


