#Final experimental platform

#before completion, note if any of these can be removed.
import pandas as p
import numpy as np
from sklearn import *
import time
import math
import tracemalloc #memory evaluation
import os

# execute - run tracemalloc
def execute(funct, *args):
    tracemalloc.start()
    funct(*args)
    current,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    f.write(f"Current memory usage is {current / 10 ** 6 } MB;'\
    ' peak was {peak / 10 ** 6} MB\n")

def execute_time(funct, *args):
    start = time.perf_counter()
    funct(*args)
    end = time.perf_counter()
    f.write("execute in " + str((end-start)) + " seconds\n")

def timevar():
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    return str(timestamp)

# data loading
def load_data(filein=None):
    if filein == None:
        print("Is the data to process contained in the same folder as the .py file? (Y/N)")
        choice = input("> ")
        folderpath = None
        if choice.upper() == 'N':
            folderpath = input("Please enter the folder path: ")
        filename = input("Please enter a complete filepath to load: ")
        if folderpath is not None:
            filename = folderpath + filename
    else:
        filename = filein
    set = p.read_csv(filename, parse_dates=True,dtype={'Label':'string'})
    return set #return a dataframe object loaded from a CSV

def first_processing(df):
    #if label_bin exists, drop it
    if 'Label_Bin' in df.columns:
        del df['Label_Bin']
    #if timestamps exists, drop it
    if 'Timestamp' in df.columns:
        del df['Timestamp']
    checkframe(df)
    attacks = df.Label.unique()
    f.write(f"The data contains the following attacks: {attacks}\n")
    f.write("Shuffling data randomly...\n")
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def checkframe(data_obj):
    #check for infinite or NaN values
    found = False
    f.write("checking for NaN values\n")
    if (data_obj.isnull().values.any()) == True:
        #iterate through columns
        cols = list(data_obj)
        for i in cols:
            if (data_obj[i].isnull().values.any()) == True:
                f.write("column " + i + " has " + str(data_obj[i].isnull().sum()) + " NaN values\n")
                found = True
    else:
        f.write("No NaN values found\n")
    f.write("checking for infinite values\n")
    cols = list(data_obj)
    for i in cols:
        # if it's a string, ignore
        if isinstance(data_obj[i][1], str) == False:
            #not a string, check for infinite
            if (np.all(np.isfinite(data_obj[i]))) == False:
                f.write("column " + i + " has infinite values\n")
                found = True
    f.write("infinite check complete\n")
    if found == True:
        start_rows = data_obj.shape[0]
        f.write("Dropping rows with infinite or NaN values\n")
        data_obj.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_obj.dropna(inplace=True)
        end_rows = data_obj.shape[0]
        f.write(f"Removed {start_rows-end_rows} rows\n")
    return found

def decision_tree(dfX, dfY, eval=False):
    clf = tree.DecisionTreeClassifier()
    split = 0.4 #test/train split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dfX, dfY,test_size = split, random_state=5)
    del dfX
    del dfY
    clf.fit(x_train, y_train)
    f.write("Tree fit complete\n")
    results = clf.predict(x_test) #predictions made, end of run unless evaluating
    if eval == True:
        #run evaluation criteria
        f.write("Confusion matrix:\n")
        f.write(str(metrics.confusion_matrix(y_test, results))+"\n")
        f.write("F1 score:\n")
        f.write(str(metrics.f1_score(y_test, results, average='micro'))+"\n")
        f.write("(F1 score uses micro average for multi-class classification)\n")
    return clf

def batch_decision_tree(dfX, dfY):
    f.write("Entering decision tree - memory\n")
    execute(decision_tree, dfX, dfY)
    f.write("Entering decision tree - timed\n")
    execute_time(decision_tree,dfX, dfY)
    f.write("Evaluating decision tree metrics\n")
    decision_tree(dfX, dfY, True)

def LLE(dfX, dfY, eval=False, method='standard'):
    if method == 'modified':
        #required to have n_neighbors > components
        n_neighbors = max(5, component_selection+1) #default is 5, min is as specified
        lle = manifold.LocallyLinearEmbedding(n_components = component_selection, n_neighbors = n_neighbors, method=method)
    elif method == 'hessian':
        v = math.ceil(component_selection*(component_selection+3)/2)+1
        n_neighbors = max(5, v) #default is 5, min is as specified
        lle = manifold.LocallyLinearEmbedding(n_components = component_selection, n_neighbors = n_neighbors, method=method)
    elif method == 'ltsa':
        #maximum components of 5
        lle = manifold.LocallyLinearEmbedding(n_components = 5, method = method)
    else:
        lle = manifold.LocallyLinearEmbedding(n_components = component_selection, method=method)
    print("Fitting manifold")
    lle.fit(dfX.iloc[0:manifold_subset])
    dfX = dfX.iloc[manifold_subset+1:]
    dfY = dfY.iloc[manifold_subset+1:]
    print("Transforming data in batch format...\n")
    start = 0
    end = dfX.shape[0]-1
    init = False
    while start < end:
        if init == False:
            #initialize the array, append if it has already been initialized
            df_out = lle.transform(dfX[start:start+batch])
            init = True
        else:
            df_out = np.concatenate((df_out,lle.transform(dfX[start:start+batch])))
        start = start+batch
    dfX = df_out
    print("Batch transform complete\n")
    if eval == True:
        #run evaluation criteria
        f.write("LLE evaluation statistics\n")
        f.write(f"Reconstruction error associated with this embedding: {lle.reconstruction_error_}\n")
        f.write(f"Parameters: {lle.get_params}\n")
    return [dfX, dfY]

def batch_LLE(dfX, dfY, method='standard'):
    f.write("LLE - evaluation\n")
    LLE(dfX, dfY, True, method)
    f.write("LLE - memory\n")
    execute(LLE, dfX, dfY, False, method)
    f.write("LLE - timed\n")
    execute_time(LLE, dfX, dfY, False, method)

def isomap(dfX, dfY, eval = False):
    f.write("Creating embedding vectors...\n")
    iso = manifold.Isomap(n_components=component_selection)
    iso.fit(dfX.iloc[0:manifold_subset])
    dfX = dfX.iloc[manifold_subset+1:]
    dfY = dfY.iloc[manifold_subset+1:]
    print("Transforming data in batch format...\n")
    start = 0
    end = dfX.shape[0]-1
    init = False
    while start < end:
        if init == False:
            #initialize the array, append if it has already been initialized
            df_out = iso.transform(dfX[start:start+batch])
            init = True
        else:
            df_out = np.concatenate((df_out,iso.transform(dfX[start:start+batch])))
        start = start+batch
    dfX = df_out
    print("Batch transform complete\n")
    if eval == True:
        f.write("Isomap reconstruction error is not available.\n")
    return [dfX, dfY]

def batch_isomap(dfX, dfY):
    f.write("Isomap - memory\n")
    execute(isomap, dfX, dfY)
    f.write("Isomap - timed\n")
    execute_time(isomap, dfX, dfY)
    f.write("Isomap - stats\n")
    isomap(dfX, dfY, True)

def spectral(dfX, dfY, eval=False):
    f.write("Creating embedding vectors...\n")
    spec = manifold.SpectralEmbedding(n_components = component_selection)
    dfX = dfX.iloc[manifold_subset+1:]
    dfY = dfY.iloc[manifold_subset+1:]
    dfX = spec.fit_transform(dfX)
    dfX = df_out
    f.write("transform complete\n")
    if eval == True:
        f.write("Spectral reconstruction error: " + str(spec.reconstruction_error_)+"\n")
    return [dfX, dfY]

def batch_spectral(dfX, dfY):
    f.write("Spectral - memory\n")
    execute(spectral, dfX, dfY)
    f.write("Spectral - timed\n")
    execute_time(spectral, dfX, dfY)
    f.write("Spectral - stats\n")
    spectral(dfX, dfY, True)

def random_forest(dfX, dfY, eval=False):
    split = 0.4 #hold 40% for classification
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dfX, dfY, test_size=split, random_state=5)
    del dfX
    del dfY
    clf = ensemble.RandomForestClassifier()
    clf.fit(x_train, y_train)
    f.write("Tree fit complete\n")
    results = clf.predict(x_test)
    if eval == True:
        f.write("Confusion matrix:\n")
        f.write(str(metrics.confusion_matrix(y_test, results))+"\n")
        f.write("f1 score:\n")
        f.write(str(metrics.f1_score(y_test, results, average='micro'))+"\n")
    return results

def batch_random_forest(dfX, dfY):
    f.write("Random forest - memor\ny")
    execute(random_forest, dfX, dfY)
    f.write("Random forest - timed\n")
    execute_time(random_forest, dfX, dfY)
    f.write("Random forest - evaluated\n")
    random_forest(dfX, dfY, True)
    return

def k_nearest(dfX, dfY, eval = False):
    split = 0.4 #hold 40% for classification
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dfX, dfY, test_size=split, random_state=5)
    del dfX
    del dfY
    neigh = neighbors.KNeighborsClassifier()
    f.write("Fitting...\n")
    neigh.fit(x_train, y_train)
    results = neigh.predict(x_test)
    if eval == True:
        f.write("Confusion matrix:\n")
        f.write(str(metrics.confusion_matrix(y_test, results))+"\n")
        f.write("F1 score:\n")
        f.write(str(metrics.f1_score(y_test, results, average='micro'))+"\n")
    return results

def batch_k_nearest(dfX, dfY):
    print("KNN - memory")
    f.write("KNN - memory\n")
    execute(k_nearest, dfX, dfY)
    print("KNN - time")
    f.write("KNN - time\n")
    execute_time(k_nearest, dfX, dfY)
    print("KNN - evaluation")
    f.write("KNN - evaluation\n")
    k_nearest(dfX, dfY, True)

def ada(dfX, dfY, eval=False):
    split = 0.4 #hold 40% for classification
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dfX, dfY, test_size=split, random_state=5)
    del dfX
    del dfY
    clf = ensemble.AdaBoostClassifier()
    f.write("Fitting...\n")
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)
    if eval == True:
        f.write("Confusion matrix:\n")
        f.write(str(metrics.confusion_matrix(y_test, results))+"\n")
        f.write("F1 score:\n")
        f.write(str(metrics.f1_score(y_test, results, average='micro'))+"\n")
    return results

def batch_ada(dfX, dfY):
    f.write("Adaboost - memory\n")
    execute(ada, dfX, dfY)
    f.write("Adaboost - time\n")
    execute_time(ada, dfX, dfY)
    f.write("Adaboost - evaluation\n")
    ada(dfX, dfY, True)

def range_scale(dfX):
    #minmax scale to reduce unit scale and place all values between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dfX)
    scaler.transform(dfX)
    return dfX

def chooser():
    choices = []
    print("Please select from the following options:")
    print("First, binary (1) or multi-label (2) classification?")
    choices.append(int(input("> ")))
    print("Second, select dimensionality reduction:")
    print("1 - None")
    print("2 - Locally Linear Embedding")
    print("3 - Locally Linear Embedding (Measured)")
    print("4 - IsoMap")
    print("5 - IsoMap (Measured)")
    print("6 - Modified LLE")
    print("7 - Modified LLE (Measured)")
    print("8 - LTSA")
    print("9 - LTSA (Measured)")
    print("10 - Hessian LLE")
    print("11 - Hessian LLE (Measured)")
    print("12 - Spectral Embedding (Laplacian Eigenmaps)")
    print("13 - Spectral Embedding (Measured)")

    choices.append(int(input("> ")))
    print("Third, classifier:")
    print("1 - Decision Tree")
    print("2 - Decision Tree (Measured)")
    print("3 - Random Forest")
    print("4 - Random Forest (Measured)")
    print("5 - k-Nearest Neighbors")
    print("6 - k-Nearest Neighbors (Measured)")
    print("7 - Adaboost")
    print("8 - Adaboost (Measured)")
    print("9 - None")
    choices.append(int(input("> ")))
    print(f"Choices are: {choices}")

    print("Choose data scaling method")
    print("1 - None")
    print("2 - Range Scale")
    choices.append(int(input("> ")))
    return choices

def runner(choices, df):
    #preprocessing
    if choices[0] == 1:
        f.write("Adjusting classification labels to Benign/Malicious only.\n")
        uniques = df.Label.unique()
        for i in uniques:
            if i != 'Benign':
                df['Label'] = df['Label'].replace([i],'Malicious')
    df = first_processing(df)
    dfY = df['Label']
    dfX = df[df.columns[0:77]]

    #scaling, if chosen
    if choices[3] == 2:
        dfX = range_scale(dfX)
        f.write("Scaling complete!\n")

    #dimensionality reduction
    if choices[1] == 1:
        #no dimensionality reduction
        f.write("No dimensionality reduction chosen.\n")
    elif choices[1] == 2:
        f.write("Running locally linear embedding\n")
        var = LLE(dfX, dfY)
        dfX = var[0]
        dfY = var[1]
        f.write("LLE complete\n")
    elif choices[1] == 3:
        f.write("Running batched LLE 5 times\n")
        for i in range(3):
            batch_LLE(dfX, dfY)
        f.write("LLE complete\n")
    elif choices[1] == 4:
        f.write("Running Isomap\n")
        var = isomap(dfX, dfY)
        dfX = var[0]
        dfY = var[1]
        f.write("Isomap complete\n")
    elif choices[1] == 5:
        f.write("Running Isomap (Measured) 5 times\n")
        for i in range(3):
            batch_isomap(dfX, dfY)
        f.write("Isomap complete\n")
    elif choices[1] == 6:
        f.write("Modified LLE\n")
        var = LLE(dfX, dfY, False, 'modified')
        dfX = var[0]
        dfY = var[1]
        f.write("Modified LLE complete\n")
    elif choices[1] == 7:
        f.write("Modified LLE (Measured)\n")
        for i in range(3):
            batch_LLE(dfX, dfY, 'modified')
        f.write("Modified LLE complete\n")
    elif choices[1] == 8:
        f.write("LTSA\n")
        var = LLE(dfX, dfY, 'ltsa')
        dfX = var[0]
        dfY = var[1]
        f.write("LTSA complete\n")
    elif choices[1] == 9:
        f.write("LTSA (Measured)\n")
        for i in range(3):
            batch_LLE(dfX, dfY, 'ltsa')
        f.write("LTSA complete\n")
    elif choices[1] == 10:
        f.write("Hessian LLE\n")
        var = LLE(dfX, dfY, False, 'hessian')
        dfX = var[0]
        dfY = var[1]
        f.write("Hessian LLE complete\n")
    elif choices[1] == 11:
        f.write("Hessian LLE (Measured)\n")
        for i in range(3):
            batch_LLE(dfX, dfY, 'hessian')
        f.write("Hessian LLE complete\n")
    elif choices[1] == 12:
        f.write("Spectral Embedding (Laplacian Eigenmaps)\n")
        spectral(dfX, dfY)
        f.write("Spectral embedding complete\n")
    elif choices[1] == 13:
        f.write("Spectral Embedding (Laplacian Eigenmaps - Measured)\n")
        for i in range(3):
            batch_spectral(dfX, dfY)
        f.write("Spectral embedding complete\n")

    #classifier
    if choices[2] == 1:
        #run decision tree batch
        f.write("Running decision tree\n")
        f.write(f"X shape is {dfX.shape} and Y shape is {dfY.shape}\n")
        decision_tree(dfX, dfY)
    elif choices[2] == 2:
        #one batch of decision tree
        f.write("Running decision tree (measured)\n")
        f.write(f"X shape is {dfX.shape} and Y shape is {dfY.shape}\n")
        for i in range(3):
            batch_decision_tree(dfX, dfY)
    elif choices[2] == 3:
        f.write("Running random forest\n")
        f.write(f"X shape is {dfX.shape} and Y shape is {dfY.shape}\n")
        random_forest(dfX, dfY)
        f.write("Complete!\n")
    elif choices[2] == 4:
        f.write("Running random forest (measured)\n")
        f.write(f"X shape is {dfX.shape} and Y shape is {dfY.shape}\n")
        for i in range(2):
            batch_random_forest(dfX, dfY)
        f.write("Complete!\n")
    elif choices[2] == 5:
        f.write("Running KNN\n")
        k_nearest(dfX, dfY)
        f.write("Complete!\n")
    elif choices[2] == 6:
        f.write("Running KNN (measured)\n")
        f.write(f"X shape is {dfX.shape} and Y shape is {dfY.shape}\n")
        for i in range(3):
            batch_k_nearest(dfX, dfY)
        f.write("Complete!\n")
    elif choices[2] == 7:
        f.write("Running adaboost\n")
        ada(dfX, dfY)
        f.write("Complete!\n")
    elif choices[2] == 8:
        f.write("Running adaboost (measured)\n")
        f.write(f"X shape is {dfX.shape} and Y shape is {dfY.shape}\n")
        for i in range(3):
            batch_ada(dfX, dfY)
        f.write("Complete!\n")
    else:
        f.write("No classifier run.\n")
    f.write("Script complete!\n")

#global variables
component_selection = 20
manifold_subset = 2000
batch = 2000 #rows to transform at a time in manifolds that are batched
#take user input
choices = chooser()
# #multiclass, LLE with stats, decision tree, no range
df = load_data() #make sure to run this after every "choices" block
f = open('lastoutput.txt', 'w')
#run on choices and input dataframe
runner(choices, df)
f.close()

# print("Moving to master plan")
# choices = chooser()
# directory = '/home/id/Documents/Thesis/Processed Traffic Data for ML Algorithms/'
# for filename in os.listdir(directory):
#     if filename[0:5] != ".~loc":
#         strtemp = ""
#         for i in choices:
#             strtemp = strtemp + str(i)
#         f = open(strtemp + filename + ".txt", 'w')
#         print("Filename: " + filename + " choices " + strtemp)
#         df = load_data(directory + filename)
#         print("Dataframe is" + str(df.shape))
#         runner(choices, df)
#         f.close()

# directory = '/home/id/Documents/Thesis/Processed Traffic Data for ML Algorithms/'
# manifolds_to_run = [2,4,6,8,10]
# classifiers_run_A = [2,4]
# classifiers_run_B = [6,8]
#
# for j in manifolds_to_run:
#     #for y in classifiers_run_A:
#     for y in classifiers_run_B:
#         print("Run B")
#         choices = [0, j, y, 2]
#         print(str(choices))
#         for filename in os.listdir(directory):
#             if filename[0:5] != ".~loc":
#                 strtemp = ""
#                 for i in choices:
#                     strtemp = strtemp + str(i)
#                 f = open(strtemp + filename + ".txt", 'w')
#                 print("Filename: " + filename + " choices " + strtemp)
#                 df = load_data(directory + filename)
#                 print("Dataframe is" + str(df.shape))
#                 runner(choices, df)
#                 f.close()

directory = '/home/id/Documents/Thesis/Processed Traffic Data for ML Algorithms/'
# choices = [0,3,9,0]
# filename = 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'
# strtemp = ""
# for i in choices:
#     strtemp = strtemp + str(i)
# f = open(strtemp + fsilename + ".txt", 'w')
# print("Filename: " + filename + " choices " + strtemp)
# df = load_data(directory + filename)
# print("Dataframe is" + str(df.shape))
# runner(choices, df)
# f.close()
# manifolds_to_run = [3]
# scale_to_run = [0,1]
#
# for j in manifolds_to_run:
#     for y in classifiers_run:
#         choices = [0, j, 9, y]
#         print(str(choices))
#         for filename in os.listdir(directory):
#             if filename[0:5] != ".~loc":
#                 strtemp = ""
#                 for i in choices:
#                     strtemp = strtemp + str(i)
#                 f = open(strtemp + filename + "_B.txt", 'w')
#                 print("Filename: " + filename + " choices " + strtemp)
#                 df = load_data(directory + filename)
#                 print("Dataframe is" + str(df.shape))
#                 runner(choices, df)
#                 f.close()
