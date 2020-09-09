# Aug 18 - Ian Dalsin
# Python script to analyze a CSV and output some descriptive statistics
# mostly using pandas, see User Guide

from sklearn import preprocessing
import pandas as p
import numpy as np
import os

def covar_matrix(data_obj, traffic_type, filename_root):
    print("Generating covariance matrix for: " + traffic_type)
    t = data_obj.cov()
    new_file = traffic_type + "_" + filename_root[:-4]+"_covar.csv"
    t.to_csv(new_file)

def corr_matrix(data_obj, traffic_type, filename_root):
    print("Generating correlation matrix for: " + traffic_type)
    t = data_obj.corr()
    new_file = traffic_type + "_" + filename_root[:-4]+"_corr.csv"
    t.to_csv(new_file)

def column_statistics(data_obj, file_out):
    # writes all stats per column to csv in an array-type format
    cols = list(data_obj)
    list_operations = ['Column Name', 'Mean', 'Mean Abs Deviation', 'Median', 'Min', 'Max', 'Bessel St Dev', 'Unbiased Variance', 'Standard Error of the Mean', '25th Percentile', '50th Percentile', '75th Percentile']
    file_out.write(str(list_operations) + "\n")
    for i in cols:
        #if value is numeric
        #original: if isinstance(data_obj[i][1], str) == False:
        if isinstance(data_obj[i].iloc[0], str) == False:
            temp = str(i) + ", " + str(data_obj[i].mean()) + ", " + str(data_obj[i].mad()) + ", " + str(data_obj[i].median()) + ", " + str(data_obj[i].min()) + ", " + str(data_obj[i].max()) + ", " + str(data_obj[i].std()) + ", " + str(data_obj[i].var()) + ", " + str(data_obj[i].sem()) + ", " + str(np.percentile(data_obj[i],25)) + ", " + str(np.percentile(data_obj[i],50)) + ", "+ str(np.percentile(data_obj[i],75))
            file_out.write(temp + "\n")
        else:
            continue
    return

def one_file_stats(filename):
    file1 = filename
    print("File to load: " + file1)
    #output file extension set to ODS for Libreoffice, or CSV for plain text
    out_ext = 'csv'
    outfile = (file1[(file1.rfind("/")+1):-4] + '_outfile.' + out_ext)
    print("Output to : " + outfile)
    f = open(outfile, 'w')
    f.write("Output for: " + file1[(file1.rfind("/")+1):] + "\n")

    set1 = p.read_csv(file1, parse_dates=True,dtype={'Label':'string'})
    print("dataset loaded: set1")
    print("This dataset contains the follow traffic object categories: " + str(set1.Label.unique()))

    x1 = set1
    del x1['Timestamp']
    print("deleting timestamp, can't do any math with it anyways")

    f.write("Column statistics - all -" + str(x1.shape[0]) + " rows \n")
    column_statistics(x1, f)
    # if uncommented below, script will create covariance and correlation matrices
    #covar_matrix(x1, 'All',file1[(file1.rfind("/")+1):-4])
    corr_matrix(x1, 'All',file1[(file1.rfind("/")+1):-4])
    print("Writing column statistics for traffic subtypes...")
    uniques = x1.Label.unique()

    for i in uniques:
        print("Subtype: " + i)
        temp_df = x1.loc[x1['Label'] == i]
        f.write("\n")
        f.write("Traffic subtype: " + str(i) + " - "+ str(temp_df.shape[0]) + "  rows \n")
        column_statistics(temp_df, f)
        # if uncommented below, script will create covariance and correlation matrices
        #covar_matrix(temp_df,i,file1[(file1.rfind("/")+1):-4])
        corr_matrix(temp_df,i,file1[(file1.rfind("/")+1):-4])
        f.write("\n")

    print("Attack output complete!")

    print("Writing stats for key ports")
    #print for certain port numbers as if they were unique attacks - ie. statistics for port 80 traffic
    key_ports = [80, 21, 22, 8080]
    #if there is no traffic in these key ports, the dataframe selected will contain zero rows
    #select only the benign traffic in the key ports, malicious traffic excluded
    for i in key_ports:
        print("Subtype: port " + str(i))
        temp_df = x1.loc[x1['Label'] == "Benign"]
        temp_df = temp_df.loc[temp_df['Dst Port'] == i]
        if temp_df.shape[0] > 0:
            f.write("\n")
            f.write("Traffic subtype: port " + str(i) + " - " + str(temp_df.shape[0]) + " rows \n")
            column_statistics(temp_df, f)
            f.write("\n")
        else:
            print("No traffic for port " + str(i))

    print("Key ports output complete")

    f.close()
    return

#the options below represent different 'calls' based on which lines are commented/uncommented

#call for all files in directory, batch process
directory = '/home/id/Documents/Thesis/Processed Traffic Data for ML Algorithms/'
for filename in os.listdir(directory):
    if filename[0:5] != ".~loc":
        one_file_stats(directory + filename)
        print("File complete!")

# call for one file only
#one_file_stats('~/Documents/Thesis/Processed Traffic Data for ML Algorithms/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv')
