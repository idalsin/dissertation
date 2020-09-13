import numpy as np
import matplotlib.pyplot as plt
import pandas as p

# Adjust the below variables in order to generate scatterplots for chosen CSV files.
file1 = '/home/id/Documents/Thesis/Processed Traffic Data for ML Algorithms/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'
key_port = 80
date = '15-02-2018'

set1 = p.read_csv(file1, parse_dates=True, dtype={'Label':'string'})
print("File loaded successfully...")
set1 = set1.loc[set1['Dst Port'] == key_port]

labels = set1.Label.unique()
print("Set contains: " + str(labels))

fig, ax = plt.subplots()
for label in labels:
    df = set1.loc[set1['Label']==label]
    x = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
    y = df['Flow Duration'] / (1000000)
    n = df.shape[0]
    ax.scatter(x,y,label=label,marker="+")

ax.legend()
ax.grid(True)
if date == '15-02-2018':
    plt.xlim(0,2000)
    # plt.ylim(0,15)
plt.ylabel('Flow Duration (seconds)')
plt.xlabel('Total Packets (combined directions)')
plt.title("Total Packets vs Flow Duration - Port "+ str(key_port) +" Traffic " + date)
plt.savefig("Total Packets vs Flow Duration - Port "+ str(key_port) +" Traffic " + date + '.png')
plt.show()

#fwd/bwd packets
fig, ax = plt.subplots()
for label in labels:
    df = set1.loc[set1['Label']==label]
    x = df['Tot Fwd Pkts']
    y = df['Tot Bwd Pkts']
    n = df.shape[0]
    ax.scatter(x,y,label=label,marker="+")

ax.legend()
ax.grid(True)
plt.ylabel('Tot Bwd Pkts')
plt.xlabel('Tot Fwd Pkts')
plt.title('Total Packets - Port '+ str(key_port)+' Traffic ' + date)
if date == '15-02-2018':
    plt.xlim(0,2000)
    plt.ylim(0,5000)
plt.savefig('Total Packets - Port '+ str(key_port)+' Traffic ' + date + '.png')
plt.show()
