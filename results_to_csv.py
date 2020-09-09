#parse my results txt files into a CSV of some type
import os #for later directory parsing

def write_array(arr, extra=None, name=None):
    for i in arr:
        if extra == None:
            outfile.write(str(i)+'\n')
        else:
            outfile.write(extra + ", " + name + ", " + str(i) + '\n')
    return

def num_only(arr):
    new_arr = []
    for x in arr:
        x = ''.join(c for c in x if c.isdigit() or c == '.')
        new_arr.append(x)
    return new_arr

def parse_csv(filename):
    filein = open(directory+filename, 'r')
    lines = filein.readlines()
    count = len(lines)-1
    temp = filename.find("_")
    temp = max(temp, 20)

    outfile.write("Sequence: " + str(sequence)+"\n")
    outfile.write("Date: " + filename[len(sequence):temp]+"\n")
    outfile.write("Filename, Date, Labelling, Dimensionality Reduction, Classifier, Scaling, Variable, Result \n")
    memory = []
    attacks = []
    time = []
    rows_dropped = []
    confusion_matrix = []
    f1_score = []
    recon_error = []
    for x in range(count):
        temp_line = lines[x]
        #memory
        if temp_line.find("peak was ") > 0:
            memory.append(temp_line[temp_line.find("peak was ")+len("peak was "):])
        #time
        if temp_line.find("seconds")>0:
            time.append(temp_line)
        #attacks
        if temp_line.find("Benign")>0:
            attacks.append(temp_line)
        #rows dropped, row count
        if (temp_line.find("emoved")>0):
            rows_dropped.append(temp_line)
        #confusion matrix
        if temp_line.find("onfusion matrix:")>0:
            mat = [lines[x+1], lines[x+2]]
            confusion_matrix.append(mat)
        #f1 score
        if temp_line.find("1 score:")>0:
            f1_score.append(lines[x+1])
        #reconstruction error
        if temp_line.find("associated with this embedding:")>0:
            recon_error.append(temp_line[temp_line.find("associated with this embedding:")+len("associated with this embedding:")+1:])



    #clean all my arrays, removing units and such
    memory = num_only(memory)
    time = num_only(time)
    rows_dropped = num_only(rows_dropped)
    f1_score = num_only(f1_score)
    recon_error = num_only(recon_error)
    #confusion matrix is a bit of an edge case
    if len(confusion_matrix)>0:
        for i in range(len(confusion_matrix[0])):
            #returns rows of matrix
            confusion_matrix[0][i] = ''.join(c for c in confusion_matrix[0][i] if c.isdigit() or c == ' ')
        for i in range(len(confusion_matrix[0])):
            confusion_matrix[0][i] = confusion_matrix[0][i].split()
        confusion_matrix = confusion_matrix[0]
        for i in range(len(confusion_matrix)):
            for x in range(len(confusion_matrix[i])):
                confusion_matrix[i][x] = int(confusion_matrix[i][x])

    #write row
    #first, turn sequence into a string of choices
    #ie. isomap to decision tree, scaled, multiclass
    if len(sequence) == 5:
        #ex 01042
        bin_mult = int(sequence[0])
        dimred = int(sequence[1:3])
        classifier = int(sequence[3])
        scaling = int(sequence[4])
    else:
        bin_mult = int(sequence[0])
        classifier = int(sequence[2])
        dimred = int(sequence[1])
        scaling = int(sequence[3])
    if bin_mult == 1:
        options_string = "Binary, "
    else:
        options_string = "Multiclass, "
    if scaling == 1:
        scaling_string = " no scaling"
    else:
        scaling_string = " range scaling"
    # print("Debug: " + str(sequence) + ", len " + str(len(sequence)))
    # print("Bin:" + str(bin_mult))
    # print("DimRed: " + str(dimred))
    # print("class: " + str(classifier))
    # print("scaling: " + str(scaling))
    dimred_opt = ['None', 'LLE', 'LLE', 'Isomap', 'Isomap', 'MLLE', 'MLLE', 'LTSA', 'LTSA', 'HLLE', 'HLLE', 'Spectral', 'Spectral']
    dimred_str = dimred_opt[dimred-1]

    classifier_opt = ['Decision Tree', 'Decision Tree', 'Random Forest', 'Random Forest', 'KNN', 'KNN', 'Adaboost', 'Adaboost', 'None']
    classifier_str = classifier_opt[classifier-1]
    #be sure to substract one for indexing
    options_string = options_string + dimred_str + ', ' + classifier_str + ',' + scaling_string
    print(options_string)
    print(str(memory))
    print(str(time))
    print(str(attacks))
    print(str(rows_dropped))
    print(str(confusion_matrix))
    print(str(f1_score))
    print(str(recon_error))
    out_str = filename + ', ' + filename[len(sequence):temp] + ', '+ options_string
    #write memory
    write_array(memory, out_str, "Memory")
    #write time
    write_array(time, out_str, "Time")
    #write attacks
    if len(attacks) > 0:
        outfile.write(out_str + ", Attacks, " + attacks[0])
    else:
        outfile.write(out_str + ", Attacks, " + "no attacks?")
    #write rows dropped
    write_array(rows_dropped, out_str, "Rows Dropped")
    #write confusion matrix
    for i in confusion_matrix:
        var = str(i).replace("[","").replace("]","")
        outfile.write(out_str + ", Confusion Matrix, " + str(var)+'\n')
    #write f1 score
    write_array(f1_score, out_str, "F1 Score")
    #write recon error
    write_array(recon_error, out_str, "Reconstruction Error")
    outfile.write('\n')
    filein.close()
    print('\n')


directory = '/Users/iandalsin/Nextcloud/School/Sem5 - Dissertation/scikit-backup-nc/Test Results/'
outfile = open('results_parsed.csv', 'w')
for filename in os.listdir(directory):
    print(filename)
    if filename[0].isdigit():
        if filename[4].isdigit():
            sequence = filename[0:5]
        else:
            sequence = filename[0:4]
        parse_csv(filename)


#close everything
outfile.close()
