__author__ = 'sagha_000'
import csv
def write_to_cvs(output_file,state,numCommunity):
    test_file = open(output_file,'wb')
    fld=['node']
    fld.extend(range(numCommunity))
    csvwriter = csv.DictWriter(test_file, delimiter=',', fieldnames=fld)
    csvwriter.writerow(dict((fn,fn) for fn in fld))
    row={}
    for node in state.keys():
            row['node']=node
            for i in range(numCommunity):
                row[i]=state[node][i]
            csvwriter.writerow(row)
    test_file.close()
