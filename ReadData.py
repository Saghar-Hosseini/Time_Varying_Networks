__author__ = 'saghar'
import os
class ReadData(object):
    def __init__(self, path):
        self.path=path
        list_files=[]
        for filename in os.listdir(path):
            list_files.append(filename)
        self.files=list_files
    def read_network_snapshot(self,time,hasHeader):
        Adj={}
        edge=[]
        # file=self.files[time]
        file='graph'+str(time)+'.csv'
        data=open(self.path+file, "r")
        for i, line in enumerate(data):
            if i < 1 and hasHeader:
                continue
            # if line[0]!="#":
            line = line.strip() # remove \n from end of the line
            edge = line.split(',')
            if not edge[1] in Adj.keys():
                Adj[edge[1]] = [edge[0]]
            else:
                Adj[edge[1]].append(edge[0])
            if not edge[0] in Adj.keys():
                Adj[edge[0]] = [edge[1]]
            else:
                Adj[edge[0]].append(edge[1])
        return Adj
