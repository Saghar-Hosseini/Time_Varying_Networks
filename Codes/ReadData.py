__author__ = 'saghar'
import os
class ReadData(object):
    def __init__(self, path):
        self.path=path
        list_files=[]
        for filename in os.listdir(path):
            list_files.append(filename)
        self.files=list_files
    def read_network_snapshot(self,time):
        Adj={}
        edge=[]
        file=self.files[time]
        data=open(self.path+file, "r")
        for line in data:
           if line[0]!="#":
              edge=line.split()
              if not edge[0] in Adj.keys():
                 Adj[edge[0]]=[edge[1]]
              else:
                 Adj[edge[0]].append(edge[1])
        return Adj

