__author__ = 'sagha_000'
import numpy as np
class Sample:
    def __init__(self,edge,none_edge,nodes):
        self.links=edge
        self.none_links=none_edge
        self.nodes=nodes
        self.pdf_links={}
        self.pdf_non_links={}
######################################################################
    def NodeSampling(self,numSamples,Edges,Nodes):
        self.nodes=np.random.choice(list(Nodes),size=numSamples,replace=False)
        self.links={}
        self.none_links={}
        self.pdf_links={}
        self.pdf_non_links={}
        n=len(Nodes)
        for i in self.nodes:
            self.links[i]=[i]
            self.none_links[i]=[]
            self.pdf_links[i]=[1.0/n]
            self.pdf_non_links[i]=[]
            for j in self.nodes:
                if j in Edges[i]:
                    self.links[i].append(j)
                    self.pdf_links[i].append(2.0/n)
                elif j!=i and not j in Edges[i]:
                    self.none_links[i].append(j)
                    self.pdf_non_links[i].append(2.0/n)

######################################################################
    def PairSampling(self,numSamples,Nodes,Edges):
        n=len(Nodes)
        self.pdf_links={}
        self.pdf_non_links={}
        self.links={}
        self.none_links={}
        while len(self.nodes)< numSamples:
            i=np.random.choice(Edges.keys())
            j=np.random.choice(Edges.keys())
            if not i in self.nodes:
                self.nodes.append(i)
                self.links[i]=[i]
                self.none_links[i]=[]
                self.pdf_links[i]=[1.0/n]
                self.pdf_non_links[i]=[]
            if not j in self.nodes:
                self.nodes.append(j)
                self.links[j]=[j]
                self.none_links[j]=[]
                self.pdf_links[j]=[1.0/n]
                self.pdf_non_links[i]=[]
            if j in Edges[i]:
                self.links[i].append(j)
                self.links[j].append(i)
                self.pdf_links[i].append(2.0/n/(n-1))
                self.pdf_links[j].append(2.0/n/(n-1))
            else:
                self.none_links[i].append(j)
                self.none_links[j].append(i)
                self.pdf_non_links[i].append(2.0/n/(n-1))
                self.pdf_non_links[j].append(2.0/n/(n-1))
######################################################################
    def InducedEdgeSampling(self,numSamples,Nodes,Edges):
        n=len(Nodes)
        self.pdf_links={}
        self.pdf_non_links={}
        deg={}
        while len(self.nodes)< numSamples:
            i=np.random.choice(Edges.keys())
            j=np.random.choice(Edges[i])
            if not i in self.nodes:
                self.nodes.append(i)
                deg[i]=len(Edges[i])
            if not j in self.nodes:
                self.nodes.append(j)
                deg[j]=len(Edges[j])
        self.links={}
        self.none_links={}
        for i in self.nodes:
            self.links[i]=[i]
            self.none_links[i]=[]
            self.pdf_links[i]=[1.0/n]
            self.pdf_non_links[i]=[]
            for j in self.nodes:
                if j in Edges[i]:
                    self.links[i].append(j)
                    self.pdf_links[i].append(1.0/n*1.0/deg[i]+1.0/n*1.0/deg[j])
                else:
                    self.none_links[i].append(j)
                    self.pdf_non_links[i].append(1.0/n*1.0/(n-deg[i])+1.0/n*1.0/(n-deg[j]))




