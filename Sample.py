__author__ = 'sagha_000'
import numpy as np
import Config
class Sample:
    def __init__(self,edge,non_edge,nodes):
        self.links=edge
        self.non_links=non_edge
        self.nodes=nodes
        self.pdf_links={}
        self.pdf_non_links={}
######################################################################
    def NodeSampling(self,numSamples):
        self.nodes=np.random.choice(Config.edges.keys(),size=numSamples,replace=False)
        self.links={}
        self.non_links={}
        self.pdf_links={}
        self.pdf_non_links={}
        n=Config.n
        for i in self.nodes:
            self.links[i]=[i]
            self.non_links[i]=[]
            self.pdf_links[i]=[1.0/n]
            self.pdf_non_links[i]=[]
            for j in self.nodes:
                if j in Config.edges[i]:
                    self.links[i].append(j)
                    self.pdf_links[i].append(2.0/n)
                elif j!=i and not j in Config.edges[i]:
                    self.non_links[i].append(j)
                    self.pdf_non_links[i].append(2.0/n)

######################################################################
    def PairSampling(self,numSamples):
        n=Config.n
        self.pdf_links={}
        self.pdf_non_links={}
        self.links={}
        self.non_links={}
        while len(self.nodes)< numSamples:
            i=np.random.choice(Config.edges.keys())
            j=np.random.choice(Config.edges.keys())
            if not i in self.nodes:
                self.nodes.append(i)
                self.links[i]=[i]
                self.non_links[i]=[]
                self.pdf_links[i]=[1.0/n]
                self.pdf_non_links[i]=[]
            if not j in self.nodes:
                self.nodes.append(j)
                self.links[j]=[j]
                self.non_links[j]=[]
                self.pdf_links[j]=[1.0/n]
                self.pdf_non_links[i]=[]
            if j in Config.edges[i]:
                self.links[i].append(j)
                self.links[j].append(i)
                self.pdf_links[i].append(2.0/n/(n-1))
                self.pdf_links[j].append(2.0/n/(n-1))
            else:
                self.non_links[i].append(j)
                self.non_links[j].append(i)
                self.pdf_non_links[i].append(2.0/n/(n-1))
                self.pdf_non_links[j].append(2.0/n/(n-1))
######################################################################
    def InducedEdgeSampling(self,numSamples):
        n=float(Config.n)
        #e=len()
        self.pdf_links={}
        self.pdf_non_links={}
        deg={}
        while len(self.nodes)< numSamples:
            i=np.random.choice(Config.edges.keys())
            j=np.random.choice(Config.edges[i])
            if not i in self.nodes:
                self.nodes.append(i)
            if not j in self.nodes:
                self.nodes.append(j)

        for i in self.nodes:
            deg[i]=0
            for j in Config.edges[i]:
                if j in self.nodes:
                    deg[i]+=1
        self.links={}
        self.non_links={}
        for i in self.nodes:
            self.links[i]=[i]
            self.non_links[i]=[]
            self.pdf_links[i]=[(len(Config.edges[i])+1)/n]
            self.pdf_non_links[i]=[]
            for j in self.nodes:
                if j in Config.edges[i]:
                    self.links[i].append(j)
                    self.pdf_links[i].append((len(Config.edges[i])+1)/n*1.0/deg[i]+(len(Config.edges[j])+1)/n*1.0/deg[j])
                else:
                    self.non_links[i].append(j)
                    self.pdf_non_links[i].append((len(Config.edges[i])+1)/n*1.0/(numSamples-deg[i]-1)+(len(Config.edges[j])+1)/n*1.0/(numSamples-deg[j]-1))




