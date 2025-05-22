from pycgp_finalclass.Genome import CGPGenome
from pycgp_finalclass.ES import ES
from pycgp_finalclass.Config import Config
from pycgp_finalclass.Mutation import Mutation

import networkx as nx
import matplotlib.pyplot as plt

class CGP: #Class in development
    def __init__(self, genome):
        self.genome = genome
        self.active_nodes = genome._get_active_nodes()
        self.graph = self._build_graph()

    def _build_graph(self):
        pass

    def visualize(self):
        pass
    #method to visualize the graph
    #Puts more comments , say when some features are not implemented
    # put on git
    # method for seing active genome
    #fix const params function

        # Thursday 11:30 22 may