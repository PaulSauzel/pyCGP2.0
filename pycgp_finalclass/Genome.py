from abc import ABC, abstractmethod
import random
import copy
from pycgp_finalclass.Node import Node
from pycgp_finalclass.Function import Func
from pycgp_finalclass.Config import CGPConfig
from pycgp_finalclass.Config import Config
import numpy as np

class Genome(ABC): #Abstract class for our genome
    @abstractmethod
    def copy(self):
        pass

    @classmethod
    @abstractmethod
    def create_genome(cls, config):
        pass

class CGPGenome: #This class contains every function that apply directly to the genome
    def __init__(self, config, nodes=None,):
        self.config = config
        self.nodes = nodes if nodes is not None else self._init_nodes() #you can pass in the parameters a specific genome
        #Choose n random outputs from the list of internal nodes
        self.outputs = [
            random.choice(
                range(config.num_inputs, config.num_inputs + config.num_nodes)
            )
            for _ in range(config.num_outputs)
        ]

    def copy(self):
        return CGPGenome(
            config=self.config,  
            nodes=self.nodes
        )        

    @classmethod
    def create_genome(cls, config):
        return cls(config)

    #Initialise a list of nodes of size num_nodes
    def _init_nodes(self): 
        nodes = []
        for i in range(self.config.num_nodes):
            cgp_func = random.choice(self.config.function_set) #Assign a random function from the function set

            const_params = [ #Random uniform values for the constants
                random.uniform(self.config.const_min, self.config.const_max)
                for _ in range(cgp_func.const_params)
            ]

            max_index = self.config.num_inputs + i   # Max valid input index
            inputs = []

            for _ in range(cgp_func.arity):
                if max_index <= self.config.num_inputs:
                    # No internal nodes available yet
                    inputs.append(random.randint(0, self.config.num_inputs - 1))
                else:
                    if random.random() < self.config.node_input_chance:
                        inputs.append(random.randint(0, self.config.num_inputs - 1))  # Input node
                    else:
                        inputs.append(random.randint(self.config.num_inputs, max_index-1))  # Previous node only

            nodes.append(Node(cgp_func, inputs, const_params, max_index)) #fill the list of nodes of the genome
        return nodes


    def get_value(self, input_values):
        values = {i: val for i, val in enumerate(input_values)}  # dictionary of input values

        active_nodes = self.get_active_nodes() #Only compute the active nodes

        for node in active_nodes:
            values[node.index] = node.execute(values) # execute function for each active node to get the end values

        return [values[i] for i in self.outputs] # get values for the outputs nodes


    #used to display the genome in an understandable way
    def to_function_string(self):
        func_str = ""
        for node in self.nodes:
            inputs = [f"x{idx}" if idx < self.config.num_inputs else f"n{idx}" for idx in node.inputs]
            func = node.Func.name
            func_str += f"n{node.index} = {func}({', '.join(inputs)})\n"
        func_str += f"Output: n{self.outputs[0]}\n\n"

        def unroll_node(idx):
            if idx < self.config.num_inputs:
                return f"x{idx}"
            else:
                node = self.nodes[idx - self.config.num_inputs]
                func = node.Func.name
                args = [unroll_node(i) for i in node.inputs]
                return f"{func}({', '.join(args)})"

        output_expr = unroll_node(self.outputs[0])
        func_str += "Unrolled output expression:\n" + output_expr
        return func_str


    #Renvoie une liste de noeufs actifs (noeuds connectés aux outputs)
    def get_active_nodes(self):
        # Créer un dictionnaire index -> Node pour accès rapide
        index_to_node = {node.index: node for node in self.nodes}

        # Commencer avec les indices des sorties
        active_indices = set(self.outputs)
        changed = True

        while changed:
            changed = False
            for node_idx in list(active_indices):
                node = index_to_node.get(node_idx)
                if node is None:
                    # Si on a un index invalide, on ignore (sécurité)
                    continue
                for input_idx in node.inputs:
                    # Si c'est une connexion vers un noeud interne non encore marqué comme actif
                    if input_idx >= self.config.num_inputs and input_idx not in active_indices:
                        if input_idx in index_to_node:
                            active_indices.add(input_idx)
                            changed = True
                        else:
                            # Optionnel : alerte de debug si une entrée ne correspond à aucun Node
                            print(f"[WARNING] input_idx {input_idx} not found in nodes.")

        # Retourner les noeud actifs dans l'ordre croissant de leur index
        active_nodes = [index_to_node[i] for i in active_indices if i in index_to_node]
        active_nodes.sort(key=lambda node: node.index)

        return active_nodes





