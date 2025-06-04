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
        return copy.deepcopy(self)      

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









    def visualize_active_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        active_nodes = self.get_active_nodes()
        G = nx.DiGraph()

        pos = {}
        labels = {}

        # Estimate how many rows we'll need to display
        total_nodes = max(len(active_nodes), self.config.num_inputs, len(self.outputs))
        max_nodes_per_column = total_nodes

        # Dynamic vertical spacing: more nodes → tighter spacing
        node_spacing = max(1.0, 15.0 / max_nodes_per_column)
        layer_spacing = 2.0

        # Dynamic node size and font size based on node count
        base_node_size = 1800
        base_font_size = 10
        scale = min(1.0, 30 / max_nodes_per_column)
        node_size = base_node_size * scale
        font_size = max(6, int(base_font_size * scale))

        # Input nodes (left)
        for i in range(self.config.num_inputs):
            name = f"x{i}"
            G.add_node(name, color='lightblue')
            pos[name] = (0, -i * node_spacing)
            labels[name] = name

        # Active internal nodes (middle)
        for j, node in enumerate(active_nodes):
            node_label = f"n{node.index}\n{node.Func.name}"
            G.add_node(node_label, color='lightgreen')
            pos[node_label] = (layer_spacing, -j * node_spacing)
            labels[node_label] = node_label

            for input_idx in node.inputs:
                if input_idx < self.config.num_inputs:
                    input_label = f"x{input_idx}"
                else:
                    input_label = f"n{input_idx}\n{self.nodes[input_idx - self.config.num_inputs].Func.name}"
                G.add_edge(input_label, node_label)

        # Output nodes (right)
        for k, output_idx in enumerate(self.outputs):
            output_node = self.nodes[output_idx - self.config.num_inputs]
            output_label = f"n{output_idx}\n{output_node.Func.name}"
            if output_label not in G.nodes:
                G.add_node(output_label, color='orange')
                pos[output_label] = (layer_spacing * 2, -k * node_spacing)
                labels[output_label] = output_label
            else:
                G.nodes[output_label]['color'] = 'orange'
                pos[output_label] = (layer_spacing * 2, -k * node_spacing)

        # Draw the graph
        node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes]
        nx.draw(G, pos, with_labels=True, labels=labels,
                node_color=node_colors, node_size=node_size,
                font_size=font_size, arrows=True, edge_color='gray')

        plt.title("Active Genome Graph (Inputs → Internal → Outputs)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

