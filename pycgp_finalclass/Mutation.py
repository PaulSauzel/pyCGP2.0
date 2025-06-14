from abc import ABC, abstractmethod
import random
class Mutation(ABC): #Abstract class for mutation
    @abstractmethod
    def mutate(self, genome):
        pass

class Golden_mutation(Mutation): 

    def __init__(self,config,input_node_mutation_rate, function_mutation_rate,input_mutation_rate,const_mutation_rate,output_node_mutation_rate):
        self.config = config
        self.input_node_mutation_rate = input_node_mutation_rate
        self.function_mutation_rate = function_mutation_rate
        self.input_mutation_rate = input_mutation_rate
        self.const_mutation_rate = const_mutation_rate
        self.output_node_mutation_rate = output_node_mutation_rate


    def mutate(self, genome):
        # Chance of mutating output
        if random.random() < self.output_node_mutation_rate:
            self.mutate_outputs(genome)

        active_node_indices = {node.index for node in genome.get_active_nodes()}

        while True:
            node = random.choice(genome.nodes)
            node_index = node.index

            # Apply one mutation
            r = random.random()
            if r < self.function_mutation_rate:
                node.mutate_function(self.config.function_set, node_index)
            elif r < self.function_mutation_rate + self.input_mutation_rate:
                node.mutate_inputs(self.config.num_inputs, node_index, self.input_node_mutation_rate)
            else:
                node.mutate_constants(self.config.const_min, self.config.const_max)

            # If the mutated node is active, we stop
            if node_index in active_node_indices:
                break


    #Mutate the outputs of the genome(used in mutate)
    def mutate_outputs(self, genome):
        # Pick a random output index to mutate
        output_index = random.randint(0, len(genome.outputs) - 1)
        current_output = genome.outputs[output_index]
        
        possible_outputs = list(set(range(self.config.num_inputs, self.config.num_inputs + self.config.num_nodes)) - {current_output})
        if possible_outputs:
            new_output = random.choice(possible_outputs)
            genome.outputs[output_index] = new_output





#IN DEVELOPMENT
class Proba_Mutation(Mutation):
    def __init__(self,config, number_mutations,input_node_mutation_rate, output_node_mutation_rate,function_mutation_rate,input_mutation_rate,const_mutation_rate):
        self.config = config
        self.number_mutations = number_mutations
        self.input_node_mutation_rate = input_node_mutation_rate
        self.output_node_mutation_rate = output_node_mutation_rate
        self.function_mutation_rate = function_mutation_rate
        self.input_mutation_rate = input_mutation_rate
        self.const_mutation_rate = const_mutation_rate

    def mutate(self, genome):
        if random.random() < self.input_node_mutation_rate:
            self.mutate_nodes(genome)
        if random.random() < self.output_node_mutation_rate:
            self.mutate_outputs(genome)
        

    def mutate_nodes(self, genome):
        assert abs((self.function_mutation_rate + self.input_mutation_rate + self.const_mutation_rate) - 1.0) < 1e-6, "Les probabilités doivent faire 1."
        for _ in range(self.number_mutations): #mutate sur un nombre de mutation définis
            node = random.choice(genome.nodes)  # sélection aléatoire du noeud
            node_index = node.index  # get the index of the node         
            p = random.random()
            #a l'interieur de cette mutation soit mutation sur fonction soit mutation sur inputs soit mutation sur constantes
            if p < self.function_mutation_rate: 
                node.mutate_function(self.config.function_set, node_index)
            elif p < self.function_mutation_rate + self.input_mutation_rate:
                node.mutate_inputs(self.config.num_inputs, node_index, self.input_node_mutation_rate)
            else:
                node.mutate_constants(self.config.const_min, self.config.const_max)
 
    def mutate_outputs(self,genome):
        # Randomly select a new output node from the internal nodes
        current_output = genome.outputs[0]
        possible_outputs = list(set(range(self.config.num_inputs, self.config.num_inputs + self.config.num_nodes)) - {current_output})
        if possible_outputs:
            new_output = random.choice(possible_outputs)
            genome.outputs = [new_output]
            print(f"Output node mutated from: n{current_output} to: n{new_output}")
        else:
            print("Only one possible output; output mutation skipped.")
