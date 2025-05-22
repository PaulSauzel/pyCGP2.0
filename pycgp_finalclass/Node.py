import random
import numpy as np
class Node:
    def __init__(self, Func, inputs,const_params,index):
        self.Func = Func # Pointer to a function
        self.inputs = inputs  # List of input indices
        self.const_params = const_params  # List of constant parameters for the function
        self.index = index #make it easier to point to a specific node
        
    def execute(self, values_dict): 
        try:
            input_values = [values_dict[i] for i in self.inputs] #Get the value that match the inputs indexes of the node
        except KeyError as e:
            print(f"Missing value for input index {e}")
            raise

        result = self.Func.function(input_values, self.const_params) #Apply the function to get the result of the node
        return result
    

    def copy(self):
        return Node(
            func=self.Func,  
            inputs=self.inputs[:],  # make a copy to avoid shared references
            const_params=self.const_params[:],  # copy list to prevent shared mutation
            index=self.index[:]
        )
    
    #Select a new function from the list and change the arity depending on it
    def mutate_function(self,function_set,node_index):    
        old_func = self.Func
        new_func = random.choice(function_set)
        self.Func = new_func
        if new_func.arity > old_func.arity: #mutation sur fonction doit garder les memes input
            self.inputs.extend([random.randint(0, node_index-1) for _ in range(new_func.arity - old_func.arity)])
        elif new_func.arity < old_func.arity:
            self.inputs = self.inputs[:new_func.arity]

    #Mutate the input of the node to either an input node or an internal node
    def mutate_inputs(self, num_inputs, node_index, input_node_mutation_rate): 
        max_index = node_index - 1  # Only use past inputs or earlier nodes
        for i in range(len(self.inputs)):
            if random.random() < input_node_mutation_rate:
                if max_index < num_inputs:  # no previous nodes
                    self.inputs[i] = random.randint(0, num_inputs - 1)
                else:
                    self.inputs[i] = random.randint(num_inputs, max_index)


    def mutate_constants(self,const_min,const_max): #choose new random constant in the range
        for i in range(len(self.const_params)):
            self.const_params[i] = random.uniform(const_min, const_max)

    
