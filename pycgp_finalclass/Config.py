from abc import ABC, abstractmethod

class Config(ABC):
    @abstractmethod
    def validate(self):
        pass

class CGPConfig(Config): #this is the class where we tune our main parameters 

    def __init__(self,num_inputs, num_nodes, num_outputs, input_node_chance, 
                  const_min,const_max,function_set) :
        self.num_inputs = num_inputs # Number of inputs nodes
        self.num_nodes = num_nodes # size of our graph
        self.num_outputs = num_outputs #most of the time equal to 1
        self.const_min = const_min #The range of our constants
        self.const_max = const_max
        self.function_set = function_set #our list of function that we define in the main
        self.node_input_chance = input_node_chance #The chance we pick from input node or other internal node when initialising
    
    def validate(self):
        pass