import random
import numpy as np

class Func:
    def __init__(self, function, name, arity, const_params): #Format to pass our function library
            self.function = function
            self.name = name
            self.arity = arity
            self.const_params = const_params

