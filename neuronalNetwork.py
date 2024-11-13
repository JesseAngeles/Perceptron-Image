from perceptron import Perceptron

class NeuronalNetwork:
    def __init__(self, first_class: list[list[float]], second_class: list[list[float]], perceptron: Perceptron):
        self.first_class = first_class
        self.second_class = second_class
        self.perceptron: Perceptron = perceptron
        
    def train(self):
        is_fit: bool = False
        count: int = 0
        
        while not is_fit:
            is_fit = True
            for item in self.first_class:
                is_fit = is_fit and self.perceptron.adjustmnentFunction(item, 0)
                
            for item in self.second_class:
                is_fit = is_fit and self.perceptron.adjustmnentFunction(item, 1)

            count += 1  # Opcional: contar las iteraciones para seguimiento o depuración