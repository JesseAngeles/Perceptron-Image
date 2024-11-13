class Perceptron:
    def __init__(self, weights: list[float], training_rate: float):
        # Añadir peso de sesgo en la inicialización
        self.weights = weights  # Peso adicional para el sesgo
        self.training_rate = training_rate
        
    def adjust(self, input_layer: list[float], cls: bool):
        num: float = -self.training_rate if not cls else self.training_rate
        for i in range(len(self.weights)):
            self.weights[i] += num * input_layer[i]
        
    def exitFunction(self, input_layer: list[float]) -> float:
        exit_value: float = 0
        for i, input in enumerate(input_layer):
            exit_value += self.weights[i] * input
        return exit_value
        
    def adjustmnentFunction(self, input_layer: list[float], cls: bool) -> bool:
        input_layer = input_layer + [1]  # Añadir el término de sesgo a la entrada
        exit_function: float = self.exitFunction(input_layer)
        
        # Verificar si se clasifica mal
        if (not cls and exit_function >= 0) or (cls and exit_function <= 0):
            self.adjust(input_layer, cls)
            return False
        return True
