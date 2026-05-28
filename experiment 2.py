class BaseLayer:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method.")

class BasicLayer(BaseLayer):
    def forward(self, x):
        return [w * x for w in self.weights]

class PrunedLayer(BaseLayer):
    def __init__(self, name, weights, threshold=0.5):
        super().__init__(name, weights) 
        self.threshold = threshold

    def forward(self, x):
        return [(w if abs(w) > self.threshold else 0.0) * x for w in self.weights]

class QuantizedLayer(BaseLayer):
    def forward(self, x):
        return [round(w) * x for w in self.weights]

layers = [
    BasicLayer("basic forward", [0.1, -0.8, 0.4, -0.2]),
    PrunedLayer("pruning forward", [0.1, -0.8, 0.4, -0.2], threshold=0.3),
    QuantizedLayer("quantization forward", [0.1, -0.8, 0.4, -0.2])
]

for layer in layers:
    print(f"{layer.name} : {layer.forward(2.0)}")