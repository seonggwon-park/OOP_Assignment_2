
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

def dot(row, x):
    return sum(w * v for w, v in zip(row, x))

def relu(values):
    return [max(0.0, v) for v in values]

def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


class LayerBlock:
    def __init__(self, name, weights, bias, activation="relu",
                 bit_width=32, sparse=False):
        self.name = name
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.bit_width = bit_width
        self.sparse = sparse

    def clone(self):
        return deepcopy(self)

    def forward(self, x):
        y = [dot(row, x) + b for row, b in zip(self.weights, self.bias)]
        if self.activation == "relu":
            return relu(y)
        if self.activation == "identity":
            return y
        raise ValueError(f"Unknown activation: {self.activation}")

    def weight_count(self):
        return sum(len(row) for row in self.weights)

    def bias_count(self):
        return len(self.bias)

    def nonzero_weight_count(self):
        return sum(1 for row in self.weights for w in row if w != 0.0)

    def memory_bytes(self):
        # Simplified memory estimator for this OOP project.
        bytes_per_value = self.bit_width / 8.0

        if self.sparse:
            # Sparse format stores only nonzero weights, plus a small index cost.
            index_overhead_bytes = 1.0
            weight_memory = self.nonzero_weight_count() * (
                bytes_per_value + index_overhead_bytes
            )
        else:
            weight_memory = self.weight_count() * bytes_per_value

        bias_memory = self.bias_count() * bytes_per_value
        return weight_memory + bias_memory

    def summary(self):
        storage = "sparse" if self.sparse else "dense"
        return (
            f"{self.name}: activation={self.activation}, "
            f"bit_width={self.bit_width}, storage={storage}, "
            f"memory={self.memory_bytes():.2f} bytes"
        )


class BinaryClassifier:
    def __init__(self, layers):
        self.layers = layers

    def clone(self):
        return deepcopy(self)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def predict(self, x):
        return argmax(self.forward(x))

    def memory_bytes(self):
        return sum(layer.memory_bytes() for layer in self.layers)

    def apply_action_to_layer(self, layer_index, action):
        new_model = self.clone()
        target_layer = new_model.layers[layer_index]
        new_model.layers[layer_index] = action.apply(target_layer)
        return new_model

    def architecture_summary(self):
        lines = [layer.summary() for layer in self.layers]
        lines.append(f"Total memory: {self.memory_bytes():.2f} bytes")
        return "\n".join(lines)


class CompressionAction(ABC):
    @abstractmethod
    def apply(self, layer):
        pass

    @abstractmethod
    def name(self):
        pass


class NoCompressionAction(CompressionAction):
    def apply(self, layer):
        return layer.clone()

    def name(self):
        return "No Compression"


class PruningAction(CompressionAction):
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, layer):
        new_layer = layer.clone()
        new_layer.weights = [
            [0.0 if abs(w) <= self.threshold else w for w in row]
            for row in layer.weights
        ]
        new_layer.sparse = True
        return new_layer

    def name(self):
        return f"Pruning(threshold={self.threshold})"


class QuantizationAction(CompressionAction):
    def __init__(self, bit_width):
        if bit_width < 2:
            raise ValueError("bit_width must be at least 2.")
        self.bit_width = bit_width

    def _quantize_value(self, value):
        # Toy symmetric quantization. This simulates lower precision.
        scale = (2 ** (self.bit_width - 1)) - 1
        return round(value * scale) / scale

    def apply(self, layer):
        new_layer = layer.clone()
        new_layer.weights = [
            [self._quantize_value(w) for w in row]
            for row in layer.weights
        ]
        new_layer.bias = [self._quantize_value(b) for b in layer.bias]
        new_layer.bit_width = self.bit_width
        return new_layer

    def name(self):
        return f"{self.bit_width}-bit Quantization"


class CompressionConstraint(ABC):
    @abstractmethod
    def is_satisfied(self, original_model, candidate_model, dataset):
        pass

    @abstractmethod
    def name(self):
        pass


class PredictionPreservationConstraint(CompressionConstraint):
    def is_satisfied(self, original_model, candidate_model, dataset):
        for x in dataset:
            if original_model.predict(x) != candidate_model.predict(x):
                return False
        return True

    def name(self):
        return "Prediction Preservation"


class MemoryBudgetConstraint(CompressionConstraint):
    def __init__(self, max_memory_bytes):
        self.max_memory_bytes = max_memory_bytes

    def is_satisfied(self, original_model, candidate_model, dataset):
        return candidate_model.memory_bytes() <= self.max_memory_bytes

    def name(self):
        return f"Memory Budget <= {self.max_memory_bytes:.2f} bytes"


@dataclass
class CandidateResult:
    layer_name: str
    action_name: str
    memory_bytes: float
    reduction_percent: float
    prediction_preserved: bool
    accepted: bool


@dataclass
class PlannerResult:
    original_memory: float
    final_memory: float
    memory_reduction_percent: float
    selected_actions: list
    candidate_history: list
    final_predictions_preserved: bool
    memory_budget_satisfied: bool


class GreedyCompressionPlanner:
    def __init__(self, actions, safety_constraints, memory_budget):
        self.actions = actions
        self.safety_constraints = safety_constraints
        self.memory_budget = memory_budget

    def _satisfies_safety_constraints(self, original_model, candidate_model, dataset):
        return all(
            constraint.is_satisfied(original_model, candidate_model, dataset)
            for constraint in self.safety_constraints
        )

    def search(self, original_model, dataset):
        current_model = original_model.clone()
        original_memory = original_model.memory_bytes()
        selected_actions = []
        candidate_history = []

        for layer_index, layer in enumerate(current_model.layers):
            best_model = current_model
            best_action_name = "No Compression"
            best_memory = current_model.memory_bytes()

            for action in self.actions:
                candidate_model = current_model.apply_action_to_layer(
                    layer_index, action
                )
                candidate_memory = candidate_model.memory_bytes()
                preserved = self._satisfies_safety_constraints(
                    original_model, candidate_model, dataset
                )
                accepted = preserved

                reduction = 100.0 * (
                    original_memory - candidate_memory
                ) / original_memory

                candidate_history.append(
                    CandidateResult(
                        layer_name=layer.name,
                        action_name=action.name(),
                        memory_bytes=candidate_memory,
                        reduction_percent=reduction,
                        prediction_preserved=preserved,
                        accepted=accepted,
                    )
                )

                if accepted and candidate_memory < best_memory:
                    best_model = candidate_model
                    best_action_name = action.name()
                    best_memory = candidate_memory

            current_model = best_model
            selected_actions.append((layer.name, best_action_name))

        final_memory = current_model.memory_bytes()
        final_reduction = 100.0 * (original_memory - final_memory) / original_memory

        final_preserved = self._satisfies_safety_constraints(
            original_model, current_model, dataset
        )
        budget_satisfied = self.memory_budget.is_satisfied(
            original_model, current_model, dataset
        )

        return PlannerResult(
            original_memory=original_memory,
            final_memory=final_memory,
            memory_reduction_percent=final_reduction,
            selected_actions=selected_actions,
            candidate_history=candidate_history,
            final_predictions_preserved=final_preserved,
            memory_budget_satisfied=budget_satisfied,
        )


def build_toy_binary_classifier():
    hidden = LayerBlock(
        name="HiddenLayer",
        weights=[
            [0.8, -0.4, 0.3],
            [-0.2, 0.9, 0.5],
            [0.6, 0.1, -0.7],
            [0.05, -0.3, 0.4],
        ],
        bias=[0.1, -0.1, 0.05, 0.0],
        activation="relu",
    )

    output = LayerBlock(
        name="OutputLayer",
        weights=[
            [0.7, -0.6, 0.4, -0.2],
            [-0.5, 0.8, -0.3, 0.5],
        ],
        bias=[0.05, -0.02],
        activation="identity",
    )

    return BinaryClassifier([hidden, output])


def print_original_predictions(model, dataset):
    print("Original predictions")
    for i, x in enumerate(dataset):
        logits = model.forward(x)
        pred = model.predict(x)
        print(f"  sample {i}: logits={logits}, prediction=class {pred}")


def print_candidate_table(result):
    print("\nCandidate evaluation history")
    print(
        f"{'Layer':<14} {'Action':<26} {'Memory':>10} "
        f"{'Reduction':>12} {'Preserved':>10} {'Accepted':>10}"
    )
    print("-" * 88)

    for row in result.candidate_history:
        print(
            f"{row.layer_name:<14} {row.action_name:<26} "
            f"{row.memory_bytes:>9.2f}B "
            f"{row.reduction_percent:>10.2f}% "
            f"{str(row.prediction_preserved):>10} "
            f"{str(row.accepted):>10}"
        )


def main():
    dataset = [
        [1.0, 0.5, -1.0],
        [-0.5, 1.0, 0.3],
        [0.2, -0.1, 1.2],
        [1.5, -0.7, 0.5],
    ]

    model = build_toy_binary_classifier()

    actions = [
        NoCompressionAction(),
        PruningAction(threshold=0.2),
        PruningAction(threshold=0.4),
        QuantizationAction(bit_width=8),
        QuantizationAction(bit_width=4),
        QuantizationAction(bit_width=2),
    ]

    planner = GreedyCompressionPlanner(
        actions=actions,
        safety_constraints=[PredictionPreservationConstraint()],
        memory_budget=MemoryBudgetConstraint(max_memory_bytes=20.0),
    )

    print("Original model architecture")
    print(model.architecture_summary())
    print()
    print_original_predictions(model, dataset)

    result = planner.search(model, dataset)
    print_candidate_table(result)

    print("\nSelected compression plan")
    for layer_name, action_name in result.selected_actions:
        print(f"  {layer_name}: {action_name}")

    print("\nFinal result")
    print(f"  Original memory: {result.original_memory:.2f} bytes")
    print(f"  Final memory: {result.final_memory:.2f} bytes")
    print(f"  Memory reduction: {result.memory_reduction_percent:.2f}%")
    print(f"  Prediction preserved: {result.final_predictions_preserved}")
    print(f"  Memory budget satisfied: {result.memory_budget_satisfied}")


if __name__ == "__main__":
    main()
