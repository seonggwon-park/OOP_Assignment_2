def forward(x, weights, compression, threshold=0.0):

    if compression == "basic":
        return [w * x for w in weights]
    
    elif compression == "pruning":
        return [(w if abs(w) > threshold else 0.0) * x for w in weights]
    
    elif compression == "quantization":
        return [round(w) * x for w in weights]
    
    else:
        raise ValueError("Unknown layer type")

weights = [0.1, -0.8, 0.4, -0.2]

print(f"basic forward : {forward(2.0, weights, "basic", threshold=0.3)}")
print(f"pruning forward : {forward(2.0, weights, "pruning", threshold=0.3)}")
print(f"quantization forward : {forward(2.0, weights, "quantization", threshold=0.3)}")
