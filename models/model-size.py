import torch

state_dict = torch.load('baseline.pth', map_location=torch.device('cpu'))

layer_types = {}

for key in state_dict.keys():
    layer_name = key.split('.')[0]
    
    if layer_name not in layer_types:
        layer_types[layer_name] = 1
    else:
        layer_types[layer_name] += 1

print("Detected Layer Types and Counts:")
for layer, count in layer_types.items():
    print(f"{layer}: {count}")

layer_sizes = {}

for key, value in state_dict.items():
    layer_name = key.split('.')[0]
    layer_size = value.shape
    if layer_name not in layer_sizes:
        layer_sizes[layer_name] = [layer_size]
    else:
        layer_sizes[layer_name].append(layer_size)

print("\nDetected Layer Sizes (Shape of Parameters):")
for layer, sizes in layer_sizes.items():
    print(f"{layer}: {sizes}")
