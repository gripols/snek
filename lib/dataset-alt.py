import numpy as np
import specutils
from tqdm import tqdm

def mixup_generator(inputs, targets, alpha):
    permutation = np.random.permutation(len(inputs))[:len(inputs) // 2]

    if len(permutation) % 2 != 0:
        permutation = permutation[:-1]

    for i in range(0, len(permutation), 2):
        lam = np.random.beta(alpha, alpha)
        index1, index2 = permutation[i], permutation[i + 1]
        inputs[index1] = lam * inputs[index1] + (1 - lam) * inputs[index2]
        targets[index1] = lam * targets[index1] + (1 - lam) * targets[index2]

    return inputs, targets

def create_dataset(file_list, crop_size, num_patches=32, is_validation=False):
    dataset_length = num_patches * len(file_list)
    input_dataset = np.zeros((dataset_length, 2, crop_size, crop_size), dtype=np.float32)
    target_dataset = np.zeros((dataset_length, 2, crop_size, crop_size), dtype=np.float32)

    for index, (input_path, target_path) in enumerate(tqdm(file_list)):
        input_data, target_data = spec_utils.cache_or_load(input_path, target_path)
        
        for patch_index in range(num_patches):
            dataset_index = index * num_patches + patch_index
            start_position = np.random.randint(0, input_data.shape[2] - crop_size)
            input_dataset[dataset_index] = input_data[:, :, start_position:start_position + crop_size]
            target_dataset[dataset_index] = target_data[:, :, start_position:start_position + crop_size]

            if not is_validation:
                if np.random.uniform() < 0.5:
                    input_dataset[dataset_index] = input_dataset[dataset_index, :, :, ::-1]
                    target_dataset[dataset_index] = target_dataset[dataset_index, :, :, ::-1]
                if np.random.uniform() < 0.5:
                    input_dataset[dataset_index] = input_dataset[dataset_index, ::-1]
                    target_dataset[dataset_index] = target_dataset[dataset_index, ::-1]

    return input_dataset, target_dataset