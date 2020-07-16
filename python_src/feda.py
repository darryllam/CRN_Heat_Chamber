from math import nan
import math
import numpy as np

def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

def expand(instance, truth_index):
    new_attr = []
    for i in range(len(instance)):
        if i == truth_index:
            continue
#         if is_numeric(instance[i]):
#             np.append(instance, nan)
#             new_attr = new_attr + [0,0,0]
#             new_attr = new_attr + [instance[i], instance[i], instance[i]]
#         else:
#             new_attr = new_attr + [instance[i],instance[i],instance[i]]
        new_attr = new_attr + [instance[i], instance[i], instance[i]]

    instance = np.append(instance, new_attr)
    return instance

def source_instance_transform(instance, truth_index):
    before = expand(instance, truth_index)
    after = np.zeros(len(before))
    i = 0
    j = 0
    while i < len(instance):
        i = i + 1
        j = j + 1
        if (i == truth_index):
            after[-1] = before[i]
            j = j - 1
        else:
            after[3*j] = before[i]
            after[3*j+1] = before[i]
#                 after[3*j+2] = nan
            after[3*j+2] = 0
    return after

def source_transform(source_data, truth_index):
    transformed_source_data = []
    source_data = np.vstack(source_data)
    
    for instance in source_data:
        after = source_instance_transform(instance, truth_index)
        transformed_source_data.append(after)
        
    transformed_source_data = np.vstack(transformed_source_data)
    source_ground_truth = transformed_source_data[:, -1]
    transformed_source_data = np.delete(transformed_source_data, -1, axis=1)
    return transformed_source_data, source_ground_truth

def target_instance_transform(instance, truth_index):
    before = expand(instance, truth_index)
    after = np.zeros(len(before))
    i = 0
    j = 0
    while i < len(instance):
        i = i + 1
        j = j + 1
        if i == truth_index:
            after[-1] = before[i]
            j = j - 1
        else:
#                 after[3*j] = nan
            after[3*j] = 0
            after[3*j+1] = before[i]
            after[3*j+2] = before[i]
    return after

def target_transform(target_data, truth_index):
    
    transformed_target_data = []
    target_data = np.vstack(target_data)
    
    for instance in target_data:
        after = target_instance_transform(instance, truth_index)   
        transformed_target_data.append(after)
        
    transformed_target_data = np.vstack(transformed_target_data)
    target_ground_truth = transformed_target_data[:, -1]
    transformed_target_data = np.delete(transformed_target_data, -1, axis=1)
    return transformed_target_data, target_ground_truth


        