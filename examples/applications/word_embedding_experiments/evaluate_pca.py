import numpy as np
from scipy.optimize import minimize
import scipy
import json

def distance(subspace, linear_combination_factors, point):
    assert(len(linear_combination_factors) == len(subspace))
    linear_combination_factors = np.array(linear_combination_factors)
    subspace = np.array(subspace)

    # next two lines can be written as a matrix multiplication...
    linear_combination = np.array([factor*basis_vector for factor, basis_vector in zip(linear_combination_factors, subspace)])
    linear_combination_vector = np.sum(linear_combination, axis=0)

    point = np.array(point)
    return np.linalg.norm(linear_combination_vector - point)

# print(distance([[0,1]],[100],[1,0]))

def find_nearest_point_in_subspace(subspace, point):
    function = lambda linear_factors : distance(subspace, linear_factors, point)
    initial_guess = [0 for k in range(len(subspace))]
    minimiser = minimize(function, initial_guess)

    # next two lines can be written as a matrix multiplication...
    subspace = np.array(subspace)
    linear_combination = np.array([factor * basis_vector for factor, basis_vector in zip(minimiser.x, subspace)])
    linear_combination_vector = np.sum(linear_combination, axis=0)

    return linear_combination_vector

# print(find_nearest_point_in_subspace([[0,1]],[1,1]))

def cosine_dist(x,y):
    return 1-scipy.spatial.distance.cosine(x,y)

def vector_subspace_angle(subspace, vector):
    return cosine_dist(find_nearest_point_in_subspace(subspace,vector), vector)

def vector_subspace_distance(subspace, vector):
    return np.linalg.norm(find_nearest_point_in_subspace(subspace,vector)-vector)

def load_principal_components(name=None):
    '''name=None gives all principal components, otherwise only the components for the dataset specified by name.'''
    with open('principal_components.json') as json_file:
        all_component_data = json.load(json_file)
    if name==None:
        return all_component_data
    else:
        return all_component_data[name]

# print("vector_subspace_angle is invariant under scaling: ")
# print(vector_subspace_angle([[0,1]],[1,1]),vector_subspace_angle([[0,1]],[5,5]))
# print("vector_subspace_distance is linear under scaling: ")
# print(vector_subspace_distance([[0,1]],[1,1]),vector_subspace_distance([[0,1]],[5,5]))

paws_data = load_principal_components('google_paws')
active_passive_data = load_principal_components('active_passive_dataset')


print(paws_data)
print(type(paws_data))
print(paws_data.keys())

import matplotlib.pyplot as plt
plt.plot(paws_data['explained_variance_ratio_'])
plt.plot(active_passive_data['explained_variance_ratio_'])
plt.show()

print(vector_subspace_angle(paws_data['components_'][0:100],active_passive_data['components_'][0]))
print(vector_subspace_angle(paws_data['components_'][400:500],active_passive_data['components_'][0]))