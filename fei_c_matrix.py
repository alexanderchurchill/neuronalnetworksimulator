# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:02:35 2014

@author: FPeng
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Part1: neuron ID and 3D grid for neuron locations from ALEX
start = time.time()
no_layers = 2
layer_dims = np.array(
                   [
                   [10,10],
                   [20,10]
                   ]
                   )
total_neurons = [
               400,
               600
               ]
layer_types = [{
               "types":[0,1],
               "probabilities":[0.8,0.2]
               },
               {
               "types":[0,1],
               "probabilities":[0.8,0.2]
               }
           ]
grid = [np.zeros(layer_dims[i],dtype=np.ndarray) for i in range(no_layers)]
# create the neurons
neuron_types = np.array([np.random.choice(layer_types[layer]["types"],total_neurons[layer],p=layer_types[layer]["probabilities"]) for layer in range(no_layers)])
# create the neuron locations
neuron_locations = [np.hstack((np.random.randint(0,layer_dims[i][0],(total_neurons[i],1)),
                           np.random.randint(0,layer_dims[i][1],(total_neurons[i],1)))) for i in range(len(layer_dims))]
# add neurons to a p x q matrix
for l in range(no_layers):
   for neuron_idx,coors in enumerate(neuron_locations[l]):
       x,y = coors
       if grid[l][x,y] == 0:
           grid[l][x,y] = [neuron_idx]
       else:
           grid[l][x,y].append(neuron_idx)

# ============================ connectivity generator ================================ #

neuron_connection_prob = [[0.2*np.ones((len(layer_types[layer]["types"]),len(layer_types[layer]["types"])),dtype=np.ndarray) for layer in range(no_layers)] for layer in range(no_layers)] # probability of connection matrix, can be set by hand       
# Define the connectivity matrix : 1-1, 1-2; 2-1,2-2 of two layers for now
connectivity_matrix = [[np.zeros((total_neurons[i],total_neurons[j])) for i in range(no_layers)] for j in range(no_layers)]
def connect_square(receiving_layer, connectivity_matrix,
                   square_half_width, square_half_height, sending_neuron, neuron_location, connection_prob):
    neuron_pool = []
    global grid, neuron_types,layer_dims
    layer_dimension = layer_dims[receiving_layer]
    layer_grid = grid[receiving_layer]
    layer_neuron_types = neuron_types[receiving_layer]
    x_start = neuron_location[0]-square_half_width
    if x_start < 0: # Bound for left
        x_start = 0
    x_end = neuron_location[0]+square_half_width
    if x_end > layer_dimension[0]-1: # Bound for right 
        x_end = layer_dimension[0]-1
    y_start = neuron_location[1]-square_half_height
    if y_start < 0: # Bound for left
        y_start = 0
    y_end = neuron_location[1]+square_half_height
    if y_end > layer_dimension[1]-1: # Bound for right
        y_end = layer_dimension[1]-1
    for x in range(x_start, x_end+1):
        for y in range(y_start, y_end+1):
            neuron_pool.append(layer_grid[x][y])
    for indices in neuron_pool:
        if indices != 0: # if not an empty grid
            print indices
            for neuron_ID in indices:
                if layer_neuron_types[neuron_ID]==0 and sending_neuron != neuron_ID: # no self connection
                    if np.random.normal(0.5,0.15) < connection_prob[0]: # connected to type 0
                        connectivity_matrix[neuron_ID][sending_neuron] = 1
                elif layer_neuron_types[neuron_ID] == 1 and sending_neuron != neuron_ID:
                    if np.random.normal(0.5,0.15) < connection_prob[1]: # connected to type 1
                        connectivity_matrix[neuron_ID][sending_neuron] = 1  
    return connectivity_matrix

# ============main loop: loop through each layer, then each neuron==================== #
square_half_width, square_half_height = 2,2
for sending_layer in range(no_layers):
    for receiving_layer in range(no_layers):
        for sending_neuron in range(total_neurons[sending_layer]): # loop through all the neurons that send axons
            neuron_location = neuron_locations[sending_layer][sending_neuron]
            connection_prob = neuron_connection_prob[sending_layer][receiving_layer][neuron_types[sending_layer][sending_neuron]]
            print "connection_prob",connection_prob
            connectivity_matrix[sending_layer][receiving_layer] = connect_square(receiving_layer, connectivity_matrix[sending_layer][receiving_layer], 
                                            square_half_width, square_half_height, sending_neuron, neuron_location, connection_prob)               

print "elapsed:",time.time()-start

# visualisation                   
#plt.plot(np.sum(connectivity_matrix, axis=0))
# plotting connectivity matrix
connectivity_matrix = np.array(connectivity_matrix)
for i in range(0,no_layers):
    for j in range(0,no_layers):
        plt.subplot(2,2,(2*i+j)+1)
        plt.pcolor(connectivity_matrix[i,j],cmap="Greys")
grid_counts = [np.zeros(layer_dims[i],dtype=int) for i in range(no_layers)]
for i in range(len(grid_counts)):
    for row in range(len(grid_counts[i])):
        for column in range(len(grid_counts[i][0])):
            if grid[i][row][column] != 0:
                grid_counts[i][row][column] = len(grid[i][row][column])

# plotting density in layer
for i in range(0,no_layers):
    plt.subplot(1,2,i+1)
    plt.pcolor(grid_counts[i],cmap="Greys")

