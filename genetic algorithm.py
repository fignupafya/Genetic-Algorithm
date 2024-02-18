import os
import random
import networkx as nx
import numpy as np
import time

def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        edges = [tuple(map(int, line.split())) for line in file.readlines()]
    return nx.Graph(edges)

def create_random_partition(graph, num_partitions):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    partition_size = len(nodes) // num_partitions
    partitions = [nodes[i:i + partition_size] for i in range(0, len(nodes), partition_size)]
    return partitions

def evaluate_partition(graph, partition):
    cut_size = 0
    for edge in graph.edges():
        if (edge[0] in partition[0] and edge[1] in partition[1]) or \
           (edge[0] in partition[1] and edge[1] in partition[0]):
            cut_size += 1
    return cut_size

def crossover(parent1, parent2):
    # Represent nodes with indices 0 and 1 to indicate partitions
    parent1_representation = sorted([(node, 0) for node in parent1[0]] + [(node, 1) for node in parent1[1]])
    parent2_representation = sorted([(node, 0) for node in parent2[0]] + [(node, 1) for node in parent2[1]])



    # Perform crossover by swapping partitions from a random index
    crossover_point = random.randint(1, len(parent1_representation) - 2)
    child1_representation = parent1_representation[:crossover_point] + parent2_representation[crossover_point:]
    child2_representation = parent2_representation[:crossover_point] + parent1_representation[crossover_point:]

    # Convert back to partition format
    child1_partition = ([node for node, partition in child1_representation if partition == 0],
                       [node for node, partition in child1_representation if partition == 1])

    child2_partition = ([node for node, partition in child2_representation if partition == 0],
                       [node for node, partition in child2_representation if partition == 1])

    return child1_partition, child2_partition


def mutate(partition):
    # Perform mutation by swapping two random nodes
    mutated_partition = (partition[0][:], partition[1][:])  # Create a copy of the partition

    mutation_point = random.randint(0, min(len(partition[0]), len(partition[1])) - 1)
    mutated_partition[0][mutation_point], mutated_partition[1][mutation_point] = mutated_partition[1][mutation_point], mutated_partition[0][mutation_point]

    return mutated_partition





def genetic_algorithm(graph, num_partitions, generations,log_filenum):
    log=""
    population_size = 10
    population = [create_random_partition(graph, num_partitions) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = [evaluate_partition(graph, individual) for individual in population]
        # Select parents based on fitness scores
        parents = random.choices(population, weights=[1/score for score in fitness_scores], k=2)

        # Perform crossover and mutation to create new individuals
        child1, child2 = crossover(parents[0], parents[1])


        # 10% chance of mutation
        if random.random() < 0.1:
            child1 = mutate(parents[0])
            child2 = mutate(parents[1])
        # 50% chance of second mutation
            if random.random() < 0.5:
                child1 = mutate(child1)
                child2 = mutate(child2)



        # Replace the two least fit individuals with the new children
        worst_indices = np.argsort(fitness_scores)[-2:]
        population[worst_indices[0]] = child1
        population[worst_indices[1]] = child2







        if generation%100==0:
            print(f"Generation {generation}, Best Cut Size: {min(fitness_scores)}")
            log+=f"{generation},{min(fitness_scores)}\n"








    # Return the best partition found
    best_partition_index = np.argmin(fitness_scores)


    print(f"Generation {generations}, Best Cut Size: {min(fitness_scores)}")
    print(f"\n{population[best_partition_index]}")




    log+=f"{generations},{min(fitness_scores)}\n"
    log+=f"{population[best_partition_index]}\n"

    savetofile(log_filenum,log)

    return population[best_partition_index]



def savetofile(log_filenum, data):
    content=os.listdir()
    if "Logs" not in content:
        os.makedirs("Logs")
    filename = f"plotting_data_{log_filenum}.txt"
    with open(os.path.join("Logs",filename), "a", encoding="Utf-8") as file:
        file.write(data)








data_file_path = 'data.txt'
graph = read_graph_from_file(data_file_path)


num_partitions = 2
generations = 200
log_filenum = 1



start_time = time.time()

genetic_algorithm(graph, num_partitions, generations, log_filenum)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time {elapsed_time}")
savetofile(log_filenum,str(elapsed_time))



