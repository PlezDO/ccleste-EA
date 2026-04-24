
from datetime import datetime

import random
import numpy as np

from dask.distributed import Client, LocalCluster

from leap_ec import ops
from leap_ec import Representation
from leap_ec import Individual
from leap_ec.algorithm import generational_ea
from leap_ec.decoder import IdentityDecoder
from leap_ec.problem import ScalarProblem
from leap_ec.distrib import DistributedIndividual, synchronous

import multiprocessing

# Easy place to manipulate the values
FRAMES = 500
POPULATION = 10
GENS = 10
TRN_SIZE = 3
N_WORKERS = multiprocessing.cpu_count()

GOAL_X = 112
GOAL_Y = 0

# Should hopefully kill off individuals who die
def remove_dead(fallback):
    def _op(population):
        #elitest_survival exepcts list in multiprocessed context
        living = []
        for ind in population:
            if ind.fitness != float('-inf'):
                living.append(ind)
        return living if living else fallback # if all died, use previos gen
    return _op

# Should mutate bytes in the genomes
def mutate_bytes(p=0.1):
    def _mutate(population):
        for individual in population:
            genome = individual.genome

            for i in range(len(genome)):
                if random.random() < p:
                    genome[i] = random.randint(0, 255)

            yield individual

    return _mutate

class ByteArrayProblem(ScalarProblem):
    def __init__(self, maximize=True):
        super().__init__(maximize=maximize)

    # Our custom fitness function
    def evaluate(self, phenome):
        # import bridge here instead so each dask gets its own shared C lib
        import bridge
        import math

        # Runs the simulation
        result = bridge.run_genome(phenome)

        if result['goal_reached']:
            return float('inf') # if player reaches goal, return infinite fitness

        # return closest point player got to goal
        # this might be problematic, we should consider time penalties, or saving last pos
        best_dist = float('inf')
        for x, y in result['trajectory']:
            dist = math.sqrt((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2)
            if dist < best_dist:
                best_dist = dist

        # Invert so higher fitness = closer to goal
        if best_dist == float('inf'):
            return 0
        return -best_dist



def create_genome(length):
    return np.random.randint(0, 256, size=length, dtype=np.uint8)

# Takes a genome and saves it in TAS in the genomes dir
def save_tas(genome, fitness, generation):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"genomes/gen{generation}_fit{fitness:.1f}_{timestamp}.tas"

    with open(path, "w") as f:
        for b in genome:
            f.write(f"{int(b)},")
    print(f"Genome saved to {path}")

    return path



if __name__ == "__main__":
    genome_length = FRAMES
    pop_size = POPULATION
    generations = GENS

    problem = ByteArrayProblem(maximize=True)


    representation = Representation(
        individual_cls=DistributedIndividual,
        decoder=IdentityDecoder(),
        initialize=lambda: create_genome(genome_length)
    )

    # LocalCluster creates processes, with each having its own memory space. 
    # This way C globals for celeste sim won't cause collisions
    with LocalCluster(n_workers=N_WORKERS, threads_per_worker=1) as cluster, Client(cluster) as client:
    
        for gen in range(generations):
            final_pop = generational_ea(
                max_generations=1,
                pop_size=pop_size,
                problem=problem,
                representation=representation,
                k_elites=2, # always keep 2 best individuals so multiprocessing doesnt crash

                pipeline=[
                    ops.tournament_selection(k=TRN_SIZE),
                    ops.clone,

                    ops.UniformCrossover(),

                    mutate_bytes(),

                    #ops.evaluate,
                    synchronous.eval_pool(client=client, size=pop_size),

                    #ops.pool(size=pop_size)
                ]
            )

            parents = final_pop
            best_fitness = float('-inf')
            best = None
            for ind in parents:
                if ind.fitness > best_fitness:
                    best_fitness = ind.fitness
                    best = ind
        print(f"Gen {gen+1} | best fitness: {best_fitness:.2f}")

    # Currently sorts the final genomes and prints them out along with their fitness
    final_pop.sort()

    # ensure pop is not empty
    if final_pop:
        best_ind = final_pop[-1]
        save_tas(best_ind.genome, best_ind.fitness, generations)
    else:
        print("No individual survived")


    for i in final_pop:
        print("Genome: ", i.genome)
        print("Fitness", i.fitness)
        print("\n")
