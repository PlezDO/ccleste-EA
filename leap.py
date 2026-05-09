
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

import os
import csv

# Easy place to manipulate the values
FRAMES = 450
POPULATION = 300
GENS = 300
TRN_SIZE = 20
N_WORKERS = multiprocessing.cpu_count()

GOAL_X = 112
GOAL_Y = 0

# --- Graph Code ---#

# Data Constants

DEFAULT_N = 300
DEFAULT_PM = .05
DEFAULT_PC = .02
DEFAULT_TRN_SIZE = 20

# Parameter Constants

SWEEP_N = [100, 200, 300, 500]
SWEEP_PM = [.01, .03, .05, .08]
SWEEP_PC = [0.0, .01, .02, .05]
SWEEP_TRN_SIZE = [5, 10, 20, 35]

N_ITERATIONS = 5

RESULTS_CSV = "./graph-data/master_results.csv"

def header_writer(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        file = open(path, "w", newline="")
        csv.writer(file).writerow([
            "N", "p_m", "p_c", "trn_size", "iteration", "generation", "best_fitness", "avg_fitness"
        ])
        file.close()

def add_generation_row(path, n, pm, pc, trn, iteration, generation, population):
    fitnesses = []

    for ind in population:
        if ind.fitness is not None and ind.fitness != float('-inf'):
            fitnesses.append(ind.fitness)

    best_fitness = 0.0
    avg_fitness = 0.0
    if fitnesses:
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses) 
    else:
        best_fitness = avg_fitness = float("nan")

    file = open(path, "a", newline="")
    csv.writer(file).writerow([
        n, pm, pc, trn, iteration, generation, best_fitness, avg_fitness
    ])

    file.close()
        

def sweep():
    configs = []

    for n in SWEEP_N:
        for it in range(N_ITERATIONS):
            configs.append((n, DEFAULT_PM, DEFAULT_PC, DEFAULT_TRN_SIZE, it))

    for pm in SWEEP_PM:
        if pm == DEFAULT_PM:
            continue   # already logged by the N sweep
        for it in range(N_ITERATIONS):
            configs.append((DEFAULT_N, pm, DEFAULT_PC, DEFAULT_TRN_SIZE, it))

    for pc in SWEEP_PC:
        if pc == DEFAULT_PC:
            continue
        for it in range(N_ITERATIONS):
            configs.append((DEFAULT_N, DEFAULT_PM, pc, DEFAULT_TRN_SIZE, it))

    for trn in SWEEP_TRN_SIZE:
        if trn == DEFAULT_TRN_SIZE:
            continue
        for it in range(N_ITERATIONS):
            configs.append((DEFAULT_N, DEFAULT_PM, DEFAULT_PC, trn, it))

    return configs



# --- End of Graph Code ---#

# Should hopefully kill off individuals who die
def remove_dead():
    def _op(population):
        #elitest_survival exepcts list in multiprocessed context
        living = []
        for ind in population:
            if ind.fitness != float('-inf'):
                living.append(ind)
        return living 
    return _op

# Should mutate bytes in the genomes
def mutate_bytes(p=0.05):
    def _mutate(population):
        for individual in population:
            genome = individual.genome

            for i in range(len(genome)):
                if random.random() < p:
                    bit = 1 << random.randint(0, 7)
                    genome[i] = np.uint8(genome[i] ^ bit)

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

        # if result['goal_reached']:
        #     return float('inf') # if player reaches goal, return infinite fitness

        # kill individuals who die
        if result['deaths'] >= 1:
           return float('-1000')

        # return closest point player got to goal
        # this might be problematic, we should consider time penalties, or saving last pos
        best_score = float('-inf')
        for x, y in result['trajectory']:
            height_score = -y * 5 # reward low y 
            dist = math.sqrt((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2)
            score = height_score - dist
            if score > best_score:
                best_score = score

        # Invert so higher fitness = closer to goal
        if best_score == float('inf'):
            return 0
        return best_score

def create_genome(length):
    return np.random.randint(0, 256, size=length, dtype=np.uint8)

# Takes a genome and saves it in TAS in the genomes dir
def save_tas(genome, fitness, label):
    os.makedirs("genomes", exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"genomes/gen{label}_fit{fitness:.1f}_{timestamp}.tas"

    with open(path, "w") as f:
        for b in genome:
            f.write(f"{int(b)},")
    print(f"Genome saved to {path}")

    return path

# --- For graphing, containerize EA ---
def run_single(client, n, pm, pc, trn_size, iteration):
    problem = ByteArrayProblem(maximize=True)

    representation = Representation(
        individual_cls=DistributedIndividual,
        decoder=IdentityDecoder(),
        initialize=lambda: create_genome(FRAMES)
    )

    # to track best individual, so we have something if every individual dies
    genghis = None
    genghis_fitness = float('-inf')

    parents = representation.create_population(n, problem)
    parents = list(synchronous.eval_pool(client=client, size=n)(iter(parents)))

    for gen in range(GENS):
        final_pop = generational_ea(
            max_generations=1,
            pop_size=n,
            problem=problem,
            representation=representation,
            k_elites=10, # always keep 10 best individuals so multiprocessing doesnt crash

            pipeline=[
                lambda _: parents,
                ops.tournament_selection(k=trn_size),
                ops.clone,
                ops.UniformCrossover(p_xover=pc),
                mutate_bytes(p=pm),
                synchronous.eval_pool(client=client, size=n),
            ]
        )

        parents = final_pop

        add_generation_row(RESULTS_CSV, n, pm, pc, trn_size, iteration, gen, parents)

        best_fitness = float('-inf')
        best = None
        for ind in parents:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best = ind
        print(f"Gen: {gen+1} | best fitness: {best_fitness:.2f}")

        if best_fitness > genghis_fitness:
            genghis_fitness = best_fitness
            genghis = best

    save_tas(genghis.genome, genghis_fitness, gen)

"""
# Pre-data generatoin main
if __name__ == "__main__":
    header_writer(RESULTS_CSV)
    configs = sweep()
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
 
        # to track best individual, so we have something if every individual dies
        genghis = None
        genghis_fitness = float('-inf')

        for gen in range(generations):
            final_pop = generational_ea(
                max_generations=1,
                pop_size=pop_size,
                problem=problem,
                representation=representation,
                k_elites=10, # always keep 2 best individuals so multiprocessing doesnt crash

                pipeline=[
                    ops.tournament_selection(k=TRN_SIZE),
                    ops.clone,

                    ops.UniformCrossover(p_xover=0.02),

                    mutate_bytes(),

                    #ops.evaluate,
                    synchronous.eval_pool(client=client, size=pop_size),
                    # remove_dead()
                    #ops.pool(size=pop_size)
                ]
            )

            add_generation_row(RESULTS_CSV, n, pm, pc, trn_size, iteration, gen, final_pop)

            parents = final_pop

            best_fitness = float('-inf')
            best = None
            for ind in parents:
                if ind.fitness > best_fitness:
                    best_fitness = ind.fitness
                    best = ind
            print(f"Gen {gen+1} | best fitness: {best_fitness:.2f}")

            if best_fitness > genghis_fitness:
                genghis_fitness = best_fitness
                genghis = best

        save_tas(genghis.genome, genghis_fitness, generations)

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
"""

if __name__ == "__main__":
    header_writer(RESULTS_CSV)
    configs = sweep()

    total = len(configs)
    print(f"Starting data collection; {total} runs")

    with LocalCluster(n_workers=N_WORKERS, threads_per_worker=1) as cluster, Client(cluster) as client:

        for idx, (n, pm, pc, trn, iteration) in enumerate(configs, 1):
            print(f"\nRun {idx}/{total}")
            run_single(client, n, pm, pc, trn, iteration)

    print(f"\nAll Runs done; Results found at: {RESULTS_CSV}")

            
