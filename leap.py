import bridge

import random
import numpy as np

from leap_ec import ops
from leap_ec import Representation
from leap_ec import Individual
from leap_ec.algorithm import generational_ea
from leap_ec.decoder import IdentityDecoder
from leap_ec.problem import ScalarProblem

# Easy place to manipulate the values
FRAMES = 60
POPULATION = 10
GENS = 10
TRN_SIZE = 3

# Should hopefully kill off individuals who die
def remove_dead():
    def _op(population):
        for ind in population:
            if ind.fitness != float('-inf'):  # or your death condition
                yield ind
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

        # Runs the simulation
        result = bridge.run_genome(phenome)

        # Should kill individuals who die with the use of the "remove_dead" function
        if result['deaths'] >= 1:
            return float('-inf')
        
        # Temporarily optimizing for x
        return result['final_x']


def create_genome(length):
    return np.random.randint(0, 256, size=length, dtype=np.uint8)


if __name__ == "__main__":
    genome_length = FRAMES
    pop_size = POPULATION
    generations = GENS

    problem = ByteArrayProblem(maximize=True)

    representation = Representation(
        individual_cls=Individual,
        decoder=IdentityDecoder(),
        initialize=lambda: create_genome(genome_length)
    )

    final_pop = generational_ea(
        max_generations=generations,
        pop_size=pop_size,
        problem=problem,
        representation=representation,

        pipeline=[
            ops.tournament_selection(k=TRN_SIZE),
            ops.clone,

            ops.UniformCrossover(),

            mutate_bytes(),

            ops.evaluate,
            remove_dead(),
            ops.pool(size=pop_size)
        ]
    )

    # Currently sorts the final genomes and prints them out along with their fitness
    final_pop.sort()
    for i in final_pop:
        print("Genome: ", i.genome)
        print("Fitness", i.fitness)
        print("\n")