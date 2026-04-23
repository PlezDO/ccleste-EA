import bridge

import random
import numpy as np

from leap_ec import ops
from leap_ec import Representation
from leap_ec import Individual
from leap_ec.algorithm import generational_ea
from leap_ec.decoder import IdentityDecoder
from leap_ec.problem import ScalarProblem

frames = 10

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

        target = np.array([0 for _ in range(frames)], dtype=np.uint8)

        diff = phenome.astype(np.int32) - target.astype(np.int32)
        return -np.sum(np.abs(diff))


def create_genome(length):
    return np.random.randint(0, 256, size=length, dtype=np.uint8)


if __name__ == "__main__":
    genome_length = frames
    pop_size = 50
    generations = 50

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
            ops.tournament_selection(k=10),
            ops.clone,

            ops.UniformCrossover(p_swap=0.5),

            mutate_bytes(),

            ops.evaluate,
            ops.pool(size=pop_size)
        ]
    )

    # Print best individual
    best = max(final_pop)
    print("Best genome:", best.genome)
    print("Best fitness:", best.fitness)