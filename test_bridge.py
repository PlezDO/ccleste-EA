# test_bridge.py
# Usage: python3 test_bridge.py <genome.csv>

import bridge
import csv
import os
import sys
import math

GOAL_X = 112
GOAL_Y = 0

# Button bit positions (from celeste.c enum)
# k_left=0, k_right=1, k_up=2, k_down=3, k_jump=4, k_dash=5
def btn(right=False, left=False, jump=False, dash=False):
    """Pack button states into a single integer bitmask."""
    b = 0
    if left:  b |= (1 << 0)
    if right: b |= (1 << 1)
    if jump:  b |= (1 << 4)
    if dash:  b |= (1 << 5)
    return b


def load_genome_from_csv(path):
    genome = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for field in row:
                field = field.strip()
                if not field:
                    continue
                genome.append(int(field))
    return genome

# A simple test genome: hold right for 60 frames, then jump+right for 20
default_genome = (
    [btn(right=True)] * 60 +
    [btn(right=True, jump=True)] * 20 +
    [btn(right=True)] * 40
)

genomes_dir = os.path.join(os.path.dirname(__file__), 'genomes')

if len(sys.argv) == 2:
    genome_name = os.path.basename(sys.argv[1])
    genome_path = os.path.join(genomes_dir, genome_name)
    test_genome = load_genome_from_csv(genome_path)
else:
    print('No genome CSV provided, using default test genome.')
    test_genome = default_genome

print(f"Running genome of {len(test_genome)} frames...")
result = bridge.run_genome(test_genome)

best_score = float('-inf')
for x, y in result['trajectory']:
    height_score = -y * 5 # reward low y 
    dist = math.sqrt((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2)
    score = height_score - dist
    if score > best_score:
        best_score = score

print("\n--- Results ---")
print(f"  final_x:      {result['final_x']}")
print(f"  final_y:      {result['final_y']}")
print(f"  deaths:       {result['deaths']}")
print(f"  goal_reached: {result['goal_reached']}")
print(f"  trajectory:   {len(result['trajectory'])} points recorded")
print(f"  first point:  {result['trajectory'][0]}")
print(f"  last point:   {result['trajectory'][-1]}")
print(f"  best_score:   {best_score:.2f}")
