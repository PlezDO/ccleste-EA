# test_bridge.py
# Usage: python3 test_bridge.py

import bridge

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

# A simple test genome: hold right for 60 frames, then jump+right for 20
test_genome = (
    [btn(right=True)] * 60 +
    [btn(right=True, jump=True)] * 20 +
    [btn(right=True)] * 40
)

print(f"Running genome of {len(test_genome)} frames...")
result = bridge.run_genome(test_genome)

print("\n--- Results ---")
print(f"  final_x:      {result['final_x']}")
print(f"  final_y:      {result['final_y']}")
print(f"  deaths:       {result['deaths']}")
print(f"  goal_reached: {result['goal_reached']}")
print(f"  trajectory:   {len(result['trajectory'])} points recorded")
print(f"  first point:  {result['trajectory'][0]}")
print(f"  last point:   {result['trajectory'][-1]}")
