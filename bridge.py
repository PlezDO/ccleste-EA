# bridge.py: This file will pull data from the cceleste C code to be used in 
# training of evolutionary algorithms


import ctypes 
from map_data import MAP_DATA, TILE_FLAGS # get hardcoded level data

# Callback type constants copied form celeste.c
CB_BTN  = 2   # game: is button pressed?
CB_MGET = 10  # game: what tile is at map pos?
CB_FGET = 12  # game: does tile t have flag f? 

# Global vars for sim tracking
genome = []
current_frame = 0 
death_count = 0
x_positions = []
y_positions = []
reached_goal = False



lib = ctypes.CDLL('./celeste.so')

# Get variables from C code we care about for training
lib.Celeste_P8_init.argtypes = []
lib.Celeste_P8_init.restype = None

lib.Celeste_P8_update.argtypes = []
lib.Celeste_P8_update.restype = None

lib.Celeste_P8_get_deaths.argtypes = []
lib.Celeste_P8_get_deaths.restype = ctypes.c_int

lib.Celeste_P8_get_player_x.argtypes = []
lib.Celeste_P8_get_player_x.restype = ctypes.c_float

lib.Celeste_P8_get_player_y.argtypes = []
lib.Celeste_P8_get_player_y.restype = ctypes.c_float

CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
lib.Celeste_P8_set_call_func.argtypes = [CALLBACK_TYPE]
lib.Celeste_P8_set_call_func.restype = None

lib.Celeste_P8_start_game.argtypes = []
lib.Celeste_P8_start_game.restype = None

lib.Celeste_P8_load_room.argtypes = [ctypes.c_int, ctypes.c_int]
lib.Celeste_P8_load_room.restype = None

def p8_callback(calltype, a0, a1):
    if calltype == CB_BTN:
        # send button
        return genome[current_frame] & (1 << a0)
    elif calltype == CB_MGET:
        # get map 
        return MAP_DATA[a1*128 + a0] # index 1d array as 2d for pos
    elif calltype == CB_FGET:
        # get tile data
        return TILE_FLAGS[a0] & (1 << a1)
    return 0


c_callback = CALLBACK_TYPE(p8_callback)
lib.Celeste_P8_set_call_func(c_callback)

# stop inputs from being made until player is spawned
def wait_for_spawn():
    for _ in range(60): # first 60 frames
        if lib.Celeste_P8_get_player_x() != -1.0: # if player not yet spawned
            return
        lib.Celeste_P8_update() 



def run_genome(input_genome, room_x=0, room_y=0):
    # reset Global vars for sim tracking
    global genome, current_frame, death_count, x_positions, y_positions, reached_goal
    genome = input_genome
    current_frame = 0      
    death_count = 0        
    x_positions = []      
    y_positions = []     
    reached_goal = False
    
    lib.Celeste_P8_init()
    lib.Celeste_P8_start_game()
    lib.Celeste_P8_load_room(room_x, room_y)
    wait_for_spawn()

    # save spawn location for data reporting
    spawn_x = lib.Celeste_P8_get_player_x()
    spawn_y = lib.Celeste_P8_get_player_y()

    # What level are we training for?
    start_level = lib.Celeste_P8_get_level_index()
    
    for _ in genome:
        lib.Celeste_P8_update()
        if lib.Celeste_P8_get_player_x() != -1.0: # filter out invalid positions
            x_positions.append(lib.Celeste_P8_get_player_x())
            y_positions.append(lib.Celeste_P8_get_player_y())
        death_count = lib.Celeste_P8_get_deaths() 
        current_frame+=1

        # if player advanced to next level, reached goal and end early
        if lib.Celeste_P8_get_level_index() != start_level:
            reached_goal = True
            break

    return {
            # use fallback pos of 8, 96 (starting pos) if last pos invalid
            "final_x": x_positions[-1] if x_positions else spawn_x,
            "final_y": y_positions[-1] if y_positions else spawn_y,

            "deaths": death_count,
            "goal_reached": reached_goal,
            "trajectory": list(zip(x_positions, y_positions))
            }
