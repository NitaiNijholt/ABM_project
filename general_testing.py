from grid import Grid
from simulation import Simulation
import numpy as np
import multiprocessing
import time
import sys


def test_agent_placement_1():

    height = 2
    width = 2
    num_agents = 4

    grid = Grid(width, height)
    sim = Simulation(num_agents, grid)

    assert np.all(sim.grid.agent_matrix != 0), f"Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(sim.grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be properly placed in a 2x2 grid")
    return

def test_agent_placement_2():

    height = 2
    width = 2
    num_agents = 4
    grid = Grid(width, height)
    grid.resource_matrix_wood = np.array([[1, 1], [1, 1]])
    sim = Simulation(num_agents, grid)

    assert np.all(sim.grid.resource_matrix_wood != 0), f"Wood matrix grid is not filled, while it should be for this test"
    assert np.all(sim.grid.agent_matrix != 0), f"Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(sim.grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be placed on a 2x2 grid where all cells contain wood")


def test_agent_placement_3():

    height = 2
    width = 2
    num_agents = 4
    grid = Grid(width, height)
    grid.resource_matrix_stone = np.array([[1, 1], [1, 1]])
    sim = Simulation(num_agents, grid)

    assert np.all(sim.grid.resource_matrix_stone != 0), f"Stone matrix grid is not filled, while it should be for this test"
    assert np.all(sim.grid.agent_matrix != 0), f"Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(sim.grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be placed on a 2x2 grid where all cells contain stone")


def test_agent_placement_4():

    height = 2
    width = 2
    num_agents = 4
    grid = Grid(width, height)
    grid.resource_matrix_stone = np.array([[1, 1], [1, 1]])
    grid.resource_matrix_wood = np.array([[1, 1], [1, 1]])
    sim = Simulation(num_agents, grid)

    assert np.all(sim.grid.resource_matrix_stone != 0), f"Stone matrix grid is not filled, while it should be for this test"
    assert np.all(sim.grid.resource_matrix_wood != 0), f"Wood matrix grid is not filled, while it should be for this test"
    assert np.all(sim.grid.agent_matrix != 0), f"Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(sim.grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be placed on a 2x2 grid where all cells contain both wood and stone")
    

def test_agent_placement_5():

    height = 2
    width = 2
    num_agents = 4
    grid = Grid(width, height)
    grid.house_matrix = np.array([[1, 1], [1, 1]])
    sim = Simulation(num_agents, grid)

    num_agents = 1
    assert np.all(sim.grid.house_matrix != 0), f"House matrix grid is not filled, while it should be for this test"
    assert np.all(sim.grid.agent_matrix == 0), f"An agent is placed, while the entire grid is filled with houses!"
    print("Agents are not placed when the grid is filled with houses!")


    # # Test if indeed no more agents can be placed than fit on the grid
    # num_agents = 5
    # grid = Grid(width, height)
    # sim = Simulation(num_agents, grid)

    # assert np.all(sim.grid.agent_matrix != 0), f"Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    # assert np.array_equal(np.sort(sim.grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"There appears to be something wrong with the IDs of the places agents"


# test_agent_placement()

def pass_function():
    time.sleep(20)
    height = 2
    width = 2
    num_agents = 1
    grid = Grid(width, height)
    grid.house_matrix = np.array([[1, 1], [1, 1]])
    # sim = Simulation(num_agents, grid)

    # assert np.all(sim.grid.house_matrix != 0), f"House matrix grid is not filled, while it should be for this test"
    # assert np.all(sim.grid.agent_matrix == 0), f"An agent is placed, while the entire grid is filled with houses!"
    # print("Agents are not placed when the grid is filled with houses!")

    return
def function_wrapper(func, queue):
    try:
        result = func()
        queue.put(result)
    except Exception as e:
        queue.put(e)

def run_with_timeout(func, timeout):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=function_wrapper, args=(func, queue))
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        print(f"!!!!!!!!!!!!!!!!!!!!!!! Function {func.__name__} timed out and was terminated !!!!!!!!!!!!!!!!!!!!!!!!!!")
        return func

def run_tests_with_timeout(functions, timeout):
    results = []
    for func in functions:
        result = run_with_timeout(func, timeout)
        if result:
            results.append(result)
    return results

def run_stuff(functions_to_test, timeout):
    failing_functions = []
    for func in functions_to_test:
        result = run_with_timeout(func, timeout)
        if result:
            failing_functions.append(result)

    print(f"\n\nThe following functions timed out after {timeout} seconds, if any:")
    for func in failing_functions:
        print(func.__name__)
    
    print("\n\nDo you wish to run those functions again with longer timout times?")
    print("If not, type anything that is not an integer")

    try:
        timeout = int(input("Otherwise, please enter an integer number of seconds reserved for each of the functions above: "))
    except ValueError:
        sys.exit()

    run_stuff(failing_functions, timeout)


if __name__ == '__main__':
    functions_to_test = [test_agent_placement_1,
                         test_agent_placement_2,
                         test_agent_placement_3,
                         test_agent_placement_4,
                         test_agent_placement_5
    ]

    timeout = 3
    run_stuff(functions_to_test, timeout)
