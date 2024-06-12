from grid import Grid
from simulation import Simulation
import numpy as np
import multiprocessing
from scipy.stats import chisquare

def test_agent_placement_1():

    height = 2
    width = 2
    num_agents = 4
    resource_spawn_period = 10000000000000

    grid = Grid(width, height)
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    assert np.issubdtype(grid.agent_matrix.dtype, np.integer), "!!!!!!!!!! Not all elements in agent_matrix are of type int"
    assert np.all(grid.agent_matrix != 0), f"!!!!!!!!!! Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"!!!!!!!!!! There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be properly placed in a 2x2 grid")

def test_agent_placement_2():

    height = 2
    width = 2
    num_agents = 4
    resource_spawn_period = 1000000000000000
    grid = Grid(width, height)
    grid.resource_matrix_wood = np.array([[1, 1], [1, 1]])
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    assert np.issubdtype(grid.agent_matrix.dtype, np.integer), "!!!!!!!!!! Not all elements in agent_matrix are of type int"
    assert np.issubdtype(grid.resource_matrix_wood.dtype, np.integer), "!!!!!!!!!! Not all elements in resource_matrix_wood are of type int"
    assert np.all(grid.resource_matrix_wood != 0), f"!!!!!!!!!! Wood matrix grid is not filled, while it should be for this test"
    assert np.all(grid.agent_matrix != 0), f"!!!!!!!!!! Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"!!!!!!!!!! There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be placed on a 2x2 grid where all cells contain wood")


def test_agent_placement_3():

    height = 2
    width = 2
    num_agents = 4
    grid = Grid(width, height)
    resource_spawn_period = 1000000000000000
    grid.resource_matrix_stone = np.array([[1, 1], [1, 1]])
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    assert np.issubdtype(grid.agent_matrix.dtype, np.integer), "!!!!!!!!!! Not all elements in agent_matrix are of type int"
    assert np.issubdtype(grid.resource_matrix_stone.dtype, np.integer), "!!!!!!!!!! Not all elements in resource_matrix_stone are of type int"
    assert np.all(grid.resource_matrix_stone != 0), f"!!!!!!!!!! Stone matrix grid is not filled, while it should be for this test"
    assert np.all(grid.agent_matrix != 0), f"!!!!!!!!!! Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"!!!!!!!!!! There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be placed on a 2x2 grid where all cells contain stone")


def test_agent_placement_4():

    height = 2
    width = 2
    num_agents = 4
    grid = Grid(width, height)
    resource_spawn_period = 10000000000000
    grid.resource_matrix_stone = np.array([[1, 1], [1, 1]])
    grid.resource_matrix_wood = np.array([[1, 1], [1, 1]])
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    assert np.issubdtype(grid.agent_matrix.dtype, np.integer), "!!!!!!!!!! Not all elements in agent_matrix are of type int"
    assert np.issubdtype(grid.resource_matrix_stone.dtype, np.integer), "!!!!!!!!!! Not all elements in resource_matrix_stone are of type int"
    assert np.issubdtype(grid.resource_matrix_wood.dtype, np.integer), "!!!!!!!!!! Not all elements in resource_matrix_wood are of type int"
    assert np.all(grid.resource_matrix_stone != 0), f"!!!!!!!!!! Stone matrix grid is not filled, while it should be for this test"
    assert np.all(grid.resource_matrix_wood != 0), f"!!!!!!!!!! Wood matrix grid is not filled, while it should be for this test"
    assert np.all(grid.agent_matrix != 0), f"!!!!!!!!!! Agent matrix grid is not filled, while {num_agents} are placed on a {width}x{height} grid"
    assert np.array_equal(np.sort(grid.agent_matrix, axis=None), np.arange(1, num_agents + 1)), f"!!!!!!!!!! There appears to be something wrong with the IDs of the places agents"
    print("4 agents can be placed on a 2x2 grid where all cells contain both wood and stone")
    

def test_agent_placement_5():

    height = 2
    width = 2
    num_agents = 1
    grid = Grid(width, height)
    resource_spawn_period = 1000000000000000
    grid.house_matrix = np.array([[1, 1], [1, 1]])
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    assert np.issubdtype(grid.agent_matrix.dtype, np.integer), "!!!!!!!!!! Not all elements in agent_matrix are of type int"
    assert np.issubdtype(grid.house_matrix.dtype, np.integer), "!!!!!!!!!! Not all elements in house_matrix are of type int"
    assert np.all(grid.house_matrix != 0), f"!!!!!!!!!! House matrix grid is not filled, while it should be for this test"
    assert np.all(grid.agent_matrix == 0), f"!!!!!!!!!! An agent is placed, while the entire grid is filled with houses!"
    print("Agents are not placed when the grid is filled with houses!")


def test_resource_placement_1():

    height = 2
    width = 2
    num_agents = 0
    num_resources = 10000
    grid = Grid(width, height)
    resource_spawn_period = 100000000000000
    Simulation(num_agents, grid, num_resources=num_resources, resource_spawn_period=resource_spawn_period)

    assert np.issubdtype(grid.resource_matrix_stone.dtype, np.integer), "!!!!!!!!!! Not all elements in resource_matrix_stone are of type int"
    assert np.issubdtype(grid.resource_matrix_wood.dtype, np.integer), "!!!!!!!!!! Not all elements in resource_matrix_wood are of type int"
    assert np.all(grid.resource_matrix_stone != 0), f"!!!!!!!!!! {num_resources} stones should be distributed, but an empty cell is still found in an 2x2 grid: {grid.resource_matrix_stone}"
    assert np.all(grid.resource_matrix_wood != 0), f"!!!!!!!!!! {num_resources} wood should be distributed, but an empty cell is still found in an 2x2 grid: {grid.resource_matrix_wood}"
    print("When placing a large amount of resources, there are no empty spots on a small grid")

def test_resource_placement_2():

    height = 2
    width = 2
    num_agents = 0
    num_resources = 100000
    grid = Grid(width, height)
    resource_spawn_period = 10000000000000
    Simulation(num_agents, grid, num_resources=num_resources, resource_spawn_period=resource_spawn_period)

    observed_stone = grid.resource_matrix_stone.flatten()
    observed_wood = grid.resource_matrix_wood.flatten()

    with np.errstate(divide='ignore', invalid='ignore'):
        _, p_wood = chisquare(observed_wood, f_exp=[np.mean(observed_wood)])
        _, p_stone = chisquare(observed_stone, f_exp=[np.mean(observed_stone)])
        observed_stone[1] += observed_stone[0]
        observed_stone[0] = 0
        _, p_stone_test = chisquare(observed_stone, f_exp=(width*height)*[num_resources/(width*height)])

    assert np.issubdtype(grid.resource_matrix_stone.dtype, np.integer), "Not all elements in resource_matrix_stone are of type int"
    assert np.issubdtype(grid.resource_matrix_wood.dtype, np.integer), "Not all elements in resource_matrix_wood are of type int"
    assert p_wood >= 0.01, f"Wood might not be properly distributed: {observed_wood}"
    assert p_stone >= 0.01, f"Stone might not be properly distributed: {observed_stone}"

    # Check for false positive my manipulating the distribution for additional check
    assert p_stone_test <= 0.01, f"This test is probably malfunctioning"
    print("The resources appear to be distributed properly in an empty grid")


def test_agent_movement_1():

    height = 2
    width = 2
    num_agents = 1
    grid = Grid(width, height)
    resource_spawn_period = 1000000000000000
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    location_history = np.zeros([width, height])

    agent = grid.agents[1]
    agent.objective = 'Nothing'

    position = agent.position
    agent_matrix = grid.agent_matrix.copy()

    location_history[position] += 1
    agent.step()
    location_history[agent.position]

    assert not np.array_equal(position, agent.position), "!!!!!!!!!! Agent does not change his position when making a random move on an empty grid"
    assert not np.array_equal(agent_matrix, grid.agent_matrix), f"!!!!!!!!!! Agent matrix does not change when making a random move on an empty grid: {agent_matrix, grid.agent_matrix}"
    assert agent.wood == 0, f"!!!!!!!!!! Agent should not have been able to collect wood but has: {agent.wood}"
    assert agent.stone == 0, f"!!!!!!!!!! Agent should not have been able to collect stone but has: {agent.stone}"
    assert len(agent.houses) == 0, f"!!!!!!!!!! Agent should not have been able to build houses but has: {agent.houses}"
    print("Single agent moves a step on an empty grid")


def test_agent_movement_2():

    height = 2
    width = 2
    num_agents = 1
    grid = Grid(width, height)
    resource_spawn_period = 10000000000000000
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    location_history = np.zeros([width, height])

    timesteps = 100000
    agent = grid.agents[1]
    agent.objective = 'Nothing'

    for _ in range(timesteps):
        agent.step()
        location_history[agent.position] += 1

    measured_history = location_history.flatten()

    with np.errstate(divide='ignore', invalid='ignore'):
        _, p = chisquare(measured_history, f_exp=[np.mean(measured_history)])
    
    assert p >= 0.01, f"Movement on empty grid might not be random, or periodic boundary conditions are not working, as visit counts are: {measured_history}"
    assert np.all(grid.resource_matrix_stone == 0), f"!!!!!!!!!! {timesteps} random movements on a 2x2 grid should not end with stone on grid when it does not spawn, but found: {grid.resource_matrix_stone}"
    assert np.all(grid.resource_matrix_wood == 0), f"!!!!!!!!!! {timesteps} random movements on a 2x2 grid should not end with wood on grid when it does not spawn, but found: {grid.resource_matrix_wood}"
    assert np.all(grid.house_matrix == 0), f"!!!!!!!!!! {timesteps} random movements on a 2x2 grid should not end with houses on grid when there are no resources spawn, but found: {grid.house_matrix}"
    assert agent.wood == 0, f"!!!!!!!!!! Agent should not have been able to collect wood but has: {agent.wood}"
    assert agent.stone == 0, f"!!!!!!!!!! Agent should not have been able to collect stone but has: {agent.stone}"
    assert len(agent.houses) == 0, f"!!!!!!!!!! Agent should not have been able to build houses but has: {agent.houses}"
    print("Movement of an agent on an empty grid appears to be properly randomized, and periodic boundary conditions function properly")


def test_agent_movement_3():

    height = 2
    width = 2
    num_agents = 1
    grid = Grid(width, height)
    resource_spawn_period = 10000000000000000000
    Simulation(num_agents, grid, resource_spawn_period=resource_spawn_period)

    timesteps = 1000
    agent = grid.agents[1]
    agent.objective = 'Nothing'

    for _ in range(timesteps):
        assert len(agent.find_moves()) == 4, "single agent on an empty 2x2 grid does not always appear to have 4 possible moves"
        agent.step()
    print("Single agent on 2x2 grid always has 4 possible moves")


def test_resource_gathering_1():

    height = 2
    width = 2
    num_agents = 1
    num_resources = 1000

    # High house cost so agent will never build
    house_cost = (100000, 100000)
    grid = Grid(width, height, house_cost=house_cost)
    resource_spawn_period = 10000000000000000
    Simulation(num_agents, grid, num_resources=num_resources, resource_spawn_period=resource_spawn_period)

    timesteps = 100000
    agent = grid.agents[1]
    agent.objective = 'Nothing'

    for _ in range(timesteps):
        agent.step()

    assert np.all(grid.resource_matrix_stone == 0), f"!!!!!!!!!! {timesteps} random movements on a 2x2 grid should be enough to collect {num_resources} stone, but but there is still uncollected stone: {grid.resource_matrix_stone}"
    assert np.all(grid.resource_matrix_stone == 0), f"!!!!!!!!!! {timesteps} random movements on a 2x2 grid should be enough to collect {num_resources} stone, but but there is still uncollected stone: {grid.resource_matrix_stone}"
    assert agent.wood == 1000, f"Agent has too little wood: {agent.wood}"
    assert agent.stone == 1000, f"Agent has too little stone: {agent.wood}"
    assert np.all(grid.house_matrix == 0), f"!!!!!!!!!! House costs was set to higher value than combined resources on the grid, so building should not be possible but found: {grid.house_matrix}"
    print("Agent gathers properly")



def function_wrapper(func, queue):
    try:
        func()
        queue.put(None)
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
        return 'timeout', func
    else:
        result = queue.get()
        if result is not None:
            return 'fail', func, result
        return 'success', func

def run_stuff(functions_to_test, timeout):
    timeout_functions = []
    for func in functions_to_test:
        result = run_with_timeout(func, timeout)
        if result[0] == 'fail':
            print(f"Exception in function {func.__name__}: {result[2]}")
        elif result[0] == 'timeout':
            timeout_functions.append(func)

    if len(timeout_functions) == 0:
        print("\n\nNo functions timed out!!")
        return

    print(f"\n\nThe following functions timed out or had exceptions after {timeout} seconds:")
    for func in timeout_functions:
        print(func.__name__)
    
    print("\n\nDo you wish to run those functions again with longer timeout times?")
    print("If not, type anything that is not an integer")

    try:
        timeout = int(input("Otherwise, please enter an integer number of seconds reserved for each of the functions above: "))
    except ValueError:
        return

    run_stuff(timeout_functions, timeout)


if __name__ == '__main__':
    functions_to_test = [
                        test_agent_placement_1,
                        test_agent_placement_2,
                        test_agent_placement_3,
                        test_agent_placement_4,
                        test_agent_placement_5,
                        test_resource_placement_1,
                        test_resource_placement_2,
                        test_agent_movement_1,
                        test_agent_movement_2,
                        test_resource_gathering_1
    ]

    timeout = 3
    run_stuff(functions_to_test, timeout)
