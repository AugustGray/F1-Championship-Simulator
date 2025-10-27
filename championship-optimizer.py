import pandas as pd
import random
import numpy as np
import math
import copy

# --- GA Configuration ---
TARGET_DRIVER = "VER"  # <--- SET YOUR TARGET DRIVER HERE
POPULATION_SIZE = 1000       # How many scenarios to test at once
GENERATIONS = 100           # How many rounds of "evolution"
MUTATION_RATE = 0.1         # Chance of a random change
ELITE_SIZE = 20             # How many of the best scenarios to keep each round

# --- Helper Functions (Copied from simulator) ---

def calculate_driver_probabilities(historical_df, drivers, all_positions):
    """
    Calculates the weighted probability for each driver to finish in each
    position, based on historical data. Uses Laplace (add-one) smoothing.
    (Identical to the function in scenario_simulator.py)
    """
    print("Calculating driver performance profiles from historical data...")
    probabilities = {}
    result_columns = [col for col in historical_df.columns if col != 'Driver']
    
    for driver in drivers:
        driver_history_row = historical_df[historical_df['Driver'] == driver]
        if driver_history_row.empty:
            print(f"  Warning: No historical data for {driver}. Using equal probabilities.")
            counts = {pos: 1 for pos in all_positions}
        else:
            driver_results = driver_history_row[result_columns].melt()['value'].astype(str)
            counts = driver_results.value_counts().to_dict()

        smoothed_counts = {pos: counts.get(pos, 0) + 1 for pos in all_positions}
        total_smoothed_count = sum(smoothed_counts.values())
        driver_probs = {pos: count / total_smoothed_count for pos, count in smoothed_counts.items()}
        probabilities[driver] = driver_probs
        
    print("Performance profiles calculated.")
    return probabilities

# --- Genetic Algorithm Core Functions ---

def create_individual(drivers, driver_probs, all_positions, event_names):
    """
    Creates one "Individual" (a complete, random championship scenario).
    This is our "chromosome".
    It's a list of dictionaries, one for each event.
    e.g., [{'DriverA': '1', 'DriverB': '5', ...}, {'DriverA': '3', ...}]
    """
    scenario = []
    for _ in event_names:
        event_results = {}
        available_positions = list(all_positions)
        shuffled_drivers = list(drivers)
        random.shuffle(shuffled_drivers)

        for driver in shuffled_drivers:
            driver_probs_all = driver_probs[driver]
            available_probs = [driver_probs_all[pos] for pos in available_positions]
            
            total_available_prob = sum(available_probs)
            if total_available_prob == 0:
                normalized_probs = [1.0 / len(available_positions)] * len(available_positions)
            else:
                normalized_probs = [p / total_available_prob for p in available_probs]
            
            chosen_position = random.choices(available_positions, weights=normalized_probs, k=1)[0]
            
            event_results[driver] = chosen_position
            available_positions.remove(chosen_position)
        scenario.append(event_results)
    return scenario

def calculate_fitness(individual, target_driver, initial_standings, points_df, driver_probs, event_names):
    """
    This is the most critical function. It scores a scenario (an "Individual").
    
    Fitness is based on two things:
    1. DOES THE TARGET DRIVER WIN? If not, fitness is infinitely bad.
    2. HOW PROBABLE IS THIS SCENARIO? We use log-probability to avoid tiny numbers.
       A higher (less negative) log-prob is better.
    """
    current_points = initial_standings.set_index('Driver')['CurrentPoints'].to_dict()
    total_log_probability = 0.0

    for i, event_results in enumerate(individual):
        event_col_name = event_names[i]
        
        for driver, position in event_results.items():
            # Add points for this event
            points_awarded = points_df.loc[position, event_col_name]
            current_points[driver] += points_awarded
            
            # Add to the log-probability
            # We use log(P) to sum probabilities instead of multiplying
            prob = driver_probs[driver][position]
            total_log_probability += math.log(prob)

    # Now, check the final standings
    sorted_standings = sorted(current_points.items(), key=lambda item: item[1], reverse=True)
    
    champion = sorted_standings[0][0]
    
    if champion != target_driver:
        # This scenario is useless, give it the worst possible score
        return -float('inf'), None
    else:
        # This scenario is valid! Its score is its log-probability.
        # We also return the final standings for display
        return total_log_probability, sorted_standings

def crossover(parent1, parent2):
    """
    Combines two parent scenarios to create two children.
    We'll do a simple "single-point crossover" by swapping race results.
    """
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    # Pick a random point to swap (e.g., after event 3)
    num_events = len(parent1)
    if num_events <= 1:
        return child1, child2
        
    crossover_point = random.randint(1, num_events - 1)
    
    # Swap the "genes" (event results)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

def mutate(individual, driver_probs, all_positions):
    """
    Randomly changes one part of a scenario ("mutation").
    We'll pick one random event and swap the positions of two random drivers.
    This introduces new variety.
    """
    mutated_individual = copy.deepcopy(individual)
    
    if not mutated_individual:
        return mutated_individual

    # Pick a random event (gene) to mutate
    event_index = random.randint(0, len(mutated_individual) - 1)
    
    # Pick two random drivers to swap
    drivers = list(mutated_individual[event_index].keys())
    if len(drivers) < 2:
        return mutated_individual
        
    driver1, driver2 = random.sample(drivers, 2)
    
    # Swap their positions
    pos1 = mutated_individual[event_index][driver1]
    pos2 = mutated_individual[event_index][driver2]
    
    mutated_individual[event_index][driver1] = pos2
    mutated_individual[event_index][driver2] = pos1
    
    return mutated_individual

# --- Main Optimizer Execution ---

if __name__ == '__main__':
    # --- 1. Load All Data (same as simulator) ---
    try:
        initial_standings_df = pd.read_csv('current_standings.csv')
        # Use 'last_races.csv' as requested
        race_points_df = pd.read_csv('last_races.csv') 
        historical_results_df = pd.read_csv('historical_results.csv')
        
        race_points_df['Position'] = race_points_df['Position'].astype(str)
        race_points_df.set_index('Position', inplace=True)
        
        historical_results_df = historical_results_df.astype(str)
        historical_results_df['Driver'] = historical_results_df['Driver'].astype(str)

        print(f"--- Championship Optimizer Initialized ---")
        print(f"TARGETING DRIVER: {TARGET_DRIVER}")

    except FileNotFoundError as e:
        print(f"Error: {e}.")
        print("Make sure 'current_standings.csv', 'last_races.csv', and 'historical_results.csv' are in the same directory.")
        exit()
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        exit()

    # --- 2. Get Shared Data (same as simulator) ---
    drivers = initial_standings_df['Driver'].tolist()
    all_positions = [str(p) for p in range(1, 21)] + ['DNF']
    
    # Dynamically get event names from 'last_races.csv'
    event_names = [col for col in race_points_df.columns if col != 'Position']
    
    if not event_names:
        print("Error: No event columns found in 'last_races.csv'.")
        print("Make sure columns are named 'Sprint_5', 'Race_21', etc.")
        exit()
        
    print(f"Simulating {len(event_names)} remaining events: {', '.join(event_names)}")
    
    if TARGET_DRIVER not in drivers:
        print(f"Error: Target driver '{TARGET_DRIVER}' not found in 'current_standings.csv'.")
        exit()

    # Calculate probabilities once
    driver_probabilities = calculate_driver_probabilities(historical_results_df, drivers, all_positions)

    # --- 3. Create Initial Population ---
    print(f"\nCreating initial population of {POPULATION_SIZE} random scenarios...")
    population = [create_individual(drivers, driver_probabilities, all_positions, event_names) for _ in range(POPULATION_SIZE)]
    
    best_overall_scenario = None
    best_overall_fitness = -float('inf')

    # --- 4. Run the Genetic Algorithm "Evolution" ---
    print(f"--- Starting Evolution for {GENERATIONS} Generations ---")
    
    for gen in range(GENERATIONS):
        # Calculate fitness for every individual in the population
        fitness_scores = [calculate_fitness(ind, TARGET_DRIVER, initial_standings_df, race_points_df, driver_probabilities, event_names) for ind in population]
        
        # Pair individuals with their scores
        pop_with_fitness = list(zip(population, fitness_scores))
        
        # Filter out all the "losing" scenarios
        winning_scenarios = [item for item in pop_with_fitness if item[1][0] > -float('inf')]
        
        if not winning_scenarios:
            print(f"Generation {gen+1}/{GENERATIONS}: No winning scenarios found. Trying again...")
            # If no winners, create a new random population and hope for the best
            population = [create_individual(drivers, driver_probabilities, all_positions, event_names) for _ in range(POPULATION_SIZE)]
            continue

        # Sort the *winning* scenarios by their fitness (log-probability)
        winning_scenarios.sort(key=lambda x: x[1][0], reverse=True)
        
        best_gen_fitness, best_gen_standings = winning_scenarios[0][1]
        
        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_scenario = winning_scenarios[0][0]

        print(f"Generation {gen+1}/{GENERATIONS}: Found {len(winning_scenarios)} winning scenarios. Best fitness (log-prob): {best_overall_fitness:.4f}")

        # --- Create the next generation ---
        new_population = []
        
        # 1. Elitism: Keep the best N scenarios as-is
        elites = [ind for ind, (score, standings) in winning_scenarios[:ELITE_SIZE]]
        new_population.extend(elites)
        
        # 2. Crossover & Mutation: Fill the rest of the population
        parents = [ind for ind, (score, standings) in winning_scenarios]
        
        while len(new_population) < POPULATION_SIZE:
            # Select two parents (favoring fitter ones)
            p1, p2 = random.choices(parents, k=2)
            
            # Create children
            child1, child2 = crossover(p1, p2)
            
            # 3. Mutate
            if random.random() < MUTATION_RATE:
                child1 = mutate(child1, driver_probabilities, all_positions)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2, driver_probabilities, all_positions)
                
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        
        population = new_population

    # --- 5. Show Final Results ---
    print("\n--- Optimization Complete ---")
    
    if best_overall_scenario is None:
        print(f"No scenario found in {GENERATIONS} generations where {TARGET_DRIVER} wins.")
        print("Try increasing POPULATION_SIZE or GENERATIONS.")
    else:
        print(f"Found a 'most probable' winning scenario for {TARGET_DRIVER}!")
        print(f"Final Fitness (log-probability): {best_overall_fitness:.4f}")
        print("\n--- OPTIMAL SCENARIO RESULTS ---")
        
        final_points, final_standings = calculate_fitness(best_overall_scenario, TARGET_DRIVER, initial_standings_df, race_points_df, driver_probabilities, event_names)
        
        # --- Prepare data for optimal_scenario.csv ---
        scenario_for_csv = []
        
        for i, event_results in enumerate(best_overall_scenario):
            print(f"\nEvent: {event_names[i]}")
            
            # Add event name to the dictionary for the CSV row
            csv_row = {'Event': event_names[i]}
            csv_row.update(event_results)
            scenario_for_csv.append(csv_row)
            
            # Sort results by position for printing
            sorted_results = sorted(event_results.items(), key=lambda item: int(item[1]) if item[1].isdigit() else 21)
            for driver, pos in sorted_results:
                print(f"  {pos.ljust(3)}: {driver}")

        # Create and save the optimal scenario CSV
        scenario_df = pd.DataFrame(scenario_for_csv)
        # Reorder columns: Event, then drivers
        cols = ['Event'] + [driver for driver in drivers if driver in scenario_df.columns]
        scenario_df = scenario_df[cols]
        scenario_df.to_csv('optimal_scenario.csv', index=False)
        print(f"\nOptimal scenario saved to 'optimal_scenario.csv'")

        print("\n--- FINAL CHAMPIONSHIP STANDINGS ---")
        
        # --- Prepare data for optimal_standings.csv ---
        standings_for_csv = []
        for i, (driver, points) in enumerate(final_standings):
            print(f"  {i+1}. {driver}: {points} points")
            standings_for_csv.append({'Rank': i+1, 'Driver': driver, 'Points': points})
            
        # Create and save the final standings CSV
        standings_df = pd.DataFrame(standings_for_csv)
        standings_df.to_csv('optimal_standings.csv', index=False)
        print(f"Optimal standings saved to 'optimal_standings.csv'")
        
        # --- Prepare data for optimal_cumulative_points.csv ---
        print("\n--- GENERATING CUMULATIVE POINTS CSV ---")
        cumulative_points_history = []
        
        # Start with initial points
        current_points = initial_standings_df.set_index('Driver')['CurrentPoints'].to_dict()
        start_row = {'Event': 'Start'}
        start_row.update(current_points)
        cumulative_points_history.append(start_row)
        
        # "Re-play" the best scenario event by event
        for i, event_results in enumerate(best_overall_scenario):
            event_col_name = event_names[i]
            event_row = {'Event': event_col_name}
            
            for driver, position in event_results.items():
                points_awarded = race_points_df.loc[position, event_col_name]
                current_points[driver] += points_awarded
            
            # Add the updated totals for this event
            event_row.update(current_points)
            cumulative_points_history.append(event_row)

        # Create and save the cumulative points CSV
        cumulative_df = pd.DataFrame(cumulative_points_history)
        # Reorder columns: Event, then drivers
        cols = ['Event'] + [driver for driver in drivers if driver in cumulative_df.columns]
        cumulative_df = cumulative_df[cols]
        cumulative_df.to_csv('optimal_cumulative_points.csv', index=False)
        print(f"Cumulative points history saved to 'optimal_cumulative_points.csv'")

