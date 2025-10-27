import pandas as pd
import random
import itertools
import numpy as np

def calculate_driver_probabilities(historical_df, drivers, all_positions):
    """
    Calculates the weighted probability for each driver to finish in each
    position, based on historical data. Uses Laplace (add-one) smoothing
    to ensure every driver has a non-zero chance of any outcome.

    Args:
        historical_df (pd.DataFrame): DataFrame with 'Driver' and past race/sprint columns.
        drivers (list): List of driver names.
        all_positions (list): List of all possible finishing positions (e.g., '1'...'20', 'DNF').

    Returns:
        dict: A dictionary where keys are driver names and values are
              another dict of {position: probability}.
              e.g., {'DriverA': {'1': 0.5, '2': 0.2, ...}}
    """
    print("Calculating driver performance profiles from historical data...")
    
    probabilities = {}
    
    # Get all columns that contain results (i.e., not the 'Driver' column)
    result_columns = [col for col in historical_df.columns if col != 'Driver']
    
    for driver in drivers:
        # Get all historical results for this driver
        driver_history_row = historical_df[historical_df['Driver'] == driver]
        
        if driver_history_row.empty:
            print(f"  Warning: No historical data for {driver}. Using equal probabilities.")
            # Assign equal probability if no history exists
            counts = {pos: 1 for pos in all_positions}
        else:
            # Melt the row into a single series of results
            driver_results = driver_history_row[result_columns].melt()['value']
            # Convert to string to match 'all_positions'
            driver_results = driver_results.astype(str)
            # Count the occurrences of each finishing position
            counts = driver_results.value_counts().to_dict()

        # --- Laplace (Add-One) Smoothing ---
        # Add 1 to every possible outcome. This ensures that even if a
        # driver has never finished P1, they still have a *small* chance
        # to do so in the simulation.
        smoothed_counts = {pos: counts.get(pos, 0) + 1 for pos in all_positions}
        
        # Calculate final probabilities
        total_smoothed_count = sum(smoothed_counts.values())
        driver_probs = {pos: count / total_smoothed_count for pos, count in smoothed_counts.items()}
        
        probabilities[driver] = driver_probs
        
    print("Performance profiles calculated.")
    return probabilities


def run_championship_simulation(
    standings_df,
    points_df,
    historical_df,
    num_scenarios,
    output_filename="championship_scenarios.csv"
):
    """
    Simulates a racing championship for a given number of scenarios
    using weighted probabilities based on historical performance.
    """
    print(f"Starting weighted simulation of {num_scenarios} scenarios...")

    drivers = standings_df['Driver'].tolist()
    num_drivers = len(drivers)

    # Possible outcomes: positions 1-20 and DNF.
    # CRITICAL: We now use strings to match the data from CSVs.
    possible_positions = [str(p) for p in range(1, 21)] + ['DNF']
    
    if num_drivers > len(possible_positions):
        print(f"Error: More drivers ({num_drivers}) than possible positions ({len(possible_positions)}).")
        return

    # Calculate the weighted performance profiles for each driver
    driver_probabilities = calculate_driver_probabilities(historical_df, drivers, possible_positions)

    # --- DYNAMICALLY FIND EVENTS ---
    # **FIX:** Instead of looking for '_Points', just get all columns.
    # Since 'Position' is the index, this list will only be event columns.
    event_point_columns = points_df.columns.tolist()
    num_events = len(event_point_columns)
    
    if num_events == 0:
        print("Error: No event columns found in 'last_races.csv'.")
        print("Please ensure your columns are named (e.g., 'Sprint_5', 'Race_21')")
        return
        
    print(f"Simulating {num_events} remaining events: {event_point_columns}")

    all_scenario_results = []

    for i in range(num_scenarios):
        if (i + 1) % 1000 == 0:
            print(f"  ...simulated {i + 1}/{num_scenarios} scenarios")

        current_scenario_points = standings_df.set_index('Driver')['CurrentPoints'].to_dict()

        # --- Simulate each event (Race or Sprint) ---
        for event_col_name in event_point_columns:
            
            # --- NEW WEIGHTED ASSIGNMENT LOGIC ---
            available_positions = list(possible_positions)
            shuffled_drivers = list(drivers)
            random.shuffle(shuffled_drivers) # Avoids bias in assignment order
            
            driver_results = {}

            for driver in shuffled_drivers:
                driver_probs_all = driver_probabilities[driver]
                available_probs = [driver_probs_all[pos] for pos in available_positions]
                
                total_available_prob = sum(available_probs)
                if total_available_prob == 0:
                    normalized_probs = [1.0 / len(available_positions)] * len(available_positions)
                else:
                    normalized_probs = [p / total_available_prob for p in available_probs]
                
                chosen_position = random.choices(available_positions, weights=normalized_probs, k=1)[0]
                
                driver_results[driver] = chosen_position
                available_positions.remove(chosen_position)
            
            # --- Award points for the event ---
            for driver in drivers:
                position = driver_results[driver]
                # Look up points from the correct column (e.g., 'Sprint_5')
                points_awarded = points_df.loc[position, event_col_name]
                current_scenario_points[driver] += points_awarded
        
        # --- Scenario complete, process and store final results ---
        sorted_standings = sorted(current_scenario_points.items(), key=lambda item: item[1], reverse=True)
        
        champion = sorted_standings[0][0]
        max_points = sorted_standings[0][1]
        second_place = sorted_standings[1][0]
        third_place = sorted_standings[2][0]

        final_results = {f'{driver}_FinalPoints': points for driver, points in current_scenario_points.items()}
        
        final_results['ScenarioID'] = i + 1
        final_results['Champion'] = champion
        final_results['Second Place'] = second_place
        final_results['Third Place'] = third_place
        final_results['Maximum Points'] = max_points

        all_scenario_results.append(final_results)

    results_df = pd.DataFrame(all_scenario_results)
    new_cols_order = ['ScenarioID', 'Champion', 'Second Place', 'Third Place', 'Maximum Points']
    final_cols = new_cols_order + [col for col in results_df.columns if col not in new_cols_order]
    results_df = results_df[final_cols]

    results_df.to_csv(output_filename, index=False)
    print(f"\nWeighted simulation complete! Results saved to '{output_filename}'")


if __name__ == '__main__':
    # --- Configuration ---
    NUMBER_OF_SCENARIOS = 10000
    
    # --- File Loading ---
    try:
        initial_standings_df = pd.read_csv('current_standings.csv')
        # **FIX:** Load 'last_races.csv' instead of 'points_system.csv'
        race_points_df = pd.read_csv('last_races.csv')
        historical_results_df = pd.read_csv('historical_results.csv')
        
        # Set the 'Position' column as the index for quick point lookups
        race_points_df['Position'] = race_points_df['Position'].astype(str)
        race_points_df.set_index('Position', inplace=True)
        
        # Ensure historical data is read as strings
        historical_results_df = historical_results_df.astype(str)
        historical_results_df['Driver'] = historical_results_df['Driver'].astype(str)

    except FileNotFoundError as e:
        print(f"Error: {e}.")
        # **FIX:** Updated error message
        print("Make sure 'current_standings.csv', 'last_races.csv', and 'historical_results.csv' are in the same directory.")
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
    else:
        # Run the simulation with the loaded data
        run_championship_simulation(
            standings_df=initial_standings_df,
            points_df=race_points_df,
            historical_df=historical_results_df,
            num_scenarios=NUMBER_OF_SCENARIOS
        )

