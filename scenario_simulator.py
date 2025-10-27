import pandas as pd
import random
import itertools

def run_championship_simulation(
    standings_df,
    points_df,
    num_scenarios,
    num_races,
    output_filename="championship_scenarios.csv"
):
    """
    Simulates a racing championship for a given number of scenarios.

    Args:
        standings_df (pd.DataFrame): DataFrame with current driver points.
                                     Expected columns: 'Driver', 'CurrentPoints'
        points_df (pd.DataFrame): DataFrame with points for each position in each race.
                                  Expected columns: 'Position', 'Race1_Points', ..., 'RaceN_Points'
        num_scenarios (int): The number of random scenarios to simulate.
        num_races (int): The number of remaining races.
        output_filename (str): The name of the CSV file to save results.
    """
    print(f"Starting simulation of {num_scenarios} scenarios...")

    drivers = standings_df['Driver'].tolist()
    num_drivers = len(drivers)

    # Possible outcomes: positions 1-20 and DNF.
    possible_positions = list(range(1, 21)) + ['DNF']

    all_scenario_results = []

    for i in range(num_scenarios):
        # Print progress
        if (i + 1) % 1000 == 0:
            print(f"  ...simulated {i + 1}/{num_scenarios} scenarios")

        # Start with the current points for each driver for this new scenario
        current_scenario_points = standings_df.set_index('Driver')['CurrentPoints'].to_dict()

        # --- Simulate each race ---
        for race_num in range(1, num_races + 1):
            race_points_col = f'Race{race_num}_Points'

            # Ensure unique finishing positions for each driver in the race
            # This is crucial for a realistic simulation.
            race_results = random.sample(possible_positions, num_drivers)
            driver_results = dict(zip(drivers, race_results))

            # --- Award points for the race ---
            for driver in drivers:
                position = driver_results[driver]
                
                # Find the points for that position in the current race's points table
                # The points_df index must be 'Position'
                # We convert `position` to a string to match the text-based index of the DataFrame.
                points_awarded = points_df.loc[str(position), race_points_col]
                
                # Add points to the driver's total for this scenario
                current_scenario_points[driver] += points_awarded
        
        # --- Scenario complete, process and store final results ---
        
        # Sort drivers by their final points in descending order to find standings
        sorted_standings = sorted(current_scenario_points.items(), key=lambda item: item[1], reverse=True)
        
        # Extract champion, second, third, and the winning point total
        champion = sorted_standings[0][0]
        max_points = sorted_standings[0][1]
        second_place = sorted_standings[1][0]
        third_place = sorted_standings[2][0]

        # Create a dictionary for this scenario's results
        final_results = {f'{driver}_FinalPoints': points for driver, points in current_scenario_points.items()}
        
        final_results['ScenarioID'] = i + 1
        final_results['Champion'] = champion
        final_results['Second Place'] = second_place
        final_results['Third Place'] = third_place
        final_results['Maximum Points'] = max_points

        all_scenario_results.append(final_results)

    # Convert the list of dictionaries to a DataFrame for easy export
    results_df = pd.DataFrame(all_scenario_results)

    # Reorder columns for better readability
    new_cols_order = [
        'ScenarioID', 'Champion', 'Second Place', 'Third Place', 'Maximum Points'
    ]
    # Add the individual driver point columns after the main summary columns
    final_cols = new_cols_order + [col for col in results_df.columns if col not in new_cols_order]
    results_df = results_df[final_cols]

    # Save the results to a CSV file
    results_df.to_csv(output_filename, index=False)
    print(f"\nSimulation complete! Results saved to '{output_filename}'")


if __name__ == '__main__':
    # --- Configuration ---
    # Set how many random scenarios you want to run. 10,000 is a good start.
    # Increase for more statistical accuracy.
    NUMBER_OF_SCENARIOS = 30000
    NUMBER_OF_RACES = 6
    
    # --- File Loading ---
    try:
        # Load the initial data from CSV files
        # The user should edit these CSV files with their actual data.
        initial_standings_df = pd.read_csv('current_standings.csv')
        race_points_df = pd.read_csv('points_system.csv')
        
        # Set the 'Position' column as the index for quick point lookups
        race_points_df.set_index('Position', inplace=True)

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'current_standings.csv' and 'points_system.csv' are in the same directory.")
    else:
        # Run the simulation with the loaded data
        run_championship_simulation(
            standings_df=initial_standings_df,
            points_df=race_points_df,
            num_scenarios=NUMBER_OF_SCENARIOS,
            num_races=NUMBER_OF_RACES
        )

