import pandas as pd
import numpy as np
from itertools import combinations, product
from datetime import datetime

def calculate_max_run(series: pd.Series) -> int:
    """
    Calculates the longest consecutive run (streak) of the same value in the series.
    Works for both numeric and categorical data.
    """
    if series.empty:
        return 0
    
    max_run = current_run = 1
    prev_value = series.iloc[0]
    
    for value in series.iloc[1:]:
        if value == prev_value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
        prev_value = value
    
    return max_run

def is_date_column(df, column):
    """Check if a column contains date/datetime data"""
    return pd.api.types.is_datetime64_any_dtype(df[column])

def analyze_data_combinations(df, selected_columns, thresholds, id_column, result_columns):

    results = []
    
    for combo_length in range(1, len(selected_columns) + 1):
        # Get all combinations of columns for this length
        for column_combo in combinations(selected_columns, combo_length):
            
            # For each combination, generate all condition variations
            condition_variations = []
            for col in column_combo:
                threshold_config = thresholds[col]
                
                if is_date_column(df, col):
                    # Handle date columns
                    if threshold_config["type"] == "range":
                        condition_variations.append([
                            (col, {"start": threshold_config["start_date"], "end": threshold_config["end_date"]}, 'date_range')
                        ])
                    elif threshold_config["type"] == "before":
                        condition_variations.append([
                            (col, threshold_config["date"], 'date_before')
                        ])
                    elif threshold_config["type"] == "after":
                        condition_variations.append([
                            (col, threshold_config["date"], 'date_after')
                        ])
                    elif threshold_config["type"] == "on":
                        condition_variations.append([
                            (col, threshold_config["date"], 'date_on')
                        ])
                        
                elif pd.api.types.is_numeric_dtype(df[col]):
                    if threshold_config["type"] == "range":
                        # For range: create conditions for each range
                        range_conditions = []
                        for i, (start, end) in enumerate(threshold_config["ranges"]):
                            range_conditions.append(
                                (col, {"start": start, "end": end, "range_id": i+1}, 'range')
                            )
                        condition_variations.append(range_conditions)
                    elif threshold_config["type"] == "greater_than":
                        condition_variations.append([
                            (col, threshold_config["value"], '>')
                        ])
                    elif threshold_config["type"] == "less_than":
                        condition_variations.append([
                            (col, threshold_config["value"], '<')
                        ])
                    else:  # mean, median, custom
                        # For traditional thresholds: both >= and < conditions
                        condition_variations.append([
                            (col, threshold_config["value"], '>='),
                            (col, threshold_config["value"], '<')
                        ])
                else:
                    # For categorical: include condition with multiple value groups
                    if threshold_config["type"] == "categorical" and threshold_config["value_groups"]:
                        group_conditions = []
                        for group in threshold_config["value_groups"]:
                            if group:  # Only if group has values
                                group_conditions.append(
                                    (col, group, 'include')
                                )
                        condition_variations.append(group_conditions)
            
            # Generate all products of condition variations
            for condition_set in product(*condition_variations):
                # Apply all conditions in this set
                filtered_df = df.copy()
                applied_conditions = {}
                
                # Initialize all selected columns as blank
                for col in selected_columns:
                    applied_conditions[col] = ""
                
                # Apply each condition
                valid_filter = True
                for col, threshold_data, operator in condition_set:
                    if is_date_column(df, col):
                        # Handle date filtering
                        if operator == 'date_range':
                            start_date = pd.to_datetime(threshold_data["start"])
                            end_date = pd.to_datetime(threshold_data["end"])
                            filtered_df = filtered_df[(filtered_df[col] >= start_date) & (filtered_df[col] <= end_date)]
                            applied_conditions[col] = f"{col}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        elif operator == 'date_before':
                            date_val = pd.to_datetime(threshold_data)
                            filtered_df = filtered_df[filtered_df[col] < date_val]
                            applied_conditions[col] = f"{col} before {date_val.strftime('%Y-%m-%d')}"
                        elif operator == 'date_after':
                            date_val = pd.to_datetime(threshold_data)
                            filtered_df = filtered_df[filtered_df[col] > date_val]
                            applied_conditions[col] = f"{col} after {date_val.strftime('%Y-%m-%d')}"
                        elif operator == 'date_on':
                            date_val = pd.to_datetime(threshold_data)
                            # For 'on' date, we check if the date part matches (ignoring time)
                            filtered_df = filtered_df[filtered_df[col].dt.date == date_val.date()]
                            applied_conditions[col] = f"{col} on {date_val.strftime('%Y-%m-%d')}"
                            
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        if operator == 'range':
                            start = threshold_data["start"]
                            end = threshold_data["end"]
                            range_id = threshold_data["range_id"]
                            
                            # Apply range condition: >= start and < end (except for last range which includes end)
                            if range_id == len(thresholds[col]["ranges"]):  # Last range
                                filtered_df = filtered_df[(filtered_df[col] >= start) & (filtered_df[col] <= end)]
                                applied_conditions[col] = f"{col}: [{start:.2f} to {end:.2f}]"
                            else:
                                filtered_df = filtered_df[(filtered_df[col] >= start) & (filtered_df[col] < end)]
                                applied_conditions[col] = f"{col}: [{start:.2f} to {end:.2f})"
                                
                        elif operator == '>=':
                            filtered_df = filtered_df[filtered_df[col] >= threshold_data]
                            applied_conditions[col] = f"{col} >= {threshold_data:.2f}"
                        elif operator == '<':
                            filtered_df = filtered_df[filtered_df[col] < threshold_data]
                            applied_conditions[col] = f"{col} < {threshold_data:.2f}"
                        elif operator == '>':
                            filtered_df = filtered_df[filtered_df[col] > threshold_data]
                            applied_conditions[col] = f"{col} > {threshold_data:.2f}"
                    else:
                        if operator == 'include' and threshold_data:  # Only if values selected
                            filtered_df = filtered_df[filtered_df[col].isin(threshold_data)]
                            applied_conditions[col] = f"{col} in {threshold_data}"
                        elif not threshold_data:
                            valid_filter = False
                            break
                
                # Calculate result if filter is valid and data remains
                if valid_filter and not filtered_df.empty:
                    # Create result row with condition columns first
                    result_row = applied_conditions.copy()
                    
                    # Add matching rows count
                    result_row['Matching_Rows'] = len(filtered_df)
                    
                    # Calculate mean and max run for selected result columns
                    for result_col in result_columns:
                        if result_col in filtered_df.columns:
                            # Calculate mean
                            mean_value = filtered_df[result_col].mean()
                            if not pd.isna(mean_value):
                                result_row[f'{result_col}_Mean'] = round(mean_value, 4)
                            

                            # Calculate max run
                            max_run = calculate_max_run(filtered_df[result_col])
                            result_row[f'{result_col}_Max_Run'] = max_run

                            # Calculate sum
                            sum_value = filtered_df[result_col].sum()
                            if not pd.isna(sum_value):
                                result_row[f'{result_col}_Sum'] = round(sum_value, 4)
                            
                            # Calculate count
                            count_value = filtered_df[result_col].count()
                            if not pd.isna(count_value):
                                result_row[f'{result_col}_Count'] = count_value
                            
                            # Calculate variance
                            var_value = filtered_df[result_col].var()
                            if not pd.isna(var_value):
                                result_row[f'{result_col}_Variance'] = round(var_value, 4)

                    
                    # Add actual IDs (first 20 if more than 20)
                    ids = filtered_df[id_column].astype(str).tolist()
                    if len(ids) > 20:
                        ids = ids[:20]
                    result_row['IDs'] = ', '.join(ids)
                    
                    results.append(result_row)
    
    return pd.DataFrame(results)

def apply_single_condition(df, column, threshold_config, operator):
    """Apply a single condition to the dataframe"""
    if is_date_column(df, column):
        # Handle date filtering
        if operator == 'date_range':
            start_date = pd.to_datetime(threshold_config["start"])
            end_date = pd.to_datetime(threshold_config["end"])
            return df[(df[column] >= start_date) & (df[column] <= end_date)]
        elif operator == 'date_before':
            date_val = pd.to_datetime(threshold_config)
            return df[df[column] < date_val]
        elif operator == 'date_after':
            date_val = pd.to_datetime(threshold_config)
            return df[df[column] > date_val]
        elif operator == 'date_on':
            date_val = pd.to_datetime(threshold_config)
            return df[df[column].dt.date == date_val.date()]
    elif pd.api.types.is_numeric_dtype(df[column]):
        if operator == 'range':
            start = threshold_config["start"]
            end = threshold_config["end"]
            range_id = threshold_config["range_id"]
            # For last range, include the end value
            if range_id == threshold_config.get("total_ranges", 1):
                return df[(df[column] >= start) & (df[column] <= end)]
            else:
                return df[(df[column] >= start) & (df[column] < end)]
        elif operator == '>':
            return df[df[column] > threshold_config]
        elif operator == '<':
            return df[df[column] < threshold_config]
        elif operator == '>=':
            return df[df[column] >= threshold_config]
        elif operator == '<=':
            return df[df[column] <= threshold_config]
    else:
        if operator == 'include':
            return df[df[column].isin(threshold_config)]
        elif operator == 'exclude':
            return df[~df[column].isin(threshold_config)]
    
    return df

def get_date_columns(df):
    """Get all date columns from the dataframe"""
    date_columns = []
    for col in df.columns:
        if is_date_column(df, col):
            date_columns.append(col)
    return date_columns