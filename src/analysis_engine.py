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

def get_date_columns(df):
    """Get all date columns from the dataframe"""
    date_columns = []
    for col in df.columns:
        if is_date_column(df, col):
            date_columns.append(col)
    return date_columns

def apply_date_filter(df, column, threshold_config):
    """Apply date filtering based on the threshold configuration"""
    col_data = pd.to_datetime(df[column], errors='coerce')
    
    if threshold_config["type"] == "range":
        start_date = pd.to_datetime(threshold_config["start_date"])
        end_date = pd.to_datetime(threshold_config["end_date"])
        # Convert both sides to date for comparison
        mask = (col_data.dt.date >= start_date.date()) & (col_data.dt.date <= end_date.date())
        return df[mask]
        
    elif threshold_config["type"] == "before":
        target_date = pd.to_datetime(threshold_config["date"])
        mask = col_data.dt.date < target_date.date()
        return df[mask]
        
    elif threshold_config["type"] == "after":
        target_date = pd.to_datetime(threshold_config["date"])
        mask = col_data.dt.date > target_date.date()
        return df[mask]
        
    elif threshold_config["type"] == "on":
        target_date = pd.to_datetime(threshold_config["date"])
        mask = col_data.dt.date == target_date.date()
        return df[mask]
        
    elif threshold_config["type"] == "last_n_days":
        cutoff_date = pd.to_datetime(threshold_config["cutoff_date"])
        mask = col_data >= cutoff_date
        return df[mask]
        
    elif threshold_config["type"] == "next_n_days":
        cutoff_date = pd.to_datetime(threshold_config["cutoff_date"])
        mask = col_data <= cutoff_date
        return df[mask]
    
    return df

def generate_date_condition_description(column, threshold_config):
    """Generate human-readable description for date conditions"""
    if threshold_config["type"] == "range":
        start = pd.to_datetime(threshold_config["start_date"]).strftime('%Y-%m-%d')
        end = pd.to_datetime(threshold_config["end_date"]).strftime('%Y-%m-%d')
        return f"{column}: {start} to {end}"
        
    elif threshold_config["type"] == "before":
        date_str = pd.to_datetime(threshold_config["date"]).strftime('%Y-%m-%d')
        return f"{column} before {date_str}"
        
    elif threshold_config["type"] == "after":
        date_str = pd.to_datetime(threshold_config["date"]).strftime('%Y-%m-%d')
        return f"{column} after {date_str}"
        
    elif threshold_config["type"] == "on":
        date_str = pd.to_datetime(threshold_config["date"]).strftime('%Y-%m-%d')
        return f"{column} on {date_str}"
        
    elif threshold_config["type"] == "last_n_days":
        days = threshold_config["days"]
        cutoff = pd.to_datetime(threshold_config["cutoff_date"]).strftime('%Y-%m-%d')
        return f"{column} last {days} days (from {cutoff})"
        
    elif threshold_config["type"] == "next_n_days":
        days = threshold_config["days"]
        cutoff = pd.to_datetime(threshold_config["cutoff_date"]).strftime('%Y-%m-%d')
        return f"{column} first {days} days (until {cutoff})"
    
    return f"{column}: unknown date filter"

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
                    # For date columns, we use the single configuration as is
                    # since each date filter type creates one condition
                    condition_variations.append([(col, threshold_config, 'date')])
                        
                elif pd.api.types.is_numeric_dtype(df[col]):
                    if threshold_config["type"] == "range":
                        # For range: create conditions for each range
                        range_conditions = []
                        for i, (start, end) in enumerate(threshold_config["ranges"]):
                            range_conditions.append(
                                (col, {"start": start, "end": end, "range_id": i+1, "total_ranges": len(threshold_config["ranges"])}, 'range')
                            )
                        condition_variations.append(range_conditions)
                    elif threshold_config["type"] == "multiple_greater_than":
                        # Create conditions for each greater than value
                        multiple_greater_conditions = []
                        for value in threshold_config["values"]:
                            multiple_greater_conditions.append(
                                (col, value, '>')
                            )
                        condition_variations.append(multiple_greater_conditions)
                    elif threshold_config["type"] == "multiple_less_than":
                        # Create conditions for each less than value
                        multiple_less_conditions = []
                        for value in threshold_config["values"]:
                            multiple_less_conditions.append(
                                (col, value, '<')
                            )
                        condition_variations.append(multiple_less_conditions)
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
                        # Handle date filtering using the new date filter function
                        filtered_df = apply_date_filter(filtered_df, col, threshold_data)
                        applied_conditions[col] = generate_date_condition_description(col, threshold_data)
                            
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        if operator == 'range':
                            start = threshold_data["start"]
                            end = threshold_data["end"]
                            range_id = threshold_data["range_id"]
                            total_ranges = threshold_data["total_ranges"]
                            
                            # Apply range condition: >= start and < end (except for last range which includes end)
                            if range_id == total_ranges:  # Last range
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
                            # Format the values list nicely
                            if len(threshold_data) == 1:
                                applied_conditions[col] = f"{col} = {threshold_data[0]}"
                            elif len(threshold_data) <= 3:
                                applied_conditions[col] = f"{col} in [{', '.join(map(str, threshold_data))}]"
                            else:
                                applied_conditions[col] = f"{col} in [{', '.join(map(str, threshold_data[:3]))}...] ({len(threshold_data)} values)"
                        elif not threshold_data:
                            valid_filter = False
                            break
                
                # Calculate result if filter is valid and data remains
                if valid_filter and not filtered_df.empty:
                    # Create result row with condition columns first
                    result_row = applied_conditions.copy()
                    
                    # Add matching rows count
                    result_row['Matching_Rows'] = len(filtered_df)
                    
                    # Calculate statistics for selected result columns
                    for result_col in result_columns:
                        if result_col in filtered_df.columns:
                            col_data = filtered_df[result_col].dropna()
                            
                            if not col_data.empty:
                                # Calculate mean
                                mean_value = col_data.mean()
                                if not pd.isna(mean_value):
                                    result_row[f'{result_col}_Mean'] = round(mean_value, 4)
                                
                                # Calculate max run
                                max_run = calculate_max_run(col_data)
                                result_row[f'{result_col}_Max_Run'] = max_run

                                # Calculate sum
                                sum_value = col_data.sum()
                                if not pd.isna(sum_value):
                                    result_row[f'{result_col}_Sum'] = round(sum_value, 4)
                                
                                # Calculate count
                                result_row[f'{result_col}_Count'] = len(col_data)
                                
                                # Calculate standard deviation
                                std_value = col_data.std()
                                if not pd.isna(std_value):
                                    result_row[f'{result_col}_Std_Dev'] = round(std_value, 4)
                                
                                # Calculate median
                                median_value = col_data.median()
                                if not pd.isna(median_value):
                                    result_row[f'{result_col}_Median'] = round(median_value, 4)
                                
                                # Calculate min and max
                                result_row[f'{result_col}_Min'] = round(col_data.min(), 4)
                                result_row[f'{result_col}_Max'] = round(col_data.max(), 4)
                    
                    # Add actual IDs (first 20 if more than 20)
                    ids = filtered_df[id_column].astype(str).tolist()
                    if len(ids) > 20:
                        ids = ids[:20]
                        result_row['IDs'] = ', '.join(ids) + f" ... ({len(filtered_df) - 20} more)"
                    else:
                        result_row['IDs'] = ', '.join(ids)
                    
                    results.append(result_row)
    
    return pd.DataFrame(results)

def apply_single_condition(df, column, threshold_config, operator):
    """Apply a single condition to the dataframe"""
    if is_date_column(df, column):
        return apply_date_filter(df, column, threshold_config)
    elif pd.api.types.is_numeric_dtype(df[column]):
        if operator == 'range':
            start = threshold_config["start"]
            end = threshold_config["end"]
            range_id = threshold_config["range_id"]
            total_ranges = threshold_config.get("total_ranges", 1)
            # For last range, include the end value
            if range_id == total_ranges:
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