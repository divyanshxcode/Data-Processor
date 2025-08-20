import pandas as pd
import numpy as np
from itertools import combinations, product

def analyze_data_combinations(df, selected_columns, thresholds, id_column, result_columns):

    results = []
    
    for combo_length in range(1, len(selected_columns) + 1):
        # Get all combinations of columns for this length
        for column_combo in combinations(selected_columns, combo_length):
            
            # For each combination, generate all condition variations
            condition_variations = []
            for col in column_combo:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric: both >= and < conditions
                    condition_variations.append([
                        (col, thresholds[col], '>='),
                        (col, thresholds[col], '<')
                    ])
                else:
                    # For categorical: include condition
                    condition_variations.append([
                        (col, thresholds[col], 'include')
                    ])
            
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
                for col, threshold, operator in condition_set:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if operator == '>=':
                            filtered_df = filtered_df[filtered_df[col] >= threshold]
                            applied_conditions[col] = f"{col} >= {threshold:.2f}"
                        elif operator == '<':
                            filtered_df = filtered_df[filtered_df[col] < threshold]
                            applied_conditions[col] = f"{col} < {threshold:.2f}"
                    else:
                        if operator == 'include' and threshold:  # Only if values selected
                            filtered_df = filtered_df[filtered_df[col].isin(threshold)]
                            applied_conditions[col] = f"{col} in {threshold}"
                        elif not threshold:
                            valid_filter = False
                            break
                
                # Calculate result if filter is valid and data remains
                if valid_filter and not filtered_df.empty:
                    # Create result row with condition columns first
                    result_row = applied_conditions.copy()
                    
                    # Add matching rows count
                    result_row['Matching_Rows'] = len(filtered_df)
                    
                    # Calculate mean for selected result columns
                    for result_col in result_columns:
                        if result_col in filtered_df.columns:
                            # Calculate mean
                            mean_value = filtered_df[result_col].mean()
                            if not pd.isna(mean_value):
                                result_row[f'{result_col}_Mean'] = round(mean_value, 4)
                            
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

def apply_single_condition(df, column, threshold, operator):
    """Apply a single condition to the dataframe"""
    if pd.api.types.is_numeric_dtype(df[column]):
        if operator == '>':
            return df[df[column] > threshold]
        elif operator == '<':
            return df[df[column] < threshold]
        elif operator == '>=':
            return df[df[column] >= threshold]
        elif operator == '<=':
            return df[df[column] <= threshold]
    else:
        if operator == 'include':
            return df[df[column].isin(threshold)]
        elif operator == 'exclude':
            return df[~df[column].isin(threshold)]
    
    return df