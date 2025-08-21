import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
from datetime import datetime, date
from data_processor import load_and_process_data
from analysis_engine import analyze_data_combinations, is_date_column, get_date_columns
from excel_handler import export_results
from similarity_utils import add_similarity_columns



def calculate_consistency(series: pd.Series) -> float:
    """
    Calculates consistency as the percentage of the most frequent value 
    compared to the total count (works for numeric or categorical data).
    """
    if series.empty:
        return 0.0
    value_counts = series.value_counts(normalize=True)
    return float(value_counts.iloc[0] * 100)  # percentage of most common value


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


def add_similarity_columns(df: pd.DataFrame, group_by_cols: list, sum_cols: list) -> pd.DataFrame:
    """
    Adds per-row similarity counts and per-group sums to the DataFrame.

    - similar_count: number of rows that share the same values for all columns in group_by_cols
    - similar_sum_<col>: sum of <col> over the group defined by group_by_cols

    If group_by_cols is empty, returns the original DataFrame.
    """
    if not group_by_cols:
        return df

    # work on a copy to avoid mutating original unintentionally
    df = df.copy()

    # Choose a stable column to count; using any column works since group size is same
    try:
        grouped = df.groupby(group_by_cols, dropna=False)
    except Exception:
        # fallback: if grouping fails (e.g., unhashable types), return original
        return df

    # similar_count
    df["similar_count"] = grouped.transform("size").iloc[:, 0]

    # similar sums for numeric sum_cols
    for col in sum_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"similar_sum_{col}"] = grouped[col].transform("sum")

    return df


st.set_page_config(
    page_title="Data Analysis Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue/white minimal theme
st.markdown("""
<style>
    .main-header {
        color: #2563eb;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #60a5fa;
        padding-bottom: 0.5rem;
    }
    .section-header {
        color: #1d4ed8;
        font-size: 1.5rem;
        font-weight: 500;
        margin: 2rem 0 1rem 0;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #60a5fa;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .metric-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.25rem;
    }
    .stButton > button {
        background-color: #ffffff;
        color: #1f2937;
        border: 2px solid #d1d5db;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #f3f4f6;
        border-color: #9ca3af;
        color: #111827;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        border: 2px solid #2563eb;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }
    .stButton > button[kind="secondary"] {
        background-color: #6b7280;
        color: white;
        border: 2px solid #6b7280;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #4b5563;
        border-color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Data Analysis Tool</h1>', unsafe_allow_html=True)

# Upload data
st.markdown('<h2 class="section-header">Data Upload</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Select Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    st.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Select columns for analysis
    st.markdown('<h2 class="section-header">Column Selection</h2>', unsafe_allow_html=True)
    
    # Show column types for user reference
    with st.expander("Column Type Reference"):
        date_cols = get_date_columns(df)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if col not in date_cols and col not in numeric_cols]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Date Columns:**")
            if date_cols:
                for col in date_cols:
                    st.write(f"• {col}")
            else:
                st.write("None detected")
                
        with col2:
            st.write("**Numeric Columns:**")
            if numeric_cols:
                for col in numeric_cols[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(numeric_cols) > 10:
                    st.write(f"... and {len(numeric_cols) - 10} more")
            else:
                st.write("None detected")
                
        with col3:
            st.write("**Categorical Columns:**")
            if categorical_cols:
                for col in categorical_cols[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(categorical_cols) > 10:
                    st.write(f"... and {len(categorical_cols) - 10} more")
            else:
                st.write("None detected")
    
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns for analysis", columns)
    

    if selected_columns:
        st.markdown('<h2 class="section-header">Statistical Analysis & Threshold Configuration</h2>', unsafe_allow_html=True)
        thresholds = {}
        
        for col in selected_columns:
            with st.expander(f"Configure {col}"):
                col_data = df[col].dropna()
                
                # Check if column is date/datetime
                if is_date_column(df, col):
                    # Show statistics for date columns
                    min_date = col_data.min()
                    max_date = col_data.max()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Earliest Date", min_date.strftime('%Y-%m-%d'))
                    with col2:
                        st.metric("Latest Date", max_date.strftime('%Y-%m-%d'))
                    
                    # Date filtering options
                    date_filter_type = st.selectbox(
                        f"Date filter method for {col}",
                        ["Range", "Before", "After", "On"],
                        key=f"date_filter_type_{col}"
                    )
                    
                    if date_filter_type == "Range":
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input(
                                f"Start date for {col}",
                                value=min_date.date(),
                                min_value=min_date.date(),
                                max_value=max_date.date(),
                                key=f"start_date_{col}"
                            )
                        with col2:
                            end_date = st.date_input(
                                f"End date for {col}",
                                value=max_date.date(),
                                min_value=min_date.date(),
                                max_value=max_date.date(),
                                key=f"end_date_{col}"
                            )
                        
                        if start_date <= end_date:
                            st.info(f"Selected range: {start_date} to {end_date}")
                            thresholds[col] = {
                                "type": "range", 
                                "start_date": start_date, 
                                "end_date": end_date
                            }
                        else:
                            st.error("Start date must be before or equal to end date")
                            
                    elif date_filter_type == "Before":
                        selected_date = st.date_input(
                            f"Before date for {col}",
                            value=max_date.date(),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"before_date_{col}"
                        )
                        st.info(f"Filter applied: Before {selected_date}")
                        thresholds[col] = {"type": "before", "date": selected_date}
                        
                    elif date_filter_type == "After":
                        selected_date = st.date_input(
                            f"After date for {col}",
                            value=min_date.date(),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"after_date_{col}"
                        )
                        st.info(f"Filter applied: After {selected_date}")
                        thresholds[col] = {"type": "after", "date": selected_date}
                        
                    else:  # On
                        selected_date = st.date_input(
                            f"On date for {col}",
                            value=min_date.date(),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"on_date_{col}"
                        )
                        st.info(f"Filter applied: On {selected_date}")
                        thresholds[col] = {"type": "on", "date": selected_date}
                
                # Check if column is numeric or categorical

                elif pd.api.types.is_numeric_dtype(col_data):
                    # Show statistics for numeric columns
                    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                    with col1:
                        st.metric("Min", f"{col_data.min():.2f}")
                    with col2:
                        st.metric("Max", f"{col_data.max():.2f}")
                    with col3:
                        st.metric("Mean", f"{col_data.mean():.2f}")
                    with col4:
                        st.metric("Median", f"{col_data.median():.2f}")
                    with col5:
                        st.metric("Count", len(col_data))
                    with col6:
                        st.metric("Variance", f"{col_data.var():.2f}")
                    with col7:
                        st.metric("Consistency", f"{calculate_consistency(col_data):.2f}")
                    with col8:
                        st.metric("Max Run", calculate_max_run(col_data))

                    # Threshold selection
                    threshold_type = st.selectbox(
                        f"Threshold method for {col}",
                        ["Mean", "Median", "Custom", "Greater Than", "Less Than", "Range"],
                        key=f"threshold_type_{col}"
                    )
                    
                    if threshold_type == "Mean":
                        threshold_value = col_data.mean()
                        st.info(f"Threshold value: {threshold_value:.2f}")
                        thresholds[col] = {"type": "mean", "value": threshold_value}
                    elif threshold_type == "Median":
                        threshold_value = col_data.median()
                        st.info(f"Threshold value: {threshold_value:.2f}")
                        thresholds[col] = {"type": "median", "value": threshold_value}
                    elif threshold_type == "Custom":
                        threshold_value = st.number_input(
                            f"Custom threshold for {col}",
                            min_value=float(col_data.min()),
                            max_value=float(col_data.max()),
                            value=float(col_data.mean()),
                            key=f"custom_threshold_{col}"
                        )
                        st.info(f"Threshold value: {threshold_value:.2f}")
                        thresholds[col] = {"type": "custom", "value": threshold_value}
                    elif threshold_type == "Greater Than":
                        # Allow user to input multiple greater than values
                        num_thresholds = st.number_input(
                            f"Number of 'greater than' thresholds for {col}",
                            min_value=1,
                            max_value=20,
                            value=3,
                            key=f"num_greater_than_{col}"
                        )
                        
                        greater_than_values = []
                        for i in range(num_thresholds):
                            value = st.number_input(
                                f"Greater than threshold {i+1} for {col}",
                                min_value=float(col_data.min()),
                                max_value=float(col_data.max()),
                                value=float(col_data.min() + (i+1) * (col_data.max() - col_data.min()) / (num_thresholds + 1)),
                                key=f"multiple_greater_than_{col}_{i}"
                            )
                            greater_than_values.append(value)
                        
                        # Sort values to ensure proper ordering
                        greater_than_values.sort()
                        st.info(f"Greater than thresholds: {[f'>{val:.2f}' for val in greater_than_values]}")
                        thresholds[col] = {"type": "multiple_greater_than", "values": greater_than_values}
                    elif threshold_type == "Less Than":
                        # Allow user to input multiple less than values
                        num_thresholds = st.number_input(
                            f"Number of 'less than' thresholds for {col}",
                            min_value=1,
                            max_value=20,
                            value=3,
                            key=f"num_less_than_{col}"
                        )
                        
                        less_than_values = []
                        for i in range(num_thresholds):
                            value = st.number_input(
                                f"Less than threshold {i+1} for {col}",
                                min_value=float(col_data.min()),
                                max_value=float(col_data.max()),
                                value=float(col_data.min() + (i+1) * (col_data.max() - col_data.min()) / (num_thresholds + 1)),
                                key=f"multiple_less_than_{col}_{i}"
                            )
                            less_than_values.append(value)
                        
                        # Sort values to ensure proper ordering
                        less_than_values.sort()
                        st.info(f"Less than thresholds: {[f'<{val:.2f}' for val in less_than_values]}")
                        thresholds[col] = {"type": "multiple_less_than", "values": less_than_values}
                    else:  # Range
                        num_divisions = st.number_input(
                            f"Number of range divisions for {col}",
                            min_value=2,
                            max_value=20,
                            value=5,
                            key=f"range_divisions_{col}"
                        )
                        
                        # Calculate ranges
                        min_val = col_data.min()
                        max_val = col_data.max()
                        range_size = (max_val - min_val) / num_divisions
                        
                        ranges = []
                        for i in range(num_divisions):
                            start = min_val + (i * range_size)
                            if i == num_divisions - 1:  # Last range gets any remainder
                                end = max_val
                            else:
                                end = min_val + ((i + 1) * range_size)
                            ranges.append((start, end))
                        
                        st.info(f"Generated {num_divisions} ranges:")
                        for i, (start, end) in enumerate(ranges):
                            st.write(f"Range {i+1}: {start:.2f} to {end:.2f}")
                        
                        thresholds[col] = {"type": "range", "ranges": ranges}
                
                else:
                    # Show statistics for categorical columns
                    unique_values = col_data.unique()
                    value_counts = col_data.value_counts()
                    mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else "N/A"
                    
                    st.write("**Unique Values:**")
                    st.write(f"Total unique: {len(unique_values)}")
                    st.write("**Value Distribution:**")
                    st.dataframe(value_counts.head(10))
                    st.write(f"**Most Frequent Value:** {mode_value}")
                    
                    # Selection method for categorical data
                    selection_method = st.selectbox(
                        f"Value selection method for {col}",
                        ["Select Specific Values", "Select All Values", "Select Top N Values"],
                        key=f"selection_method_{col}"
                    )
                    
                    if selection_method == "Select Specific Values":
                        # For categorical data, let user select specific values
                        selected_values = st.multiselect(
                            f"Select values to include for {col}",
                            options=unique_values.tolist(),
                            default=[mode_value] if mode_value != "N/A" else [],
                            key=f"selected_values_{col}"
                        )
                        
                        if selected_values:
                            # Create multiple selection options for combinations
                            value_groups = []
                            for i, value in enumerate(selected_values):
                                value_groups.append([value])
                            
                            # Also add option for all selected values together
                            if len(selected_values) > 1:
                                value_groups.append(selected_values)
                            
                            thresholds[col] = {"type": "categorical", "value_groups": value_groups}
                        else:
                            thresholds[col] = {"type": "categorical", "value_groups": []}
                    
                    elif selection_method == "Select All Values":
                        # Use all unique values
                        value_groups = []
                        # Each value as individual group
                        for value in unique_values:
                            value_groups.append([value])
                        # All values together as one group
                        value_groups.append(unique_values.tolist())
                        
                        thresholds[col] = {"type": "categorical", "value_groups": value_groups}
                        st.info(f"Analyzing all {len(unique_values)} values individually and combined")
                    
                    else:  # Select Top N Values
                        top_n = st.number_input(
                            f"Number of top values to include for {col}",
                            min_value=1,
                            max_value=len(unique_values),
                            value=min(5, len(unique_values)),
                            key=f"top_n_{col}"
                        )
                        
                        top_values = value_counts.head(top_n).index.tolist()
                        st.info(f"Selected top {top_n} values: {top_values}")
                        
                        value_groups = []
                        # Each top value as individual group
                        for value in top_values:
                            value_groups.append([value])
                        # All top values together as one group
                        if len(top_values) > 1:
                            value_groups.append(top_values)
                        
                        thresholds[col] = {"type": "categorical", "value_groups": value_groups}

        # Step 4: Select ID column and result columns
        st.markdown('<h2 class="section-header">Output Configuration</h2>', unsafe_allow_html=True)
        
        # ID column selection
        id_column = st.selectbox("Select identifier column", columns)
        
        # Result columns selection
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Select all toggle
        select_all_results = st.checkbox("Select all numeric columns for analysis")
        
        if select_all_results:
            result_columns = numeric_columns
            st.info(f"Selected all {len(numeric_columns)} numeric columns for analysis")
        else:
            result_columns = st.multiselect(
                "Select columns for statistical calculations", 
                numeric_columns,
                default=[]
            )

        # Step 5: Run analysis
        st.markdown('<h2 class="section-header">Execute Analysis</h2>', unsafe_allow_html=True)
        st.markdown("---")

        
        if st.button("Run Combination Analysis", type="primary"):
            if thresholds and id_column and result_columns:
                with st.spinner("Processing data combinations..."):
                    results = analyze_data_combinations(df, selected_columns, thresholds, id_column, result_columns)
                    
                st.markdown('<h3 style="color: #374151;">Result of Analysis</h3>', unsafe_allow_html=True)
                st.dataframe(results)

                # Step 6: Download results
                if not results.empty:
                    if st.button("Download Results", type="secondary"):
                        export_results(results)
                        st.success("Results exported successfully")
                else:
                    st.warning("No combinations produced results")
            else:
                st.error("Please complete configuration: thresholds, identifier column, and result columns are required")
