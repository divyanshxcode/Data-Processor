import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
from data_processor import load_and_process_data
from analysis_engine import analyze_data_combinations
from excel_handler import export_results



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


st.set_page_config(page_title="Data Analysis Tool", layout="wide")
st.title("IFA Testing Pipeline")

# Step 1: Upload data
st.header("Upload Excel Data")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    # Step 2: Select columns for analysis
    st.header("Select Columns for Analysis")
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns for filtering", columns)
    

    if selected_columns:
        # Step 3: Show column statistics and define thresholds
        st.header("Column Statistics & Threshold Settings")
        
        thresholds = {}
        
        for col in selected_columns:
            with st.expander(f"{col} - Statistics & Settings"):
                col_data = df[col].dropna()
                
                # Check if column is numeric or categorical
                if pd.api.types.is_numeric_dtype(col_data):
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
                        f"Threshold type for {col}",
                        ["Mean", "Median", "Variance", "Sum", "Count", "Custom"],
                        key=f"threshold_type_{col}"
                    )
                    
                    if threshold_type == "Mean":
                        threshold_value = col_data.mean()
                    elif threshold_type == "Median":
                        threshold_value = col_data.median()
                    elif threshold_type == "Variance":
                        threshold_value = col_data.var()
                    elif threshold_type == "Sum":
                        threshold_value = col_data.sum()
                    elif threshold_type == "Count":
                        threshold_value = len(col_data)
                    else:  # Custom
                        threshold_value = st.number_input(
                            f"Custom threshold for {col}",
                            min_value=float(col_data.min()),
                            max_value=float(col_data.max()),
                            value=float(col_data.mean()),
                            key=f"custom_threshold_{col}"
                        )
                    
                    st.info(f"Selected threshold: {threshold_value:.2f}")
                    thresholds[col] = threshold_value
                
                else:
                    # Show statistics for categorical columns
                    unique_values = col_data.unique()
                    value_counts = col_data.value_counts()
                    mode_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else "N/A"
                    
                    st.write("**Unique Values:**")
                    st.write(f"Total unique: {len(unique_values)}")
                    st.write("**Most Common Values:**")
                    st.dataframe(value_counts.head(10))
                    st.write(f"**Mode (Most Frequent):** {mode_value}")
                    
                    # For categorical data, let user select specific values
                    selected_values = st.multiselect(
                        f"Select values to include for {col}",
                        options=unique_values.tolist(),
                        default=[mode_value] if mode_value != "N/A" else [],
                        key=f"selected_values_{col}"
                    )
                    
                    thresholds[col] = selected_values

        # Step 4: Select ID column and result columns
        st.header("4) Select ID Column & Result Columns")
        
        # ID column selection
        id_column = st.selectbox("Select ID column", columns)
        
        # Result columns selection
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Select all toggle
        select_all_results = st.checkbox("Select all numeric columns for results")
        
        if select_all_results:
            result_columns = numeric_columns
            st.info(f"Selected all {len(numeric_columns)} numeric columns for results")
        else:
            result_columns = st.multiselect(
                "Select columns to calculate means for results", 
                numeric_columns,
                default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
            )

        # Step 5: Run analysis
        st.header("Run Combination Analysis")
        st.markdown("---")

        # Optional: add similarity/count columns
        st.subheader("Optional: Add similarity/count columns")
        compute_similarity = st.checkbox("Compute per-row similar counts and group sums")
        similarity_group_cols = []
        similarity_sum_cols = []

        if compute_similarity:
            similarity_group_cols = st.multiselect(
                "Select columns to define similarity groups (rows identical on these columns are considered similar)",
                options=columns,
                default=[id_column] if id_column else []
            )

            numeric_opts = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            similarity_sum_cols = st.multiselect(
                "Select numeric columns to compute group sums for",
                options=numeric_opts,
                default=numeric_opts[:3] if len(numeric_opts) > 0 else []
            )

            if st.button("Apply similarity columns"):
                with st.spinner("Computing similarity columns..."):
                    df = add_similarity_columns(df, similarity_group_cols, similarity_sum_cols)
                st.success("Added similarity columns to DataFrame")
                st.dataframe(df.head(50))
        
        if st.button("Analyze All Combinations"):
            if thresholds and id_column and result_columns:
                with st.spinner("Running analysis for all combinations..."):
                    results = analyze_data_combinations(df, selected_columns, thresholds, id_column, result_columns)
                    
                st.subheader("Analysis Results")
                st.dataframe(results)

                # Step 6: Download results
                if not results.empty:
                    if st.button("📥 Download Results"):
                        export_results(results)
                        st.success("Results exported successfully!")
                else:
                    st.warning("No combinations produced results.")
            else:
                st.error("Please configure thresholds, select ID column, and select result columns.")
