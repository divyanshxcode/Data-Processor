import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
from datetime import datetime, date
from data_processor import load_and_process_data
from analysis_engine import analyze_data_combinations, is_date_column, get_date_columns
from excel_handler import export_results

st.set_page_config(page_title="Data Analysis Tool", layout="wide")
st.title("IFA Testing Pipeline")

# Upload data
st.header("Upload Excel Data")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    # Select columns for analysis
    st.header("Select Columns for Analysis")
    
    # Show column types for user reference
    with st.expander("📊 Column Type Information"):
        date_cols = get_date_columns(df)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if col not in date_cols and col not in numeric_cols]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**📅 Date Columns:**")
            if date_cols:
                for col in date_cols:
                    st.write(f"• {col}")
            else:
                st.write("None detected")
                
        with col2:
            st.write("**🔢 Numeric Columns:**")
            if numeric_cols:
                for col in numeric_cols[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(numeric_cols) > 10:
                    st.write(f"... and {len(numeric_cols) - 10} more")
            else:
                st.write("None detected")
                
        with col3:
            st.write("**📝 Categorical Columns:**")
            if categorical_cols:
                for col in categorical_cols[:10]:  # Show first 10
                    st.write(f"• {col}")
                if len(categorical_cols) > 10:
                    st.write(f"... and {len(categorical_cols) - 10} more")
            else:
                st.write("None detected")
    
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns for filtering", columns)

    if selected_columns:
        st.header("Column Statistics & Threshold Settings")
        thresholds = {}
        
        for col in selected_columns:
            with st.expander(f"{col} - Statistics & Settings"):
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
                        f"Date filter type for {col}",
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
                            st.info(f"Date range: {start_date} to {end_date}")
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
                        st.info(f"Filter: Before {selected_date}")
                        thresholds[col] = {"type": "before", "date": selected_date}
                        
                    elif date_filter_type == "After":
                        selected_date = st.date_input(
                            f"After date for {col}",
                            value=min_date.date(),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"after_date_{col}"
                        )
                        st.info(f"Filter: After {selected_date}")
                        thresholds[col] = {"type": "after", "date": selected_date}
                        
                    else:  # On
                        selected_date = st.date_input(
                            f"On date for {col}",
                            value=min_date.date(),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"on_date_{col}"
                        )
                        st.info(f"Filter: On {selected_date}")
                        thresholds[col] = {"type": "on", "date": selected_date}
                
                # Check if column is numeric or categorical
                elif pd.api.types.is_numeric_dtype(col_data):
                    # Show statistics for numeric columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", f"{col_data.min():.2f}")
                    with col2:
                        st.metric("Max", f"{col_data.max():.2f}")
                    with col3:
                        st.metric("Mean", f"{col_data.mean():.2f}")
                    with col4:
                        st.metric("Median", f"{col_data.median():.2f}")
                    
                    # Threshold selection
                    threshold_type = st.selectbox(
                        f"Threshold type for {col}",
                        ["Mean", "Median", "Custom", "Greater Than", "Less Than", "Range"],
                        key=f"threshold_type_{col}"
                    )
                    
                    if threshold_type == "Mean":
                        threshold_value = col_data.mean()
                        st.info(f"Selected threshold: {threshold_value:.2f}")
                        thresholds[col] = {"type": "mean", "value": threshold_value}
                    elif threshold_type == "Median":
                        threshold_value = col_data.median()
                        st.info(f"Selected threshold: {threshold_value:.2f}")
                        thresholds[col] = {"type": "median", "value": threshold_value}
                    elif threshold_type == "Custom":
                        threshold_value = st.number_input(
                            f"Custom threshold for {col}",
                            min_value=float(col_data.min()),
                            max_value=float(col_data.max()),
                            value=float(col_data.mean()),
                            key=f"custom_threshold_{col}"
                        )
                        st.info(f"Selected threshold: {threshold_value:.2f}")
                        thresholds[col] = {"type": "custom", "value": threshold_value}
                    elif threshold_type == "Greater Than":
                        threshold_value = st.number_input(
                            f"Greater than value for {col}",
                            min_value=float(col_data.min()),
                            max_value=float(col_data.max()),
                            value=float(col_data.mean()),
                            key=f"greater_than_{col}"
                        )
                        st.info(f"Filter: > {threshold_value:.2f}")
                        thresholds[col] = {"type": "greater_than", "value": threshold_value}
                    elif threshold_type == "Less Than":
                        threshold_value = st.number_input(
                            f"Less than value for {col}",
                            min_value=float(col_data.min()),
                            max_value=float(col_data.max()),
                            value=float(col_data.mean()),
                            key=f"less_than_{col}"
                        )
                        st.info(f"Filter: < {threshold_value:.2f}")
                        thresholds[col] = {"type": "less_than", "value": threshold_value}
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
                        
                        st.info(f"Created {num_divisions} ranges:")
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
                    st.write("**Most Common Values:**")
                    st.dataframe(value_counts.head(10))
                    st.write(f"**Mode (Most Frequent):** {mode_value}")
                    
                    # Selection method for categorical data
                    selection_method = st.selectbox(
                        f"Selection method for {col}",
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
                        st.info(f"Will analyze all {len(unique_values)} values individually and combined")
                    
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
