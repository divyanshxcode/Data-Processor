import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
from data_processor import load_and_process_data
from analysis_engine import analyze_data_combinations
from excel_handler import export_results

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
                        ["Mean", "Median", "Custom"],
                        key=f"threshold_type_{col}"
                    )
                    
                    if threshold_type == "Mean":
                        threshold_value = col_data.mean()
                    elif threshold_type == "Median":
                        threshold_value = col_data.median()
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
