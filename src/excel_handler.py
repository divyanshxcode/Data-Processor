import pandas as pd
import streamlit as st
from io import BytesIO
import datetime

def export_results(results_df, filename=None):
    """
    Export results DataFrame to Excel and provide download link
    
    Args:
        results_df: DataFrame with analysis results
        filename: Optional custom filename
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.xlsx"
    
    # Create Excel file in memory
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Analysis Results']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with formatting
        for col_num, value in enumerate(results_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            # Auto-adjust column width
            worksheet.set_column(col_num, col_num, max(len(value) + 2, 15))
    
    # Reset buffer position
    output.seek(0)
    
    # Provide download button
    st.download_button(
        label="📥 Download Excel File",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results"
    )

def save_results_locally(results_df, filepath):
    """
    Save results to local file (for testing purposes)
    """
    try:
        results_df.to_excel(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False