import streamlit as st
import pandas as pd
import io

def generic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df_processed = df.copy()

    # detect numeric columns (float, int)
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        df_processed[f"{col}_last5avg"] = (
            df_processed[col].rolling(window=5, min_periods=1).mean()
        )


    return df_processed

# ---------------------------- Streamlit UI Layer ------------------------------------

def main():
    st.title("Generic Excel Feature Engineering App")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Load Excel
        df_raw = pd.read_excel(uploaded_file)

        df_processed = generic_feature_engineering(df_raw)

        st.subheader("Processed Data")
        st.dataframe(df_processed)

        # Download processed file
        buffer = io.BytesIO()
        df_processed.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            label="Download processed Excel",
            data=buffer,
            file_name="processed_output.xlsx"
        )

if __name__ == "__main__":
    main()
