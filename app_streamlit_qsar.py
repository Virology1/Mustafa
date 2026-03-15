import traceback
import streamlit as st
import pandas as pd

from qsar_core_streamlit import QSARConfig, run_qsar_workflow

st.set_page_config(page_title="QSAR Dashboard", page_icon="🧪", layout="wide")

st.title("🧪 QSAR Prediction Dashboard")
st.write(
    "Upload a training Excel file and a prediction Excel file, map the required columns, and generate "
    "the same QSAR outputs produced by your Python workflow."
)

with st.expander("Expected input format", expanded=False):
    st.markdown(
        """
**Training file** needs columns for:
- compound name
- SMILES
- IC50 in µM

**Prediction file** needs columns for:
- compound name
- SMILES

Both files can be `.xlsx` workbooks. You can choose the sheet index and map the column names below.
        """
    )

col_left, col_right = st.columns(2)
with col_left:
    training_file = st.file_uploader("Upload training_set.xlsx", type=["xlsx"], key="train")
with col_right:
    prediction_file = st.file_uploader("Upload dataset.xlsx", type=["xlsx"], key="pred")

if training_file:
    try:
        preview_train = pd.read_excel(training_file, sheet_name=0)
        training_file.seek(0)
    except Exception as e:
        preview_train = None
        st.error(f"Could not read training file: {e}")
else:
    preview_train = None

if prediction_file:
    try:
        preview_pred = pd.read_excel(prediction_file, sheet_name=0)
        prediction_file.seek(0)
    except Exception as e:
        preview_pred = None
        st.error(f"Could not read prediction file: {e}")
else:
    preview_pred = None

with st.sidebar:
    st.header("Settings")
    train_sheet = st.number_input("Training sheet index", min_value=0, value=0, step=1)
    predict_sheet = st.number_input("Prediction sheet index", min_value=0, value=0, step=1)

    if preview_train is not None:
        train_cols = list(preview_train.columns)
        train_name_col = st.selectbox("Training compound name column", train_cols, index=train_cols.index("Compound") if "Compound" in train_cols else 0)
        train_smiles_col = st.selectbox("Training SMILES column", train_cols, index=train_cols.index("SMILES") if "SMILES" in train_cols else 0)
        train_target_col = st.selectbox("Training IC50 (µM) column", train_cols, index=train_cols.index("IC50_uM") if "IC50_uM" in train_cols else 0)
    else:
        train_name_col = st.text_input("Training compound name column", value="Compound")
        train_smiles_col = st.text_input("Training SMILES column", value="SMILES")
        train_target_col = st.text_input("Training IC50 (µM) column", value="IC50_uM")

    if preview_pred is not None:
        pred_cols = list(preview_pred.columns)
        predict_name_col = st.selectbox("Prediction compound name column", pred_cols, index=pred_cols.index("Compound Name") if "Compound Name" in pred_cols else 0)
        predict_smiles_col = st.selectbox("Prediction SMILES column", pred_cols, index=pred_cols.index("SMILES") if "SMILES" in pred_cols else 0)
    else:
        predict_name_col = st.text_input("Prediction compound name column", value="Compound Name")
        predict_smiles_col = st.text_input("Prediction SMILES column", value="SMILES")

run_clicked = st.button("Run QSAR workflow", type="primary", use_container_width=True)

if preview_train is not None:
    st.subheader("Training file preview")
    st.dataframe(preview_train.head(10), use_container_width=True)

if preview_pred is not None:
    st.subheader("Prediction file preview")
    st.dataframe(preview_pred.head(10), use_container_width=True)

if run_clicked:
    if not training_file or not prediction_file:
        st.error("Please upload both the training and prediction Excel files.")
    else:
        progress_bar = st.progress(0)
        status = st.empty()

        def progress_callback(percent, message):
            progress_bar.progress(int(percent))
            status.info(message)

        try:
            config = QSARConfig(
                train_sheet=int(train_sheet),
                predict_sheet=int(predict_sheet),
                train_smiles_col=train_smiles_col,
                train_target_col=train_target_col,
                train_name_col=train_name_col,
                predict_smiles_col=predict_smiles_col,
                predict_name_col=predict_name_col,
            )

            results = run_qsar_workflow(training_file, prediction_file, config, progress_callback=progress_callback)
            status.success("QSAR workflow completed successfully.")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Best model", results["best_model_name"])
            with m2:
                st.metric("Training compounds", int(results["summary_df"].loc[results["summary_df"]["Metric"] == "Number_of_training_compounds", "Value"].iloc[0]))
            with m3:
                st.metric("Prediction compounds", int(results["summary_df"].loc[results["summary_df"]["Metric"] == "Number_of_prediction_compounds", "Value"].iloc[0]))

            st.subheader("Model comparison")
            st.dataframe(results["model_results_df"], use_container_width=True)

            st.subheader("Top predicted compounds")
            st.dataframe(results["pred_df"].head(20), use_container_width=True)

            st.subheader("Dataset summary")
            st.dataframe(results["summary_df"], use_container_width=True)

            st.subheader("Plots")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.image(results["outputs"]["bytes"]["model_comparison_rmse.png"], caption="Model comparison RMSE")
            with pcol2:
                st.image(results["outputs"]["bytes"]["observed_vs_fitted_best_model.png"], caption="Observed vs fitted")
            with pcol3:
                st.image(results["outputs"]["bytes"]["top_predicted_gcms_compounds.png"], caption="Top predicted compounds")

            st.subheader("Download outputs")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download full ZIP bundle",
                    data=results["outputs"]["bytes"]["qsar_output_bundle.zip"],
                    file_name="qsar_output_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "Download Excel workbook",
                    data=results["outputs"]["bytes"]["QSAR_PLA_IC50_Predictions.xlsx"],
                    file_name="QSAR_PLA_IC50_Predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            with st.expander("Download individual files", expanded=False):
                for filename, file_bytes in results["outputs"]["bytes"].items():
                    if filename == "qsar_output_bundle.zip":
                        continue
                    mime = "text/csv" if filename.endswith(".csv") else "image/png" if filename.endswith(".png") else "application/octet-stream"
                    if filename.endswith(".xlsx"):
                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    st.download_button(
                        f"Download {filename}",
                        data=file_bytes,
                        file_name=filename,
                        mime=mime,
                        key=f"download_{filename}",
                    )

        except Exception as e:
            status.error("Workflow failed.")
            st.error(str(e))
            st.code(traceback.format_exc())
