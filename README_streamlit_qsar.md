# QSAR Streamlit Dashboard

## Files
- `app_streamlit_qsar.py` - Streamlit dashboard entrypoint
- `qsar_core_streamlit.py` - reusable QSAR workflow backend
- `requirements_streamlit_qsar.txt` - Python dependencies

## Local run
```bash
pip install -r requirements_streamlit_qsar.txt
streamlit run app_streamlit_qsar.py
```

## Expected inputs
Training workbook columns:
- compound name
- SMILES
- IC50 in µM

Prediction workbook columns:
- compound name
- SMILES

The app lets you map the exact column names from the sidebar.
