import io
import os
import zipfile
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, AllChem, DataStructs

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


@dataclass
class QSARConfig:
    train_sheet: int = 0
    predict_sheet: int = 0
    train_smiles_col: str = "SMILES"
    train_target_col: str = "IC50_uM"
    train_name_col: str = "Compound"
    predict_smiles_col: str = "SMILES"
    predict_name_col: str = "Compound Name"


def ic50_uM_to_pIC50(ic50_uM):
    ic50_uM = np.asarray(ic50_uM, dtype=float)
    return 6.0 - np.log10(ic50_uM)


def pIC50_to_ic50_uM(pic50):
    pic50 = np.asarray(pic50, dtype=float)
    return 10 ** (6.0 - pic50)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "ExactMolWt": rdMolDescriptors.CalcExactMolWt(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "NHOHCount": Lipinski.NHOHCount(mol),
        "NOCount": Lipinski.NOCount(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "Chi0v": Descriptors.Chi0v(mol),
        "Chi1v": Descriptors.Chi1v(mol),
        "Chi2v": Descriptors.Chi2v(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
    }


def morgan_bitvect(smiles, radius=2, nbits=512):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_table(df, smiles_col):
    rows, fps, valid_idx = [], [], []
    for idx, smi in df[smiles_col].items():
        desc = compute_rdkit_descriptors(smi)
        fp = morgan_bitvect(smi)
        if desc is None or fp is None:
            continue
        rows.append(desc)
        fps.append(fp)
        valid_idx.append(idx)

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), []

    X_desc = pd.DataFrame(rows, index=valid_idx)
    X_fp = pd.DataFrame(fps, index=valid_idx, columns=[f"FP_{i}" for i in range(len(fps[0]))])
    return X_desc, X_fp, valid_idx


def get_model_space():
    return {
        "Ridge": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        ),
        "Lasso": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=10000, random_state=RANDOM_STATE))
            ]),
            {"model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]}
        ),
        "ElasticNet": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=10000, random_state=RANDOM_STATE))
            ]),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]}
        ),
        "BayesianRidge": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", BayesianRidge())
            ]),
            {
                "model__alpha_1": [1e-7, 1e-6, 1e-5],
                "model__alpha_2": [1e-7, 1e-6, 1e-5],
                "model__lambda_1": [1e-7, 1e-6, 1e-5],
                "model__lambda_2": [1e-7, 1e-6, 1e-5],
            }
        ),
        "PLS": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", PLSRegression())
            ]),
            {"model__n_components": [1, 2, 3, 4, 5]}
        ),
        "SVR_rbf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf"))
            ]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            }
        ),
        "RandomForest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE))
            ]),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 2, 3, 5],
                "model__min_samples_leaf": [1, 2, 3],
            }
        ),
        "ExtraTrees": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(random_state=RANDOM_STATE))
            ]),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 2, 3, 5],
                "model__min_samples_leaf": [1, 2, 3],
            }
        ),
        "GradientBoosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
            ]),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.9, 1.0],
            }
        ),
        "KNN": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor())
            ]),
            {"model__n_neighbors": [2, 3, 4, 5], "model__weights": ["uniform", "distance"], "model__p": [1, 2]}
        ),
    }


def nested_loocv(X, y, model_space):
    outer_cv = LeaveOneOut()
    results = []
    predictions = {}

    for model_name, (pipe, param_grid) in model_space.items():
        y_true_all, y_pred_all, chosen_params = [], [], []

        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid = GridSearchCV(
                estimator=clone(pipe),
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=LeaveOneOut(),
                n_jobs=1,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            pred = float(best_model.predict(X_test)[0])

            y_true_all.append(float(y_test.values[0]))
            y_pred_all.append(pred)
            chosen_params.append(str(grid.best_params_))

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        results.append({
            "Model": model_name,
            "LOOCV_RMSE_pIC50": rmse(y_true_all, y_pred_all),
            "LOOCV_MAE_pIC50": mean_absolute_error(y_true_all, y_pred_all),
            "LOOCV_R2_pIC50": r2_score(y_true_all, y_pred_all),
            "Mean_Selected_Params": pd.Series(chosen_params).mode().iloc[0] if chosen_params else "",
        })
        predictions[model_name] = pd.DataFrame({"Observed_pIC50": y_true_all, "Predicted_pIC50": y_pred_all})

    res_df = pd.DataFrame(results).sort_values(by="LOOCV_RMSE_pIC50", ascending=True).reset_index(drop=True)
    return res_df, predictions


def fit_best_model(X, y, best_model_name, model_space):
    pipe, param_grid = model_space[best_model_name]
    grid = GridSearchCV(
        estimator=clone(pipe),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=LeaveOneOut(),
        n_jobs=1,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def max_tanimoto_to_training(train_smiles, query_smiles):
    train_fps = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512))

    vals = []
    for smi in query_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or len(train_fps) == 0:
            vals.append(np.nan)
            continue
        qfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
        vals.append(float(np.max(sims)))
    return vals


def confidence_flag(sim):
    if pd.isna(sim):
        return "Unknown"
    if sim >= 0.60:
        return "Higher confidence"
    if sim >= 0.40:
        return "Moderate confidence"
    return "Low confidence"


def load_excel_from_upload(uploaded_file, sheet=0):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], label: str):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}. Available columns: {list(df.columns)}")


def run_qsar_workflow(training_file, predict_file, config: QSARConfig, progress_callback=None) -> Dict[str, Any]:
    if progress_callback:
        progress_callback(5, "Reading uploaded Excel files...")
    train_df = load_excel_from_upload(training_file, config.train_sheet)
    pred_df = load_excel_from_upload(predict_file, config.predict_sheet)

    validate_required_columns(train_df, [config.train_name_col, config.train_target_col, config.train_smiles_col], "training file")
    validate_required_columns(pred_df, [config.predict_name_col, config.predict_smiles_col], "prediction file")

    train_df = train_df[[config.train_name_col, config.train_target_col, config.train_smiles_col]].copy()
    pred_df = pred_df[[config.predict_name_col, config.predict_smiles_col]].copy()

    if progress_callback:
        progress_callback(15, "Cleaning structures and target values...")
    train_df = train_df.dropna(subset=[config.train_smiles_col, config.train_target_col]).copy()
    pred_df = pred_df.dropna(subset=[config.predict_smiles_col]).copy()

    train_df["Canonical_SMILES"] = train_df[config.train_smiles_col].apply(canonicalize_smiles)
    pred_df["Canonical_SMILES"] = pred_df[config.predict_smiles_col].apply(canonicalize_smiles)

    invalid_train = int(train_df["Canonical_SMILES"].isna().sum())
    invalid_pred = int(pred_df["Canonical_SMILES"].isna().sum())

    train_df = train_df.dropna(subset=["Canonical_SMILES"]).copy()
    pred_df = pred_df.dropna(subset=["Canonical_SMILES"]).copy()

    if train_df.empty:
        raise ValueError("No valid training SMILES remained after cleaning.")
    if pred_df.empty:
        raise ValueError("No valid prediction SMILES remained after cleaning.")

    if (train_df[config.train_target_col] <= 0).any():
        raise ValueError("All IC50 values in the training set must be > 0 for pIC50 conversion.")

    train_df["pIC50"] = ic50_uM_to_pIC50(train_df[config.train_target_col].values)
    before_dups = len(train_df)
    train_df = train_df.drop_duplicates(subset=["Canonical_SMILES"]).reset_index(drop=True)
    removed_dups = before_dups - len(train_df)

    if len(train_df) < 6:
        raise ValueError("At least 6 unique valid training compounds are recommended for this workflow.")

    if progress_callback:
        progress_callback(30, "Calculating descriptors...")
    X_train_desc, _, valid_train_idx = build_feature_table(train_df, "Canonical_SMILES")
    train_df = train_df.loc[valid_train_idx].reset_index(drop=True)
    X_train_desc = X_train_desc.reset_index(drop=True)

    X_pred_desc, _, valid_pred_idx = build_feature_table(pred_df, "Canonical_SMILES")
    pred_df = pred_df.loc[valid_pred_idx].reset_index(drop=True)
    X_pred_desc = X_pred_desc.reset_index(drop=True)

    if X_train_desc.empty or X_pred_desc.empty:
        raise ValueError("Descriptor generation failed for one or both uploaded datasets.")

    y = train_df["pIC50"].reset_index(drop=True)

    summary_df = pd.DataFrame({
        "Metric": [
            "Number_of_training_compounds",
            "Number_of_prediction_compounds",
            "Removed_duplicate_training_structures",
            "Invalid_training_smiles_removed",
            "Invalid_prediction_smiles_removed",
            "Min_IC50_uM",
            "Max_IC50_uM",
            "Min_pIC50",
            "Max_pIC50",
            "Median_pIC50",
        ],
        "Value": [
            len(train_df),
            len(pred_df),
            removed_dups,
            invalid_train,
            invalid_pred,
            float(train_df[config.train_target_col].min()),
            float(train_df[config.train_target_col].max()),
            float(train_df["pIC50"].min()),
            float(train_df["pIC50"].max()),
            float(train_df["pIC50"].median()),
        ],
    })

    if progress_callback:
        progress_callback(45, "Running nested LOOCV model benchmarking...")
    model_space = get_model_space()
    model_results_df, cv_predictions = nested_loocv(X_train_desc, y, model_space)

    best_model_name = model_results_df.iloc[0]["Model"]
    if progress_callback:
        progress_callback(75, f"Refitting best model: {best_model_name}...")
    best_model, best_params = fit_best_model(X_train_desc, y, best_model_name, model_space)
    best_model.fit(X_train_desc, y)

    train_df["Fitted_pIC50"] = best_model.predict(X_train_desc)
    train_df["Fitted_IC50_uM"] = pIC50_to_ic50_uM(train_df["Fitted_pIC50"])

    pred_df["Predicted_pIC50"] = best_model.predict(X_pred_desc)
    pred_df["Predicted_IC50_uM"] = pIC50_to_ic50_uM(pred_df["Predicted_pIC50"])
    pred_df["Max_Tanimoto_to_Training"] = max_tanimoto_to_training(
        train_df["Canonical_SMILES"].tolist(), pred_df["Canonical_SMILES"].tolist()
    )
    pred_df["Prediction_Confidence"] = pred_df["Max_Tanimoto_to_Training"].apply(confidence_flag)
    pred_df = pred_df.sort_values(by="Predicted_pIC50", ascending=False).reset_index(drop=True)

    if progress_callback:
        progress_callback(85, "Building Excel, CSV, and PNG outputs...")
    outputs = build_output_files(
        summary_df=summary_df,
        model_results_df=model_results_df,
        train_df=train_df,
        pred_df=pred_df,
        cv_predictions=cv_predictions,
        X_train_desc=X_train_desc,
        X_pred_desc=X_pred_desc,
        best_model_name=best_model_name,
        best_params=best_params,
        predict_name_col=config.predict_name_col,
    )

    if progress_callback:
        progress_callback(100, "Finished.")
    return {
        "summary_df": summary_df,
        "model_results_df": model_results_df,
        "train_df": train_df,
        "pred_df": pred_df,
        "cv_predictions": cv_predictions,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "outputs": outputs,
    }


def build_output_files(summary_df, model_results_df, train_df, pred_df, cv_predictions,
                       X_train_desc, X_pred_desc, best_model_name, best_params, predict_name_col):
    temp_dir = tempfile.mkdtemp(prefix="qsar_streamlit_")
    excel_path = os.path.join(temp_dir, "QSAR_PLA_IC50_Predictions.xlsx")
    model_csv = os.path.join(temp_dir, "model_comparison_loocv.csv")
    summary_csv = os.path.join(temp_dir, "dataset_summary.csv")
    train_csv = os.path.join(temp_dir, "training_set_with_fitted_values.csv")
    pred_csv = os.path.join(temp_dir, "gcms_predictions_ranked.csv")

    model_results_df.to_csv(model_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Dataset_Summary", index=False)
        model_results_df.to_excel(writer, sheet_name="Model_Comparison", index=False)
        train_df.to_excel(writer, sheet_name="Training_Fit", index=False)
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        for model_name, df_pred in cv_predictions.items():
            df_pred.to_excel(writer, sheet_name=model_name[:31], index=False)
        X_train_desc.to_excel(writer, sheet_name="Train_Descriptors", index=False)
        X_pred_desc.to_excel(writer, sheet_name="Predict_Descriptors", index=False)
        pd.DataFrame({"Best_Model": [best_model_name], "Best_Params": [str(best_params)]}).to_excel(
            writer, sheet_name="Best_Model", index=False
        )

    png_paths = plot_outputs(temp_dir, model_results_df, train_df, pred_df, best_model_name, predict_name_col)

    files = {
        "QSAR_PLA_IC50_Predictions.xlsx": excel_path,
        "model_comparison_loocv.csv": model_csv,
        "dataset_summary.csv": summary_csv,
        "training_set_with_fitted_values.csv": train_csv,
        "gcms_predictions_ranked.csv": pred_csv,
    }
    files.update({os.path.basename(p): p for p in png_paths})

    zip_path = os.path.join(temp_dir, "qsar_output_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files.items():
            zf.write(path, arcname=name)
    files["qsar_output_bundle.zip"] = zip_path

    file_bytes = {}
    for name, path in files.items():
        with open(path, "rb") as f:
            file_bytes[name] = f.read()

    return {"dir": temp_dir, "paths": files, "bytes": file_bytes}


def plot_outputs(temp_dir, model_results_df, train_df, pred_df, best_model_name, predict_name_col):
    out = []

    p1 = os.path.join(temp_dir, "model_comparison_rmse.png")
    plt.figure(figsize=(8, 5))
    plt.bar(model_results_df["Model"], model_results_df["LOOCV_RMSE_pIC50"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("LOOCV RMSE (pIC50)")
    plt.title("QSAR model comparison")
    plt.tight_layout()
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    out.append(p1)

    p2 = os.path.join(temp_dir, "observed_vs_fitted_best_model.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(train_df["pIC50"], train_df["Fitted_pIC50"])
    lims = [
        min(train_df["pIC50"].min(), train_df["Fitted_pIC50"].min()) - 0.2,
        max(train_df["pIC50"].max(), train_df["Fitted_pIC50"].max()) + 0.2,
    ]
    plt.plot(lims, lims, "--")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Observed pIC50")
    plt.ylabel("Fitted pIC50")
    plt.title(f"Observed vs fitted ({best_model_name})")
    plt.tight_layout()
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    out.append(p2)

    p3 = os.path.join(temp_dir, "top_predicted_gcms_compounds.png")
    top_n = min(20, len(pred_df))
    plt.figure(figsize=(10, 6))
    plt.barh(pred_df.loc[:top_n - 1, predict_name_col][::-1], pred_df.loc[:top_n - 1, "Predicted_pIC50"][::-1])
    plt.xlabel("Predicted pIC50")
    plt.title("Top predicted compounds")
    plt.tight_layout()
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    out.append(p3)

    return out
