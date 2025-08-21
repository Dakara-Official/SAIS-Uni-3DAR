import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def reduce_bi_classification(predicts, targets, key):
    mol_key = key
    df = pd.DataFrame(
        {
            "predict": predicts,
            "targets": targets.reshape(-1),
            "mol_key": mol_key.reshape(-1),
        }
    )
    auc = roc_auc_score(df["targets"], df["predict"])
    df = df.groupby("mol_key").mean()
    agg_auc = roc_auc_score(df["targets"], df["predict"])
    return {"auc": auc, "agg_auc": agg_auc}


def reduce_regression(predicts, targets, key):
    mol_key = key
    df = pd.DataFrame(
        {
            "predict": predicts.reshape(-1),
            "target": targets.reshape(-1),
            "mol_key": mol_key.reshape(-1),
        }
    )
    mae = np.abs(df["predict"] - df["target"]).mean()
    mse = ((df["predict"] - df["target"]) ** 2).mean()
    df = df.groupby("mol_key").mean()
    agg_mae = np.abs(df["predict"] - df["target"]).mean()
    agg_mse = ((df["predict"] - df["target"]) ** 2).mean()
    return {
        "mae": mae,
        "mse": mse,
        "agg_mae": agg_mae,
        "agg_mse": agg_mse,
        "agg_rmse": np.sqrt(agg_mse),
    }


def reduce_regression_atom(predicts, targets, keys):
    mol_key, atom_key = keys
    predicts = predicts.tolist()
    targets = targets.tolist()

    def aggregate_repeat(df):
        predict = np.mean(np.stack(df["predict"].values), axis=0)  # repeat, 3 - > 3
        target = np.mean(np.stack(df["target"].values), axis=0)
        return pd.Series({"predict": predict, "target": target})

    def aggregate_mae(df):
        mae = np.abs(df["predict"] - df["target"]).mean().sum()
        mse = ((df["predict"] - df["target"]) ** 2).mean().sum()
        return pd.Series({"mae": mae, "mse": mse})

    df = pd.DataFrame(
        {
            "predict": predicts,
            "target": targets,
            "mol_key": mol_key,
            "atom_key": atom_key,
        }
    )
    grouped = df.groupby(["mol_key", "atom_key"]).apply(aggregate_repeat).reset_index()
    grouped = grouped.groupby(["mol_key"]).apply(aggregate_mae).reset_index()
    agg_mae = grouped["mae"].mean()
    agg_mse = grouped["mse"].mean()

    return {
        "agg_mae": agg_mae,
        "agg_mse": agg_mse,
        "agg_rmse": np.sqrt(agg_mse),
    }


def reduce_bi_classification_atom(predicts, targets, keys):
    mol_key, atom_key = keys

    def auc_group(df):
        predict = df.predict
        target = df.target
        assert ((target == 0) | (target == 1)).all()
        target = target.astype(int)
        try:
            auc = roc_auc_score(target, predict)
        except ValueError:
            auc = np.nan
        predict = (predict > 0.5).astype(float)
        union = ((target + predict) > 0).sum()
        intersection = (target * predict > 0).sum()
        if union == 0:
            iou = -1
        else:
            iou = intersection / union
        return (auc, iou)

    df = pd.DataFrame(
        {
            "predict": predicts,
            "target": targets,
            "mol_key": mol_key,
            "atom_key": atom_key,
        }
    )
    grouped = df.groupby(["mol_key", "atom_key"]).mean()
    grouped_res = grouped.groupby(["mol_key"]).apply(auc_group)
    auc = [item[0] for item in grouped_res if not np.isnan(item[0])]
    iou = [item[1] for item in grouped_res if item[1] >= 0]
    auc_count = len(auc)
    auc = sum(auc) / auc_count
    iou_count = len(iou)
    iou = sum(iou) / iou_count
    return {
        "agg_iou": iou,
        "iou_count": iou_count,
        "agg_auc": auc,
        "auc_count": auc_count,
    }
