from . import *


def classify_data(
    data,
    target_col="_target",
    cv=10,
    verbose=True,
    balance=False,
    normalize=NORMALIZE_CLASSIFY_DATA,
    sample_size=None,
    **kwargs,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import cross_val_predict
    import numpy as np

    # Ensure data is clean (fill NaNs)
    _data = data.copy()
    if not target_col in _data.columns:
        _data[target_col] = _data.index.str.split("/").str[0]

    if balance:
        min_target_size = min(_data[target_col].value_counts())
        sample_size = (
            min_target_size
            if sample_size is None or sample_size > min_target_size
            else sample_size
        )
        _data = _data.groupby(target_col).sample(n=sample_size)
        if verbose:
            print(f"Balanced data: {sample_size} samples per target")

    df_data = _data.drop(columns=[target_col])
    for c in df_data:
        df_data[c] = pd.to_numeric(df_data[c], errors="coerce")
    X_data_norm = df_data.fillna(0).values
    y_data = _data[target_col].fillna("").values

    # Mean feature values per target (for interpreting weights)
    # Use the same numeric + fillna(0) treatment as the classifier input.
    df_means_by_target = (
        df_data.fillna(0)
        .assign(_target=_data[target_col].fillna("").values)
        .groupby("_target")
        .mean(numeric_only=True)
    )

    # Initialize Logistic Regression
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.features_ = df_data.columns.tolist()

    if verbose:
        print(f"Running {cv}-fold Cross-Validation on {len(X_data_norm)} samples...")

    # Get predictions and probabilities for all items in the balanced set
    y_pred = cross_val_predict(model, X_data_norm, y_data, cv=cv, n_jobs=1)
    y_probas = cross_val_predict(
        model, X_data_norm, y_data, cv=cv, n_jobs=1, method="predict_proba"
    )

    # Confidence is the maximum probability across classes
    confidence_scores = np.max(y_probas, axis=1)

    accuracy = accuracy_score(y_data, y_pred)
    if verbose:
        print(f"\nClassifier Results ({cv}-fold CV):")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_data, y_pred))

    # Fit on all data to get final feature weights
    model.fit(X_data_norm, y_data)
    feature_names = _data.drop(columns=[target_col]).columns

    if len(model.classes_) <= 2:
        # Binary case: coef_ is (1, n_features)
        weights_df = pd.DataFrame(
            {"feature": feature_names, "weight": model.coef_[0]}
        ).sort_values("weight", ascending=False)
        # Add mean_{class} columns for the two classes
        cls1, cls2 = model.classes_[:2]
        means1 = df_means_by_target.loc[cls1] if cls1 in df_means_by_target.index else None
        means2 = df_means_by_target.loc[cls2] if cls2 in df_means_by_target.index else None
        if means1 is not None:
            weights_df[f"mean_{cls1}"] = weights_df["feature"].map(means1.to_dict()).fillna(0.0)
        if means2 is not None:
            weights_df[f"mean_{cls2}"] = weights_df["feature"].map(means2.to_dict()).fillna(0.0)
    else:
        # Multi-class case: coef_ is (n_classes, n_features)
        weights_df = pd.DataFrame(
            model.coef_.T, columns=model.classes_, index=feature_names
        )
        weights_df.index.name = "feature"
        weights_df = weights_df.reset_index()
        # Add mean_{class} columns for each class
        for cls in model.classes_:
            if cls in df_means_by_target.index:
                means = df_means_by_target.loc[cls].to_dict()
                weights_df[f"mean_{cls}"] = weights_df["feature"].map(means).fillna(0.0)

    # Return a DataFrame of relevant information
    test_label = " / ".join(model.classes_)
    prob_name1,prob_name2 = model.classes_[:2]
    results_df = pd.DataFrame(
        {
            "id": _data.index,
            "true_label": y_data,
            "pred_label": y_pred,
            f"prob_{prob_name1}": y_probas[:, 0],
            f"prob_{prob_name2}": y_probas[:, 1],
            "test_label": test_label,
            "confidence": confidence_scores,
            "correct": (y_pred == y_data),
            "accuracy": accuracy,
            "support": _data.shape[0],
        }
    )
    results_df.set_index("id", inplace=True)
    return results_df, weights_df, model



def get_df_feats_with_pos_mdw(df_feats, groups, **kwargs):
    from .constants import BAD_SLICE_FEATS
    from .statistics import get_mdw_pos

    df_mdw = get_mdw_pos(groups, **kwargs).reset_index()
    df_mdw["feat"] = [
        "deprel_" + x if x == x.lower() else "pos_" + x for x in df_mdw.feat
    ]
    df_mdw["feat_type"] = [x.split("_")[0] for x in df_mdw.feat]
    df_mdw = df_mdw[~df_mdw.feat.isin(BAD_SLICE_FEATS)]
    df_mdw = df_mdw.set_index("feat").fillna(0).round(1)
    odf = df_feats.merge(df_mdw, left_on="feature", right_on="feat", how="left")
    odf.sort_values("weight", ascending=False, inplace=True)
    odf["feat1_rank"] = [i + 1 for i in range(len(odf))]
    odf.sort_values("weight", ascending=True, inplace=True)
    odf["feat2_rank"] = [i + 1 for i in range(len(odf))]
    odf["weight_abs"] = np.abs(odf["weight"])
    odf.sort_values("weight_abs", ascending=False, inplace=True)
    odf["feat_rank"] = [i + 1 for i in range(len(odf))]
    odf.drop(columns=["weight_abs"], inplace=True)
    return odf.sort_values("weight", ascending=False)


def classify_then_predict_group(
    groups_train,
    target_col="discipline",
    balance=True,
    num_runs=1,
    verbose=False,
    return_models=False,
    normalize=NORMALIZE_CLASSIFY_DATA,
    also_predict_unseen=True,
    **kwargs,
):
    from .features import get_balanced_cv_data, get_mdw_feats

    l_preds = []
    l_feats = []
    l_models = []
    for nrun in tqdm(list(range(num_runs))):
        df_scores = get_balanced_cv_data(
            groups_train, target_col=target_col, balance=balance, normalize=normalize, **kwargs
        )
        hdrs = [c for c in df_scores.columns if c and c != "_target" and c[0] == "_"]
        df_scores_target = df_scores.query('_type=="CV"').drop(columns=hdrs)

        cv_preds, cv_feats, cv_model = classify_data(
            df_scores_target,
            target_col="_target",
            verbose=verbose,
            balance=True,
            **kwargs,
        )

        if also_predict_unseen:
            df_scores_unseen = df_scores.query('_type=="Unseen"').drop(columns=hdrs)


            new_target = df_scores_unseen._target.tolist()
            new_probs = cv_model.predict_proba(df_scores_unseen.drop(columns=["_target"]))
            df_new_probs = pd.DataFrame(new_probs)
            df_new_probs.columns = [f'prob_{x}' for x in cv_model.classes_]
            # df_new_probs["pred_label"] = df_new_probs.idxmax(axis=1)[:5] # max prob class
            # df_new_probs["true_label"] = new_target
            # df_new_probs["correct"] = (
                # df_new_probs["pred_label"] == df_new_probs["true_label"]
            # ).apply(int)
            # df_new_probs["test_label"] = " / ".join(cv_model.classes_)
            df_new_probs['support'] = len(df_scores_target)
            df_new_probs["id"] = df_scores_unseen.index
            df_new_probs.set_index("id", inplace=True)
            # df_new_probs

            df_out_probs = pd.concat(
                [
                    cv_preds.assign(run=nrun, predict_type="cv"),
                    df_new_probs.assign(run=nrun, predict_type="unseen"),
                ]
            )
        else:
            df_out_probs = cv_preds.assign(run=nrun, predict_type="cv")
        l_preds.append(df_out_probs)
        l_feats.append(cv_feats.assign(run=nrun))
        l_models.append(cv_model)
    df_preds = pd.concat(l_preds)
    df_feats = pd.concat(l_feats)

    df_feats_cols = [
        x
        for x in [
            "feature",
            "feat_desc",
            "comparison",
            # 'group1',
            # 'group2',
        ]
        if x in df_feats.columns
    ]

    df_feats = df_feats.groupby(df_feats_cols).mean(numeric_only=True).reset_index()
    # df_mdw = get_mdw_feats(groups_train, **kwargs)
    # df_feats = df_feats.merge(df_mdw, on="feature", how="left")
    # df_feats['group1'],df_feats['group2'] = zip(*df_feats['comparison'].str.split(' vs '))
    return (df_preds, df_feats) if not return_models else (df_preds, df_feats, l_models)


@STASH_CUSTOM_PREDS.stashed_result
def classify_custom_comparison(
    group1_name: str,
    group1_query: str,
    group2_name: str,
    group2_query: str,
    num_runs: int = 10,
    sample_size: int = 1000,
    balance: bool = True,
    replace: bool = False,
    cv: int = 10,
    normalize: bool = True,
):
    """
    Stashed wrapper for classify_then_predict_group.
    Uses string parameters for reliable cache keys.
    Returns (df_preds, df_feats) as dictionaries for JSON serialization.
    """
    groups_train = [
        (group1_name, group1_query),
        (group2_name, group2_query),
    ]
    
    df_preds, df_feats = classify_then_predict_group(
        groups_train,
        target_col='discipline',
        balance=balance,
        num_runs=num_runs,
        verbose=True,
        return_models=False,
        normalize=normalize,
        sample_size=sample_size,
        cv=cv,
        replace=replace,
    )
    
    # Convert to dict for JSON serialization in stash
    return {
        'preds': df_preds.reset_index().to_dict(orient='records'),
        'preds_index': df_preds.index.name or 'index',
        'feats': df_feats.to_dict(orient='records'),
    }


def load_custom_comparison_results(result_dict):
    """Convert stashed result dict back to DataFrames."""
    df_preds = pd.DataFrame(result_dict['preds'])
    if result_dict['preds_index'] and result_dict['preds_index'] in df_preds.columns:
        df_preds = df_preds.set_index(result_dict['preds_index'])
    elif 'id' in df_preds.columns:
        df_preds = df_preds.set_index('id')
    
    df_feats = pd.DataFrame(result_dict['feats'])
    return df_preds, df_feats


def classify_then_predict_comparisons(
    comparisons,
    return_models=False,
    normalize=NORMALIZE_CLASSIFY_DATA,
    **kwargs,
):
    l_preds = []
    l_feats = []
    d_models = {}
    for groups_train in comparisons:
        comparison_name = f"{groups_train[0][0]} vs {groups_train[1][0]}"
        print("##", comparison_name)
        df_preds, df_feats, models = classify_then_predict_group(groups_train, return_models=True, normalize=normalize, **kwargs)
        l_preds.append(df_preds.assign(comparison=comparison_name))
        l_feats.append(df_feats.assign(comparison=comparison_name))
        d_models[comparison_name] = models
    odf_preds, odf_feats = pd.concat(l_preds), pd.concat(l_feats)
    # odf_feats["group1"] = [x.split(" vs ")[0] for x in odf_feats["comparison"]]
    # odf_feats["group2"] = [x.split(" vs ")[1] for x in odf_feats["comparison"]]

    # print(odf_feats.columns)
    # odf_feats["score_mean_diff"] = odf_feats["score_mean1"] - odf_feats["score_mean2"]
    # odf_feats["score_mean_diff_abs"] = np.abs(odf_feats["score_mean_diff"])
    # odf_feats["score_mean_diff_pct"] = (
    #     odf_feats["score_mean_diff"] / odf_feats["score_mean2"]
    # )
    # odf_feats["score_mean_div"] = odf_feats["score_mean1"] / odf_feats["score_mean2"]
    # odf_feats["score_mean_div_abs"] = np.abs(odf_feats["score_mean_div"])
    # odf_feats["score_z_diff"] = odf_feats["score_z1"] - odf_feats["score_z2"]
    # odf_feats["score_z_diff_abs"] = np.abs(odf_feats["score_z_diff"])
    # odf_feats["score_z_diff_pct"] = odf_feats["score_z_diff"] / odf_feats["score_z2"]
    # odf_feats["score_z_div"] = odf_feats["score_z1"] / odf_feats["score_z2"]
    # odf_feats["score_z_div_abs"] = np.abs(odf_feats["score_z_div"])

    # odf_feats["feat_name"] = [x.split("_", 1)[-1] for x in odf_feats.feature]
    # odf_feats["feat_type"] = [x.split("_")[0] for x in odf_feats.feature]
    # odf_feats.sort_values("weight", ascending=False, inplace=True)
    return (odf_preds, odf_feats) if not return_models else (odf_preds, odf_feats, d_models)


# @cache
@STASH_PREDS_FEATS.stashed_result
def get_preds_feats(
    comparisons=COMPARISONS,
    num_runs=10,
    sample_size=1_000,
    feat_n=25,
    feat_n_egs=10,
    verbose=False,
    return_models=True,
    normalize=NORMALIZE_CLASSIFY_DATA,
    **kwargs,
):
    return classify_then_predict_comparisons(
        comparisons,
        num_runs=num_runs,
        sample_size=sample_size,
        feat_n=feat_n,
        feat_n_egs=feat_n_egs,
        verbose=verbose,
        return_models=return_models,
        normalize=normalize,
        **kwargs,
    )
















def get_new_preds_feats(txt, slice_id=None):
    doc = get_nlp_doc(txt) if isinstance(txt, str) else txt

    df_preds, df_feats, d_models = get_preds_feats()
    feat_names = get_feat_names_from_models(d_models)

    df_all_feats = get_all_feats(normalize=True)
    df_all_feats_raw = get_all_feats(normalize=False)
    df_all_feats_raw_means = df_all_feats_raw.mean()
    df_all_feats_raw_stds = df_all_feats_raw.std()

    # get feats
    if slice_id and slice_id in df_all_feats_raw.index:
        new_feats = df_all_feats_raw.loc[slice_id].to_dict()
        new_feats_z = df_all_feats.loc[slice_id].to_dict()
    else:
        new_feats = extract_slice_feats(doc)
        new_feats = {fname:new_feats.get(fname, 0) for fname in feat_names}
        new_feats_z = {
            k: float((v - df_all_feats_raw_means[k]) / df_all_feats_raw_stds[k])
            for k, v in new_feats.items()
            if k in df_all_feats_raw_means
        }
    
    new_feats_df = pd.DataFrame([new_feats])
    if slice_id and slice_id in df_preds.index:
        df_preds_new = df_preds.loc[slice_id].copy()
        df_preds_new['predict_type'] = 'stashed'
    else:
        # get preds
        ld_preds_new = []
        for cmpname, models in d_models.items():
            for nrun, mdl in enumerate(models):
                # Ensure new_feats_df has columns in same order as mdl.features_
                X = new_feats_df[mdl.features_].values
                new_prob1,new_prob2 = mdl.predict_proba(X)[0]
                new_name1,new_name2 = mdl.classes_
                new_pred = mdl.predict(X)[0]
                d_preds_new = {
                    'comparison': cmpname,
                    'run':nrun,
                    'predict_type': 'new',
                    'test_label': ' / '.join(mdl.classes_),
                    'true_label': '',
                    'pred_label': new_pred,
                    f'prob_{new_name1}': new_prob1,
                    f'prob_{new_name2}': new_prob2,
                }
                ld_preds_new.append(d_preds_new)
        df_preds_new = pd.DataFrame(ld_preds_new)

    # reintegrate feats into df_feats
    ld_feats_new = []
    for cmpname, cdf in df_feats.groupby('comparison'):
        for feat in feat_names:
            matches = df_feats.query('feature==@feat & comparison==@cmpname')
            if matches.empty: continue
            d_feats_new = dict(matches.iloc[0])
            d_feats_new['score_mean3'] = new_feats.get(feat, 0)
            d_feats_new['score_z3'] = new_feats_z.get(feat, 0)
            ld_feats_new.append(d_feats_new)

    df_feats_new = pd.DataFrame(ld_feats_new)
    return df_preds_new, df_feats_new

def get_feat_names_from_models(d_models):
    for cmpname, models in d_models.items():
        for mdl in models:
            return mdl.features_





def get_pred_label(row):
    prob_fields = [c for c in row.keys() if c.startswith('prob_')]
    pred_label = None
    for c in prob_fields:
        if row[c] > 0.5:
            pred_label = c.split('_',1)[-1]
            break
    return pred_label

def get_nice_df_preds(df_preds = None, metadata_cols = DF_PREDS_METADATA_COLS, average_by=DF_PREDS_AVERAGE_BY):
    if df_preds is None:
        df_preds, df_feats, d_models = get_preds_feats()
    odf_preds=(
        df_preds.drop(columns=['run','correct'])
        .query('predict_type=="unseen"')
        .groupby(['id','true_label','comparison'])
        .mean(numeric_only=True)
    ).reset_index()    
    odf_preds['text_id'] = [i.split('__')[0] for i in odf_preds.id]

    mdf = get_corpus_metadata().rename_axis('text_id').rename_axis('text_id')
    odf = odf_preds.merge(mdf,on='text_id',how='left')
    odf = odf.groupby(['true_label'] +average_by).mean(numeric_only=True).reset_index()
    odf['prob_Phil-Lit'] = odf['prob_Philosophy'] - odf['prob_Literature']
    # odf['pred_label'] = odf.apply(get_pred_label, axis=1)
    # odf['prob_correct'] = (odf.prob_pred == odf.true_label).apply(int)

    def get_accuracy_score(row):
        return row[f'prob_{row["true_label"]}']

    odf['prob_accuracy'] = odf.apply(get_accuracy_score, axis=1)

    outcols = average_by + [c for c in odf if c.startswith('prob_')]
    return odf[outcols]


@STASH_DF_PREDS_FOR_SLICES.stashed_result
def get_df_preds_for_slices(df_preds = None):
    if df_preds is None:
        df_preds, df_feats, d_models = get_preds_feats()
    inp_df = df_preds.query('predict_type=="unseen"').groupby(['comparison','id']).mean(numeric_only=True).reset_index()
    
    out_ld = []
    for idx,id_df in tqdm(inp_df.groupby('id'), total=inp_df.id.nunique()):
        out_d = {'id':idx}
        vals = []
        out_d2 = {}
        for cmp,cmp_df in id_df.groupby('comparison'):
            cmp_prd = cmp.split(' ')[0].split('-')[0]
            cmp_key = f'P{cmp_prd}'
            val = float(cmp_df.prob_Philosophy.mean())
            vals.append(val)
            out_d2[cmp_key] = val
        out_d['P'] = np.mean(vals)
        out_ld.append({**out_d, **out_d2})
    out_df = pd.DataFrame(out_ld)
    # out_df['Phil (2000-2025) - Phil (1925-1950)'] = out_df['Phil (2000-2025)'] - out_df['Phil (1925-1950)']
    # out_df = out_df.dropna().sort_values('Phil (2000-2025) - Phil (1925-1950)',ascending=False)
    out_df = out_df.set_index('id')
    return out_df

def get_nice_df_preds2(df_preds = None, metadata_cols = DF_PREDS_METADATA_COLS, by='text', incl_slice_ids=False, sort_by='n'):
    df = get_df_preds_for_slices(df_preds=df_preds)
    for c in df: 
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.reset_index()
    df['text_id'] = [i.split('__')[0] for i in df.id]
    mdf = get_corpus_metadata()[metadata_cols]
    mdf['text_id'] = mdf.index
    odf = df.merge(mdf,on='text_id',how='left')
    odf['year'] = odf.year.astype(str)

    groupby_cols = []
    if by=='text':
        groupby_cols = ['discipline', 'author', 'title', 'journal', 'year']
    elif by=='discipline':
        groupby_cols = ['discipline']
    elif isinstance(by, str):
        groupby_cols = ['discipline', by]
    elif isinstance(by, list):
        groupby_cols = ['discipline'] + [x for x in by if x!='discipline']
    
    
    if groupby_cols:
        numbcols = odf.select_dtypes(include='number').columns
        newld = []
        for g,gdf in odf.groupby(groupby_cols):
            newd = dict(gdf.mean(numeric_only=True))
            metad = dict(zip(groupby_cols, g))
            metad['n'] = len(gdf)
            metad['slice_ids'] = '; '.join(gdf.id.astype(str))
            newld.append({**metad, **newd})
        odf = pd.DataFrame(newld)
    
    if 'year' in odf.columns:
        odf['year'] = odf.year.astype(int)

    a,b='P1900','P2000'
    diffcol = f'{b}-{a}'
    if a in odf.columns and b in odf.columns:
        odf[diffcol] = odf[b] - odf[a]
    lower_cols = [c for c in odf if c and c[0]==c[0].lower()]
    upper_cols = [c for c in odf if c and c[0]==c[0].upper()]
    odf = odf[lower_cols + upper_cols].fillna(0)

    if sort_by in odf.columns:
        odf = odf.sort_values(sort_by,ascending=False)
    if not incl_slice_ids and 'slice_ids' in odf.columns:
        odf = odf.drop(columns=['slice_ids'])
    odf = odf.set_index(groupby_cols) if groupby_cols else (odf.set_index('id') if 'id' in odf.columns else odf)
    odf = odf.rename(
        columns={
            'P': 'P(Phil)',
            'P1900': 'P(Phil|1900)',
            'P1925': 'P(Phil|1925)',
            'P1950': 'P(Phil|1950)',
            'P1975': 'P(Phil|1975)',
            'P2000': 'P(Phil|2000)',
            # 'P2000/P1900': 'ΔP(Phil|1900→2000)',
            'P2000-P1900': 'ΔP(Phil|1900→2000)',
        }
    )
    return odf


def get_df_preds(*x,**y):
    out = get_preds_feats(*x,**y)
    return out[0]

def get_df_feats(*x,**y):
    out = get_preds_feats(*x,**y)
    return out[1]

def get_d_models(*x,**y):
    y['return_models'] = True
    out = get_preds_feats(*x,**y)
    return out[2]

@cache
def get_current_pred_probs(target_col='discipline'):
    df_preds = get_df_preds()
    num_runs = df_preds['run'].nunique()
    odf = df_preds.groupby(['predict_type','comparison','id']).mean(numeric_only=True).reset_index().drop(columns=['run'])
    odf['target'] = odf['id'].map(lambda x: get_text_metadata(x).get(target_col,''))
    return odf.sort_values('prob_Philosophy',ascending=False).set_index('id').assign(num_runs=num_runs)