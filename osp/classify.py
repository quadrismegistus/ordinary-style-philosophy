from . import *


def classify_data(
    data,
    target_col="_target",
    cv=10,
    verbose=True,
    balance=False,
    normalize=NORMALIZE_DATA,
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
    else:
        # Multi-class case: coef_ is (n_classes, n_features)
        weights_df = pd.DataFrame(
            model.coef_.T, columns=model.classes_, index=feature_names
        )
        weights_df.index.name = "feature"
        weights_df = weights_df.reset_index()

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


def iter_pairwise_samples(
    data, colname="period", data_meta=None, balance=True, sample_size=None
):
    from .data_loaders import get_corpus_metadata

    data_meta = get_corpus_metadata() if data_meta is None else data_meta
    min_grp_size = min(data_meta.groupby(colname).size())
    data_meta = data_meta.groupby(colname).sample(n=min_grp_size)
    data_smpl = data.sample(frac=1)

    coltypes = data_meta[colname].unique()
    for coltype1 in coltypes:
        for coltype2 in coltypes:
            if coltype1 >= coltype2:
                continue
            df_meta_yes = data_meta.query(f"{colname}==@coltype1")
            df_meta_no = data_meta.query(f"{colname}==@coltype2")
            index_name = "id"
            data_smpl2 = data_smpl.reset_index()
            ok_yes = list(df_meta_yes.index)
            ok_no = list(df_meta_no.index)

            data_smpl_yes = data_smpl2.query(f"{index_name} in @ok_yes").assign(
                _target=coltype1
            )
            data_smpl_no = data_smpl2.query(f"{index_name} in @ok_no").assign(
                _target=coltype2
            )
            data_smpl_yes = data_smpl_yes.set_index(index_name)
            data_smpl_no = data_smpl_no.set_index(index_name)

            if balance:
                if sample_size is None:
                    minsize = min(data_smpl_yes.shape[0], data_smpl_no.shape[0])
                    data_smpl_yes = data_smpl_yes.sample(n=minsize)
                    data_smpl_no = data_smpl_no.sample(n=minsize)
                else:
                    data_smpl_yes = data_smpl_yes.sample(n=sample_size, replace=True)
                    data_smpl_no = data_smpl_no.sample(n=sample_size, replace=True)
            yield (coltype1, coltype2), pd.concat([data_smpl_yes, data_smpl_no]).sample(
                frac=1
            )


def classify_pairwise_samples(
    data, colname="period", data_meta=None, balance=True, sample_size=None, verbose=True
):
    iterr = iter_pairwise_samples(
        data,
        colname=colname,
        data_meta=data_meta,
        balance=balance,
        sample_size=sample_size,
    )

    out_preds = []
    out_feats = []
    for g, gfx in tqdm(list(iterr), desc=f"classifying pairwise samples for {colname}"):
        meta = {
            "test_group": " vs. ".join(g) if isinstance(g, (list, tuple)) else g,
            "test_group_type": colname,
        }
        gdf_preds, gdf_feats = classify_data(gfx, verbose=verbose)
        out_preds.append(gdf_preds.assign(**meta))
        out_feats.append(gdf_feats.assign(**meta))
    odf_preds, odf_feats = pd.concat(out_preds), pd.concat(out_feats)
    odf_preds.sort_values("accuracy", ascending=False, inplace=True)
    odf_feats.sort_values("weight", ascending=False, inplace=True)
    return odf_preds, odf_feats


def classify_all_data(
    wordset2data,
    classify_by=["discipline"],
    data_meta=None,
    sample_size=2500,
    verbose=False,
):
    all_df_preds = []
    all_df_feats = []

    for wordset, data in wordset2data.items():
        for classify_by_x in classify_by:
            df_preds, df_feats = classify_pairwise_samples(
                data,
                classify_by_x,
                data_meta=data_meta,
                sample_size=sample_size,
                verbose=verbose,
            )
            all_df_preds.append(
                df_preds.assign(wordset=wordset, classify_by=classify_by_x)
            )
            all_df_feats.append(
                df_feats.assign(wordset=wordset, classify_by=classify_by_x)
            )

    all_df_preds = pd.concat(all_df_preds) if all_df_preds else pd.DataFrame()
    all_df_feats = pd.concat(all_df_feats) if all_df_feats else pd.DataFrame()

    return all_df_preds, all_df_feats


def classify_by_feat_counts(
    groups,
    predict_all=False,
    num_runs=1,
    sample_size=None,
    incl_deprel=True,
    incl_pos=True,
    cv=10,
    feat_n=10,
    feat_min_count=1,
    avg_runs=True,
    balance=True,
    **kwargs,
):
    from .constants import CLASSIFY_BY_FEAT_SAMPLE_SIZE
    from .features import get_feat_counts, get_all_feats

    if sample_size is None:
        sample_size = CLASSIFY_BY_FEAT_SAMPLE_SIZE

    name1, ids1 = groups[0]
    name2, ids2 = groups[1]
    df_pos_grp1 = get_feat_counts(ids1)
    df_pos_grp2 = get_feat_counts(ids2)

    # Get all features for prediction if requested
    df_predict_all = get_all_feats(normalize=True) if predict_all else None

    min_grp_size = min(len(ids1), len(ids2))
    if sample_size is None:
        sample_size = min_grp_size
    print(f"{name1} {len(ids1)} / {name2} {len(ids2)}")
    l_preds = []
    l_feats = []
    l_predicts = []
    iterr = tqdm(list(range(num_runs)))
    for nrun in iterr:
        dfx1 = df_pos_grp1.sample(sample_size, replace=True) if balance else df_pos_grp1
        dfx2 = df_pos_grp2.sample(sample_size, replace=True) if balance else df_pos_grp2
        df_pos = pd.concat([dfx1, dfx2])
        df_pos["_target"] = [name1] * len(dfx1) + [name2] * len(dfx2)
        iterr.set_description(f"{name1} {len(dfx1)} / {name2} {len(dfx2)}")

        # Call classify_data with optional predict_df
        res = classify_data(
            df_pos,
            predict_df=df_predict_all,
            target_col="_target",
            cv=cv,
            balance=balance,
            **kwargs,
        )
        df_preds, df_feats = res[0], res[1]

        l_preds.append(df_preds.assign(run=nrun))
        l_feats.append(df_feats.assign(run=nrun))

    if not len(l_preds) or not len(l_feats):
        return None, None

    odf_preds = pd.concat(l_preds)
    odf_feats = pd.concat(l_feats)

    # Return predictions if they were collected
    if l_predicts:
        odf_predicts = pd.concat(l_predicts)
        odf_preds = pd.concat(
            [
                odf_preds.assign(predict_type="cv"),
                odf_predicts.assign(predict_type="unseen"),
            ]
        )

    return odf_preds, get_df_feats_with_pos_mdw(odf_feats, groups, **kwargs)


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
    normalize=NORMALIZE_DATA,
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
        df_scores_unseen = df_scores.query('_type=="Unseen"').drop(columns=hdrs)

        cv_preds, cv_feats, cv_model = classify_data(
            df_scores_target,
            target_col="_target",
            verbose=verbose,
            balance=True,
            **kwargs,
        )

        new_target = df_scores_unseen._target.tolist()
        new_probs = cv_model.predict_proba(df_scores_unseen.drop(columns=["_target"]))
        df_new_probs = pd.DataFrame(new_probs)
        df_new_probs.columns = [f'prob_{x}' for x in cv_model.classes_]
        df_new_probs["pred_label"] = df_new_probs.idxmax(axis=1)[:5] # max prob class
        df_new_probs["true_label"] = new_target
        df_new_probs["correct"] = (
            df_new_probs["pred_label"] == df_new_probs["true_label"]
        ).apply(int)
        df_new_probs["test_label"] = " / ".join(cv_model.classes_)
        df_new_probs["id"] = df_scores_unseen.index
        df_new_probs.set_index("id", inplace=True)
        # df_new_probs

        df_out_probs = pd.concat(
            [
                cv_preds.assign(run=nrun, predict_type="cv"),
                df_new_probs.assign(run=nrun, predict_type="unseen"),
            ]
        )
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
    df_mdw = get_mdw_feats(groups_train, **kwargs)
    df_feats = df_feats.merge(df_mdw, on="feature", how="left")
    # df_feats['group1'],df_feats['group2'] = zip(*df_feats['comparison'].str.split(' vs '))
    return (df_preds, df_feats) if not return_models else (df_preds, df_feats, l_models)


def classify_then_predict_comparisons(
    comparisons,
    return_models=False,
    normalize=NORMALIZE_DATA,
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
    odf_feats["group1"] = [x.split(" vs ")[0] for x in odf_feats["comparison"]]
    odf_feats["group2"] = [x.split(" vs ")[1] for x in odf_feats["comparison"]]

    odf_feats["score_mean_diff"] = odf_feats["score_mean1"] - odf_feats["score_mean2"]
    odf_feats["score_mean_diff_abs"] = np.abs(odf_feats["score_mean_diff"])
    odf_feats["score_mean_diff_pct"] = (
        odf_feats["score_mean_diff"] / odf_feats["score_mean2"]
    )
    odf_feats["score_mean_div"] = odf_feats["score_mean1"] / odf_feats["score_mean2"]
    odf_feats["score_mean_div_abs"] = np.abs(odf_feats["score_mean_div"])
    odf_feats["score_z_diff"] = odf_feats["score_z1"] - odf_feats["score_z2"]
    odf_feats["score_z_diff_abs"] = np.abs(odf_feats["score_z_diff"])
    odf_feats["score_z_diff_pct"] = odf_feats["score_z_diff"] / odf_feats["score_z2"]
    odf_feats["score_z_div"] = odf_feats["score_z1"] / odf_feats["score_z2"]
    odf_feats["score_z_div_abs"] = np.abs(odf_feats["score_z_div"])

    odf_feats["feat_name"] = [x.split("_", 1)[-1] for x in odf_feats.feature]
    odf_feats["feat_type"] = [x.split("_")[0] for x in odf_feats.feature]
    odf_feats.sort_values("weight", ascending=False, inplace=True)
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
    normalize=NORMALIZE_DATA,
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

def get_new_preds_feats(txt):
    doc = get_nlp_doc(txt) if isinstance(txt, str) else txt

    df_preds, df_feats, d_models = get_preds_feats()
    feat_names = get_feat_names_from_models(d_models)

    df_all_feats = get_all_feats(normalize=True)
    df_all_feats_raw = get_all_feats(normalize=False)
    df_all_feats_raw_means = df_all_feats_raw.mean()
    df_all_feats_raw_stds = df_all_feats_raw.std()

    # get feats
    new_feats = extract_slice_feats(doc)
    new_feats = {fname:new_feats.get(fname, 0) for fname in feat_names}
    new_feats_z = {
        k: float((v - df_all_feats_raw_means[k]) / df_all_feats_raw_stds[k])
        for k, v in new_feats.items()
        if k in df_all_feats_raw_means
    }
    new_feats_df = pd.DataFrame([new_feats])
    new_feats_z_df = pd.DataFrame([new_feats_z])

    # reintegrate feats into df_feats
    ld_feats_new = []
    for cmpname, cdf in df_feats.groupby('comparison'):
        for feat in feat_names:
            d_feats_new = dict(df_feats.query('feature==@feat & comparison==@cmpname').iloc[0])
            d_feats_new['score_mean3'] = new_feats.get(feat, 0)
            d_feats_new['score_z3'] = new_feats_z.get(feat, 0)
            ld_feats_new.append(d_feats_new)

    df_feats_new = pd.DataFrame(ld_feats_new)
    df_feats_new['score_mean_diff_3-1'] = df_feats_new['score_mean3'] - df_feats_new['score_mean1']
    df_feats_new['score_mean_diff_3-2'] = df_feats_new['score_mean3'] - df_feats_new['score_mean2']

    df_feats_new['score_mean_div_3-1'] = df_feats_new['score_mean3'] / df_feats_new['score_mean1']
    df_feats_new['score_mean_div_3-2'] = df_feats_new['score_mean3'] / df_feats_new['score_mean2']

    df_feats_new['score_mean_diff_abs_3-1'] = df_feats_new['score_mean_diff_3-1'].abs()
    df_feats_new['score_mean_diff_abs_3-2'] = df_feats_new['score_mean_diff_3-2'].abs()

    df_feats_new['score_z_diff_3-1'] = df_feats_new['score_z3'] - df_feats_new['score_z1']
    df_feats_new['score_z_diff_3-2'] = df_feats_new['score_z3'] - df_feats_new['score_z2']

    df_feats_new['score_z_div_3-1'] = df_feats_new['score_z3'] / df_feats_new['score_z1']
    df_feats_new['score_z_div_3-2'] = df_feats_new['score_z3'] / df_feats_new['score_z2']

    df_feats_new['score_z_diff_abs_3-1'] = df_feats_new['score_z_diff_3-1'].abs()
    df_feats_new['score_z_diff_abs_3-2'] = df_feats_new['score_z_diff_3-2'].abs()

    # get preds
    ld_preds_new = []
    for cmpname, models in d_models.items():
        for nrun, mdl in enumerate(models):
            new_prob1,new_prob2 = mdl.predict_proba(new_feats_df.values)[0]
            new_name1,new_name2 = mdl.classes_
            new_pred = mdl.predict(new_feats_df.values)[0]
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


@HashStash('osp_df_preds_for_slices').stashed_result
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