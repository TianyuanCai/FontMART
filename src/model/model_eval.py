from __future__ import annotations

import itertools
import json
import os
import re
import warnings

import h2o
import joblib
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import neptunecontrib.monitoring.skopt as sk_utils
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import skopt
import statsmodels.api as sm
from absl import flags
from joblib import delayed
from joblib import Parallel
from kneed import KneeLocator
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from helper.data_build import make_name
from src.data.survey_data import Data

h2o.init()

np.random.seed(1)
warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set_style("ticks")
group_col = "toggle_id_user"
plt.rcParams.update({"font.size": 22})
maj_figsize = (12, 8)
plt.style.use("seaborn-ticks")
plt.close("all")

flags.DEFINE_string(
    "label_type", "nonormalize", "[nonormalize, normalize, graded]",
)
flags.DEFINE_string(
    "preference_type", "include", "[include, exclude, font_char, font_char_diff]",
)
flags.DEFINE_bool("retrain", True, "Whether to retrain model.")
flags.DEFINE_float(
    "wpm_threshold", 0.95, "Minimum WPM to be considered a fast font",
)
flags.DEFINE_string("objective", "performance", "[preference, performance]")
FLAGS = flags.FLAGS

SEARCH_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 15,
    "num_leaves": 32,
    "feature_fraction": 0.8,
    "subsample": 0.2,
    "num_boost_round": 300,
    "early_stopping_rounds": 30,
    "lambdarank_truncation_level": 5,
}

FIXED_PARAMS = {
    "objective": "lambdarank",
    "verbose": -1,
    "metric": ["ndcg", "map"],
    "boosting_type": "gbdt",
    "ndcg_at": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "device": "cpu",
}

# tuning booster
SPACE = [
    skopt.space.Real(1e-3, 0.5, name="learning_rate", prior="log-uniform"),
    skopt.space.Integer(5, 30, name="max_depth"),
    skopt.space.Integer(10, 50, name="num_leaves"),
    skopt.space.Real(0.1, 1.0, name="feature_fraction", prior="uniform"),
    skopt.space.Real(0.8, 1.0, name="subsample", prior="uniform"),
    skopt.space.Integer(100, 1000, name="num_boost_round"),
    skopt.space.Integer(10, 80, name="early_stopping_rounds"),
    skopt.space.Integer(6, 8, name="lambdarank_truncation_level"),
]

mpl.rcParams["axes.unicode_minus"] = False

df_wpm = pd.read_csv("data/processed/new_df_wpm.csv")


def feature_convert(col_title):

    col_title = col_title.replace("pre_reading_speed", "SR_reading_speed")
    col_title = col_title.replace(
        "pre_reading_frequency", "SR_reading_frequency",
    )
    col_title = col_title.replace("toggle_", "font: ")
    col_title = col_title.replace("pre_", "user: ")
    col_title = col_title.replace("rank_", "Rank: ")
    col_title = col_title.replace("upper", "Uppercase")
    col_title = col_title.replace("grey", "gray")
    col_title = col_title.replace("gray_scale", "grayscale")
    col_title = col_title.replace("lower", "Lowercase")
    col_title = make_name(col_title).replace("_", " ").title()

    # replace post title case
    col_title = col_title.replace("Sr ", "SR-")
    col_title = col_title.replace("Pref ", "UP-")
    col_title = col_title.replace("H ", "h-")
    col_title = col_title.replace("G ", "g-")
    col_title = col_title.replace("X ", "x-")
    col_title = col_title.replace("Font Rank", "Preference")
    col_title = col_title.replace("Fam Level", "Familiarity")
    col_title = col_title.replace(" Sd", " SD")
    col_title = col_title.replace(" Avg", " Average")
    col_title = col_title.replace("x-Height Relative", "Relative x-Height")
    col_title = col_title.replace("x-Width Relative", "Relative x-Width")
    return col_title


def get_train_test_lgb_data(df, train_idx, test_idx, n_subsample=100):
    """generate grouped lgbm data slices by train test idx"""
    tmp_train = df.iloc[train_idx, :]
    tmp_test = df.iloc[test_idx, :]

    def subsample_rank(i):
        """augment training input"""
        augment_frac = 0.8
        aug_tmp_train = tmp_train.groupby(group_col).sample(
            frac=augment_frac, replace=False, random_state=i,
        )
        aug_tmp_train["toggle_id_user"] = (
            aug_tmp_train["toggle_id_user"].astype(str) + "_" + str(i)
        )
        return aug_tmp_train

    # concatenate sampled training data
    aug_train_list = Parallel(n_jobs=-1)(
        delayed(subsample_rank)(i) for i in range(n_subsample)
    )
    tmp_train = pd.concat([tmp_train] + aug_train_list, axis=0)

    tmp_train.loc[:, group_col] = tmp_train.loc[:, group_col].astype(str)
    tmp_test.loc[:, group_col] = tmp_test.loc[:, group_col].astype(str)

    # remove training observations with no top fonts, i.e. no meaningful info
    tmp_top_font_df = (
        tmp_train.groupby("toggle_id_user")["outcome"]
        .sum()
        .reset_index(name="sum_n_top_fonts")
    )
    tmp_top_font_df = tmp_top_font_df.query("sum_n_top_fonts>=1")
    tmp_train = tmp_train[
        tmp_train["toggle_id_user"].isin(tmp_top_font_df["toggle_id_user"])
    ]

    # create group labels
    tmp_train = tmp_train.sort_values(group_col).reset_index(drop=True)
    tmp_test = tmp_test.sort_values(group_col).reset_index(drop=True)

    query_train = (
        tmp_train.groupby(group_col)[group_col]
        .count()
        .reset_index(name="group")
        .sort_values(group_col)["group"]
    )
    query_test = (
        tmp_test.groupby(group_col)[group_col]
        .count()
        .reset_index(name="group")
        .sort_values(group_col)["group"]
    )

    # lgbm train api
    tmp_train_lgb = lgb.Dataset(
        tmp_train.drop(["outcome", "toggle_font", "toggle_id_user"], axis=1),
        label=tmp_train["outcome"],
        free_raw_data=False,
        group=query_train.values,
    )
    tmp_test_lgb = lgb.Dataset(
        tmp_test.drop(["outcome", "toggle_font", "toggle_id_user"], axis=1),
        label=tmp_test["outcome"],
        free_raw_data=False,
        group=query_test.values,
    )

    return tmp_train_lgb, tmp_test_lgb


class Diagnostic:
    def __init__(self) -> None:
        self.df_wpm = pd.read_csv("data/processed/new_df_wpm.csv")
        pass

    def plot_n_top_fonts(self):
        # plt: number of top fonts by user
        plt.close("all")
        self.df_full.groupby("toggle_id_user")["outcome"].sum().hist()
        plt.title("Distribution of number of Top Fonts per User")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/input/"
            f"{self.objective}_n_top_fonts_distribution_user_level.png",
        )

    def plot_font_features(self):
        df_toggle = self.df_full.filter(
            regex="toggle_(?!(id_user|font_rank|fam_level))", axis=1,
        ).drop_duplicates()

        plt.close("all")
        g = sns.FacetGrid(
            df_toggle.melt("toggle_font"),
            col="variable",
            col_wrap=3,
            sharey=True,
            sharex=False,
        )
        g.map_dataframe(sns.barplot, x="value", y="toggle_font")

        axes = g.axes.flatten()
        for a in axes:
            previous_title = a.get_title()
            a.set_title(
                feature_convert(
                    previous_title.replace(
                        "variable = toggle_", "",
                    ),
                ),
            )

        g.set_axis_labels("Value", "Font")
        plt.tight_layout()
        plt.savefig(f"{self.report_dir}/input/font_chars.png")

    def calculate_baseline_metrics(self):
        if self.objective == "performance":
            """calculate baseline ndcg metrics"""
            original_ordered_outcome = (
                self.df_full.sort_values(["toggle_id_user", "toggle_font"])
                .groupby("toggle_id_user")["outcome"]
                .apply(list)
            )
            original_ordered_fonts = (
                self.df_full.sort_values(["toggle_id_user", "toggle_font"])
                .groupby("toggle_id_user")["toggle_font"]
                .apply(list)
            )

            df_preferred_full = self.df_preferred_wide

            baseline_score_dict = {
                "k": [],
                "ndcg_random": [],
                "ndcg_preference": [],
                "map_random": [],
                "map_preference": [],
            }

            # when calculating the baseline approach, we assume that the top 2
            # preferred fonts may both be "relevant" items, meaning the users
            # may perform well in their top 2 preferred font
            top_n_preferred = 2

            # for loop iterate through metrics at different rank positions
            for rank_k in range(1, 9):
                baseline_score_dict["k"].append(rank_k)

                ndcg_list = []
                ndcg_list_pref = []
                map_list = []
                map_list_pref = []

                # rank metrics calculated for each user
                for i in range(len(original_ordered_outcome)):
                    uid = original_ordered_outcome.index[i]

                    # get preference ranking and mark top k as relevant
                    preferred_ranking = df_preferred_full.query(
                        f"toggle_id_user=={uid}",
                    )
                    if len(preferred_ranking) == 0:
                        print("--- missing user in elo score table")
                        continue
                    preferred_ranking_long = preferred_ranking.melt(
                        "toggle_id_user",
                    )
                    preferred_ranking_long = preferred_ranking_long.sort_values(
                        "value", ascending=True,
                    ).reset_index(drop=True)
                    preferred_ranking_long["variable"] = preferred_ranking_long[
                        "variable"
                    ].str.replace("rank_", "")
                    preferred_ranking_long["relevancy"] = [
                        1 if i < top_n_preferred else 0
                        for i, k in enumerate(preferred_ranking_long["value"])
                    ]

                    preferred_ranking_long[["variable", "relevancy"]].to_dict()
                    # font_ranking_dict = {
                    #     row[1]: row[-1]
                    #     for row in preferred_ranking_long.values.tolist()
                    # }
                    font_ranking_dict_graded = {
                        row[1]: row[-2]
                        for row in preferred_ranking_long.values.tolist()
                    }

                    tmp_font_list = original_ordered_fonts.values[i]
                    if len(tmp_font_list) <= 1:
                        print("only 1 font remains in recommendation")
                        continue

                    # the negative sign is here so that the larger values
                    # are more preferred user fonts
                    tmp_font_preference_list_graded = [
                        -font_ranking_dict_graded[k] for k in tmp_font_list
                    ]

                    arr = np.array(original_ordered_outcome.values[i])

                    ndcg_list.append(
                        ndcg_score(
                            np.asarray([arr]).astype(int),
                            np.asarray(
                                [np.random.permutation(arr).astype(int)],
                            ),
                            k=rank_k,
                        ),
                    )
                    ndcg_list_pref.append(
                        ndcg_score(
                            np.asarray([arr]).astype(int),
                            np.asarray([tmp_font_preference_list_graded]),
                            k=rank_k,
                        ),
                    )

                    if self.label_type != "graded":
                        map_list.append(
                            label_ranking_average_precision_score(
                                np.asarray([arr]).tolist(),
                                np.asarray(
                                    [
                                        np.random.permutation(
                                            list(range(len(arr))),
                                        ),
                                    ],
                                ).tolist(),
                            ),
                        )

                        map_list_pref.append(
                            label_ranking_average_precision_score(
                                np.asarray([arr]).tolist(),
                                [tmp_font_preference_list_graded],
                            ),
                        )

                baseline_score_dict["ndcg_random"].append(np.mean(ndcg_list))
                baseline_score_dict["ndcg_preference"].append(
                    np.mean(ndcg_list_pref),
                )
                baseline_score_dict["map_random"].append(np.mean(map_list))
                baseline_score_dict["map_preference"].append(
                    np.mean(map_list_pref),
                )

            baseline_score_dict = pd.DataFrame(
                {k: v for k, v in baseline_score_dict.items() if len(v) > 0},
            )

            df_tmp = self.df_full.copy()

            params_file = f"{self.report_dir}/params_{self.objective}.joblib"
            best_search_params = joblib.load(params_file)

            group_kfold = GroupKFold(n_splits=self.n_splits)
            group_idx = df_tmp.groupby(group_col).ngroup()

            train_test_folds = list(
                group_kfold.split(
                    df_tmp.drop(
                        "outcome", axis=1,
                    ), df_tmp["outcome"], group_idx,
                ),
            )

            cv_metrics = []
            for f in tqdm(train_test_folds):
                tmp_train_lgb, tmp_test_lgb = get_train_test_lgb_data(
                    df_tmp, f[0], f[1], n_subsample=100,
                )
                self.gbm = lgb.train(
                    {**best_search_params, **FIXED_PARAMS},
                    tmp_train_lgb,
                    valid_sets=tmp_test_lgb,
                    verbose_eval=False,
                )

                test_X = self.df_full.iloc[
                    f[1],
                ]
                test_X["pred"] = self.gbm.predict(
                    test_X[tmp_test_lgb.data.columns],
                )

                cv_metrics.append(self.gbm.best_score["valid_0"])

            # finalized score dictionary
            cv_metrics = {k: [d[k] for d in cv_metrics] for k in cv_metrics[0]}
            cv_metrics_df = pd.DataFrame.from_dict(
                cv_metrics, orient="index",
            ).transpose()
            cv_metrics_df = cv_metrics_df.mean().reset_index()
            cv_metrics_df = (
                cv_metrics_df.assign(
                    metric=[s.split("@")[0] for s in cv_metrics_df["index"]],
                )
                .assign(k=[int(s.split("@")[1]) for s in cv_metrics_df["index"]])
                .drop("index", axis=1)
            )
            cv_metrics_df = pd.concat(
                [
                    cv_metrics_df.query(
                        'metric=="ndcg"',
                    ).reset_index(drop=True),
                    cv_metrics_df.query(
                        'metric=="map"',
                    ).reset_index(drop=True),
                ],
                axis=1,
            )
            cv_metrics_df.columns = [
                "ndcg_model",
                "metric",
                "k",
                "map_model",
                "metric2",
                "k2",
            ]
            baseline_score_dict = baseline_score_dict.merge(
                cv_metrics_df[["k", "ndcg_model", "map_model"]], on="k",
            )

        else:
            original_ordered_outcome = (
                self.df_full.sort_values("toggle_id_user")
                .groupby("toggle_id_user")["outcome"]
                .apply(list)
            )

            rank_k = 2
            ndcg_list = []
            for i in range(len(original_ordered_outcome)):
                arr = np.array(original_ordered_outcome.values[i])
                ndcg_list.append(
                    ndcg_score(
                        np.asarray([arr]),
                        np.asarray([np.random.permutation(arr)]),
                        k=rank_k,
                    ),
                )

            print(f"--- baseline ndcg at {rank_k} is", np.mean(ndcg_list))

    def plot_top1_proportion(self):
        """plot the proportion of the number of top fonts per user"""
        plt.close("all")
        self.df_full.groupby("toggle_id_user")["outcome"].sum().hist()
        plt.title("Number of Top Fonts per User")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/input/" f"{self.objective}_top1_font_proportion.png",
        )

    def calculate_prop_identified_top1(self):
        n_qual_users = (
            self.df_full.query("outcome==1 & toggle_font_rank==1")[
                "toggle_id_user"
            ]
            .drop_duplicates()
            .nunique()
        )
        print(
            "--- proportion of the users who identified one of their their "
            "top-5th percentile fonts with the preference test",
            n_qual_users /
            self.df_full["toggle_id_user"].drop_duplicates().shape[0],
        )
        prop_match = (
            n_qual_users /
            self.df_full["toggle_id_user"].drop_duplicates().shape[0]
        )

        df_top1_uid = pd.DataFrame.from_dict(
            {"prop_match_pref_perf": prop_match}, orient="index", columns=["value"],
        ).to_csv(f"{self.report_dir}/model_performance/pref_perf_match.csv")
        return df_top1_uid

    def plot_pred_dist(self):
        plt.close("all")
        plt.hist(
            self.gbm.predict(
                self.df_full.drop(
                    ["outcome", "toggle_font", "toggle_id_user"], axis=1,
                ),
            ),
        )

    def preferred_vs_fastest(self):
        df_tmp = self.df_full.copy()

        # wpm rank
        df_wpm = self.df_wpm.merge(
            self.df_wpm.groupby("id_user")["avg_wpm"].max().reset_index(),
            on=["id_user", "avg_wpm"],
        )
        df_tmp_wpm = df_tmp.merge(
            df_wpm.add_prefix("toggle_"), on=["toggle_id_user", "toggle_font"],
        )

        # font rank
        df_tmp_min_font_rank = (
            df_tmp.groupby(["toggle_id_user"])[
                "toggle_font_rank"
            ].min().reset_index()
        )
        df_tmp_pref = df_tmp.merge(
            df_tmp_min_font_rank,
            on=["toggle_id_user", "toggle_font_rank"],
            how="inner",
        )

        plt.close("all")
        df_merged = df_tmp_wpm.merge(df_tmp_pref, on="toggle_id_user")
        df_merged_grouped = df_merged.groupby("toggle_font_x")[
            df_merged.filter(
                regex="toggle.*(height|width|contrast"
                + "|weight|_sd|ascender|descender).*_y$",
                axis=1,
            ).columns
        ].mean()

        df_merged_grouped.columns = [
            feature_convert(c).replace(" Y", "") for c in df_merged_grouped.columns
        ]
        df_merged_grouped.index = [
            feature_convert(
                c,
            ) for c in df_merged_grouped.index
        ]
        g = sns.clustermap(
            df_merged_grouped, standard_scale=1, cmap="YlGn", annot=True, fmt=".2f",
        )
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        ax = g.ax_heatmap
        ax.set_xlabel("Preferred Font Characteristics")
        ax.set_ylabel("Fastest Font")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_pref_vs_fast_feature_dendro.png",
        )

        # violin plot alternative to see distribution
        plt.close("all")
        for c in df_merged.filter(
            regex="toggle.*(height|width|contrast|weight|_sd|ascender|descender).*_y$",
            axis=1,
        ).columns:
            tmp_c = re.sub("_y$", "", c)
            sns.boxplot(f"{tmp_c}_y", f"{tmp_c}_x", data=df_merged)
            sns.swarmplot(
                f"{tmp_c}_y", f"{tmp_c}_x", data=df_merged, color=".25",
            )
            plt.xlabel(f"preferred fonts {tmp_c}")
            plt.ylabel(f"fast fonts {tmp_c}")
            plt.savefig(
                f"{self.report_dir}/explain/" f"{tmp_c}_preferred_fastest.png",
            )

    def plot_wpm_improvement(self, top_n=1):
        df_tmp = self.df_full.copy()

        params_file = f"{self.report_dir}/params_{self.objective}.joblib"
        best_search_params = joblib.load(params_file)

        # prepare model data set
        group_kfold = GroupKFold(n_splits=self.n_splits)
        group_idx = df_tmp.groupby(group_col).ngroup()

        train_test_folds = list(
            group_kfold.split(
                df_tmp.drop("outcome", axis=1), df_tmp["outcome"], group_idx,
            ),
        )

        # append predictions from the best model
        df_tmp_base = []
        cv_metrics = []
        for f in tqdm(train_test_folds):
            tmp_train_lgb, tmp_test_lgb = get_train_test_lgb_data(
                df_tmp, f[0], f[1], n_subsample=100,
            )
            self.gbm = lgb.train(
                {**best_search_params, **FIXED_PARAMS},
                tmp_train_lgb,
                valid_sets=tmp_test_lgb,
                verbose_eval=False,
            )

            test_X = self.df_full.iloc[
                f[1],
            ]
            test_X["pred"] = self.gbm.predict(
                test_X[tmp_test_lgb.data.columns],
            )

            df_tmp_base.append(test_X)
            cv_metrics.append(self.gbm.best_score["valid_0"]["ndcg@2"])

        print(f"cv metrics from the best model is {np.mean(cv_metrics)}")

        df_tmp = (
            pd.concat(df_tmp_base, axis=0)
            .merge(
                df_wpm,
                left_on=["toggle_id_user", "toggle_font"],
                right_on=["id_user", "font"],
            )
            .drop("id_user", axis=1)
        )

        # random tie-breaking for the prediction
        rand_id_user = np.random.choice(df_tmp["toggle_id_user"].unique())
        print(
            df_tmp.query(f"toggle_id_user=={rand_id_user}")[
                ["pred", "toggle_font", "outcome"]
            ],
        )
        print(df_wpm.query(f"id_user=={rand_id_user}")[["font", "avg_wpm"]])
        print(
            df_tmp.query(f"toggle_id_user=={rand_id_user}")
            .assign(
                pred_rank=df_tmp.sample(frac=1, random_state=0, replace=False)
                .groupby("toggle_id_user")["pred"]
                .rank(ascending=False, method="first"),
            )
            .assign(
                wpm_rank=df_tmp.groupby("toggle_id_user")["avg_wpm"].rank(
                    ascending=False, method="dense",
                ),
            )[["font", "wpm_rank", "pred_rank"]],
        )

        df_tmp = df_tmp.assign(
            pred_rank=df_tmp.sample(frac=1, random_state=0, replace=False)
            .groupby("toggle_id_user")["pred"]
            .rank(ascending=False, method="first"),
        ).assign(
            wpm_rank=df_tmp.groupby("toggle_id_user")["avg_wpm"].rank(
                ascending=False, method="dense",
            ),
        )

        # create the order of fonts for plotting
        font_order = (
            df_tmp["font"].drop_duplicates(
            ).sort_values().apply(feature_convert)
        )

        # top 1 predicted
        plt.close("all")
        sns.countplot(
            df_tmp.query("pred_rank==1")["font"].apply(feature_convert),
            order=font_order,
        )
        plt.xlabel("Font")
        plt.ylabel("Number of Users")
        plt.title("Distribution of Predicted Fastest Fonts")
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_recommended_font.png",
        )
        pred_sum_df = df_tmp.query("pred_rank==1")["font"].value_counts()
        pred_sum_df.to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_recommended_font.csv",
        )

        # get the most preferred fonts
        if self.preference_mode == "exclude":
            pref_df = pd.read_csv("data/interim/elo_transposed_new.csv")
            pref_df = pref_df.drop("elo", axis=1).add_prefix("toggle_")
            df_tmp = df_tmp.merge(
                pref_df, on=["toggle_id_user", "toggle_font"],
            )

        df_tmp_min_font_rank = (
            df_tmp.groupby(["toggle_id_user"])[
                "toggle_font_rank"
            ].min().reset_index()
        )
        df_tmp_pref = df_tmp.merge(
            df_tmp_min_font_rank,
            on=["toggle_id_user", "toggle_font_rank"],
            how="inner",
        )[["toggle_id_user", "font"]]

        # top 1 preferred
        plt.close("all")
        sns.countplot(
            df_tmp_pref["font"].apply(feature_convert), order=font_order,
        )
        plt.xlabel("Font")
        plt.ylabel("Number of Users")
        plt.title("Distribution of Most Preferred Fonts")
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_preferred_font.png",
        )
        pref_sum_df = df_tmp_pref["font"].value_counts()
        pref_sum_df.to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_preferred_font.csv",
        )

        # top 1 wpm
        plt.close("all")
        # plt.figure(figsize=maj_figsize)
        sns.countplot(
            df_tmp.query("wpm_rank==1")["font"].apply(feature_convert),
            order=font_order,
        )
        plt.xlabel("Font")
        plt.ylabel("Number of Users")
        plt.title("Distribution of Actual Fastest Fonts")
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_wpm_font.png",
        )
        wpm_sum_df = df_tmp.query("wpm_rank==1")["font"].value_counts()
        wpm_sum_df.to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_wpm_font.csv",
        )

        summary_df = (
            wpm_sum_df.reset_index(name="speed")
            .merge(pref_sum_df.reset_index(name="preference"))
            .merge(pred_sum_df.reset_index(name="prediction"))
        )
        summary_df.to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_top_font_all.csv",
        )

        # view top recommended font by different user characteristics
        for sub_col in df_tmp.filter(regex="pre_").columns:
            plt.close("all")
            sns.countplot(
                df_tmp.query("pred_rank==1")["font"].apply(feature_convert),
                order=font_order,
                hue=pd.qcut(
                    df_tmp.query("pred_rank==1")[sub_col], 4, duplicates="drop",
                ),
            )
            plt.xlabel("Font")
            plt.ylabel("Number of Users")
            plt.title(
                f"Distribution of Predicted Fastest Fonts by "
                f"{feature_convert(sub_col)}",
            )
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.legend(title=feature_convert(sub_col))
            plt.savefig(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_recommended_font_by_{sub_col}.png",
            )

            pd.concat(
                [
                    df_tmp.query("pred_rank==1")[
                        "font"
                    ].apply(feature_convert),
                    pd.qcut(
                        df_tmp.query("pred_rank==1")[sub_col], 4, duplicates="drop",
                    ).astype(str),
                ],
                axis=1,
            ).to_csv(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_recommended_font_by_{sub_col}.csv",
            )

            pd.concat(
                [
                    df_tmp.query("pred_rank==1")[
                        "font"
                    ].apply(feature_convert),
                    pd.qcut(
                        df_tmp.query("pred_rank==1")[sub_col], 4, duplicates="drop",
                    ).astype(str),
                ],
                axis=1,
            ).groupby(["font", sub_col])[sub_col].count().to_csv(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_recommended_font_by_{sub_col}_grouped.csv",
            )

            plt.close("all")
            sns.countplot(
                df_tmp.query("toggle_font_rank==1")[
                    "font"
                ].apply(feature_convert),
                order=font_order,
                hue=pd.qcut(
                    df_tmp.query("toggle_font_rank==1")[sub_col], 4, duplicates="drop",
                ),
            )
            plt.xlabel("Font")
            plt.ylabel("Number of Users")
            plt.title(
                f"Distribution of Most Preferred Fonts by "
                f"{feature_convert(sub_col)}",
            )
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.legend(title=feature_convert(sub_col))
            plt.savefig(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_preferred_font_by_{sub_col}.png",
            )

            pd.concat(
                [
                    df_tmp.query("toggle_font_rank==1")[
                        "font"
                    ].apply(feature_convert),
                    pd.qcut(
                        df_tmp.query("toggle_font_rank==1")[sub_col],
                        4,
                        duplicates="drop",
                    ).astype(str),
                ],
                axis=1,
            ).to_csv(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_preferred_font_by_{sub_col}.csv",
            )

            pd.concat(
                [
                    df_tmp.query("toggle_font_rank==1")[
                        "font"
                    ].apply(feature_convert),
                    pd.qcut(
                        df_tmp.query("toggle_font_rank==1")[sub_col],
                        4,
                        duplicates="drop",
                    ).astype(str),
                ],
                axis=1,
            ).groupby(["font", sub_col])[sub_col].count().to_csv(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_preferred_font_by_{sub_col}_grouped.csv",
            )

            plt.close("all")
            sns.countplot(
                df_tmp.query("wpm_rank==1")["font"].apply(feature_convert),
                order=font_order,
                hue=pd.qcut(
                    df_tmp.query("wpm_rank==1")[sub_col], 4, duplicates="drop",
                ),
            )
            plt.xlabel("Font")
            plt.ylabel("Number of Users")
            plt.title(
                f"Distribution of Actual Fastest Fonts by "
                f"{feature_convert(sub_col)}",
            )
            plt.tight_layout()
            plt.xticks(rotation=45)
            plt.legend(title=feature_convert(sub_col))
            plt.savefig(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_wpm_font_by_{sub_col}.png",
            )

            pd.concat(
                [
                    df_tmp.query("wpm_rank==1")["font"].apply(feature_convert),
                    pd.qcut(
                        df_tmp.query("wpm_rank==1")[sub_col], 4, duplicates="drop",
                    ).astype(str),
                ],
                axis=1,
            ).to_csv(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_wpm_font_by_{sub_col}.csv",
            )

            pd.concat(
                [
                    df_tmp.query("wpm_rank==1")["font"].apply(feature_convert),
                    pd.qcut(
                        df_tmp.query("wpm_rank==1")[sub_col], 4, duplicates="drop",
                    ).astype(str),
                ],
                axis=1,
            ).groupby(["font", sub_col])[sub_col].count().to_csv(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_top_wpm_font_by_{sub_col}_grouped.csv",
            )

        # only measure WPM improvement with Top-n font
        eligible_users = (
            df_tmp.groupby("toggle_id_user")["toggle_font"]
            .nunique()
            .reset_index(name="n_fonts")
        )
        eligible_users = eligible_users.query("n_fonts>@top_n")
        print("eligible users", len(eligible_users))
        df_tmp = df_tmp[
            df_tmp["toggle_id_user"].isin(eligible_users["toggle_id_user"])
        ]

        print(
            "unique users",
            df_tmp.query("pred_rank<=@top_n")["toggle_id_user"].nunique(),
        )
        pred_avg_wpm_arr = (
            df_tmp.query("pred_rank<=@top_n")
            .groupby(["toggle_id_user", "font"])["avg_wpm"]
            .mean()
            .reset_index(name="pred_avg_wpm")
        )
        pred_avg_wpm_arr = pred_avg_wpm_arr.rename(
            columns={"font": "pred_font"},
        )

        # array characterizing the favorite font by user
        user_avg_wpm_arr = (
            df_tmp.merge(df_tmp_pref, on=["toggle_id_user", "font"])
            .groupby(["toggle_id_user", "font"])["avg_wpm"]
            .mean()
            .reset_index(name="user_avg_wpm")
        )
        user_avg_wpm_arr = user_avg_wpm_arr.rename(
            columns={"font": "pref_font"},
        )

        # include range info into consideration to understand effect of rec
        df_wpm_range = (
            df_wpm.groupby("id_user")
            .agg({"avg_wpm": ["min", "max", "std"]})
            .reset_index()
        )
        df_wpm_range.columns = [
            "toggle_id_user",
            "min_wpm",
            "max_wpm",
            "wpm_std",
        ]
        df_wpm_range["wpm_range"] = df_wpm_range["max_wpm"] - \
            df_wpm_range["min_wpm"]
        df_wpm_max = df_wpm.merge(
            df_wpm.groupby("id_user")["avg_wpm"].max().reset_index(),
            on=["id_user", "avg_wpm"],
        )
        df_wpm_max = df_wpm_max[["id_user", "font", "avg_wpm"]]
        df_wpm_max = df_wpm_max.rename(
            columns={"font": "fastest_font", "id_user": "toggle_id_user"},
        )

        # n top fonts
        df_wpm_n_top = df_wpm.merge(
            df_wpm_range[["toggle_id_user", "max_wpm"]],
            left_on="id_user",
            right_on="toggle_id_user",
            validate="m:1",
        )
        df_wpm_n_top = df_wpm_n_top[
            df_wpm_n_top["avg_wpm"] >= self.wpm_threshold *
            df_wpm_n_top["max_wpm"]
        ]
        df_wpm_n_top = (
            df_wpm_n_top.groupby("toggle_id_user")["font"]
            .nunique()
            .reset_index(name="pre_n_top_fonts")
        )

        df_improve = (
            pred_avg_wpm_arr.merge(user_avg_wpm_arr, on="toggle_id_user")
            .merge(
                df_tmp.filter(
                    regex="(^pre_)|id_user",
                    axis=1,
                ).drop_duplicates(),
                on="toggle_id_user",
            )
            .merge(df_wpm_range, on="toggle_id_user")
            .merge(df_wpm_n_top, on="toggle_id_user")
            .merge(df_wpm_max.drop("avg_wpm", axis=1), on="toggle_id_user")
        )
        # user_avg_wpm here is the preferred speed
        df_improve["wpm_improve"] = df_improve.eval(
            "pred_avg_wpm-user_avg_wpm",
        )
        df_improve["wpm_improve_perc"] = df_improve.eval(
            "(pred_avg_wpm-user_avg_wpm)/user_avg_wpm",
        )

        # wpm improvement fastest over preference
        pd.DataFrame.from_dict(
            {
                "mean_speed_improvement": np.mean(
                    df_improve["max_wpm"] - df_improve["user_avg_wpm"],
                ),
            },
            orient="index",
        ).to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_improve_max_over_pref.csv",
        )

        fast_pref_improve_df = pd.concat(
            [
                df_improve["fastest_font"],
                df_improve["pref_font"],
                df_improve.eval("max_wpm-user_avg_wpm"),
            ],
            axis=1,
        )
        fast_pref_improve_df.columns = [
            "fastest_font",
            "pref_font",
            "wpm_improve",
        ]
        fast_pref_improve_df = (
            fast_pref_improve_df.groupby(["fastest_font", "pref_font"])[
                "wpm_improve"
            ]
            .mean()
            .reset_index()
        )
        fast_pref_improve_df["fastest_font"] = fast_pref_improve_df[
            "fastest_font"
        ].apply(feature_convert)
        fast_pref_improve_df["pref_font"] = fast_pref_improve_df["pref_font"].apply(
            feature_convert,
        )

        plt.close("all")
        sns.heatmap(
            fast_pref_improve_df.pivot(
                "fastest_font", "pref_font", "wpm_improve",
            ),
            cmap="YlGn",
            annot=True,
            fmt=".2f",
        )
        plt.xlabel("Preferred Font")
        plt.ylabel("Fastest Font")
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_improve_max_over_pref.png",
        )

        # cluster map
        plt.close("all")
        fast_pref_improve_df_pivot = fast_pref_improve_df.pivot(
            "fastest_font", "pref_font", "wpm_improve",
        )
        fg = sns.clustermap(
            fast_pref_improve_df_pivot.fillna(0),
            standard_scale=1,
            cmap="YlGn",
            annot=True,
            fmt=".2f",
        )
        ax = fg.ax_heatmap
        ax.set_xlabel("Preferred Font")
        ax.set_ylabel("Fastest Font")
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_improve_max_over_pref_dendro.png",
        )

        # wpm improvement fastest over preference
        pd.DataFrame.from_dict(
            {
                "mean_speed_improvement": np.mean(
                    df_improve["pred_avg_wpm"] - df_improve["user_avg_wpm"],
                ),
            },
            orient="index",
        ).to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_improve_pred_max_over_pref.csv",
        )

        pred_pref_improve_df = pd.concat(
            [
                df_improve["pred_font"],
                df_improve["pref_font"],
                df_improve.eval("pred_avg_wpm-user_avg_wpm"),
            ],
            axis=1,
        )
        pred_pref_improve_df.columns = [
            "pred_font",
            "pref_font",
            "wpm_improve",
        ]
        pred_pref_improve_df = (
            pred_pref_improve_df.groupby(["pred_font", "pref_font"])[
                "wpm_improve"
            ]
            .mean()
            .reset_index()
        )
        pred_pref_improve_df["pred_font"] = pred_pref_improve_df["pred_font"].apply(
            feature_convert,
        )
        pred_pref_improve_df["pref_font"] = pred_pref_improve_df["pref_font"].apply(
            feature_convert,
        )
        pred_pref_improve_df = pred_pref_improve_df.pivot(
            "pred_font", "pref_font", "wpm_improve",
        )
        cmap_max_val = np.max(
            [
                np.abs(np.nanmax(pred_pref_improve_df.values)),
                np.abs(np.nanmin(pred_pref_improve_df.values)),
            ],
        )

        df_improve.query('pred_font=="poppins"&pref_font=="arial"').eval(
            "pred_avg_wpm-user_avg_wpm",
        ).mean()

        plt.close("all")
        sns.heatmap(
            pred_pref_improve_df,
            cmap="RdYlGn",
            vmax=cmap_max_val,
            vmin=-cmap_max_val,
            annot=True,
            fmt=".2f",
        )
        plt.xlabel("Preferred Font")
        plt.ylabel("Recommended Font")
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_improve_pred_max_over_pref.png",
        )

        font_ref_improve = {}
        for f in df_tmp["toggle_font"].drop_duplicates():
            # set font
            tmp_font_speed = (
                df_tmp.query("toggle_font==@f")[["toggle_id_user", "avg_wpm"]]
                .drop_duplicates()
                .rename(columns={"avg_wpm": f"ref_{f}_wpm"})
            )
            tmp_df_improve = df_improve.merge(
                tmp_font_speed, on="toggle_id_user",
            )
            font_ref_improve[f] = (
                tmp_df_improve["pred_avg_wpm"] - tmp_df_improve[f"ref_{f}_wpm"]
            )

            # distribution of wpm improvements
            plt.close("all")
            plt.figure(figsize=maj_figsize)
            plt.hist(font_ref_improve[f], bins=50, alpha=0.8)
            plt.title(
                f"Distribution of WPM Improvement from {feature_convert(f)}, "
                f"mean {font_ref_improve[f].mean():.2f}, "
                f"median {font_ref_improve[f].median():.2f}",
            )
            plt.axvline(0, marker=".", c="black")
            plt.tight_layout()
            plt.savefig(
                f"{self.report_dir}/model_performance/"
                f"{self.objective}_wpm_improvement_from_{f}.png",
            )

        # improvement from random font
        f = "random"
        tmp_font_speed = (
            df_tmp.groupby("toggle_id_user")["avg_wpm"]
            .agg(np.random.choice)
            .reset_index()
            .drop_duplicates()
            .rename(columns={"avg_wpm": f"ref_{f}_wpm"})
        )
        tmp_df_improve = df_improve.merge(tmp_font_speed, on="toggle_id_user")
        font_ref_improve[f] = (
            tmp_df_improve["pred_avg_wpm"] - tmp_df_improve[f"ref_{f}_wpm"]
        )

        # distribution of wpm improvements
        plt.close("all")
        plt.figure(figsize=maj_figsize)
        plt.hist(font_ref_improve[f], bins=50, alpha=0.8)
        plt.title(
            f"Distribution of WPM Improvement from {feature_convert(f)}, "
            f"mean {font_ref_improve[f].mean():.2f}, "
            f"median {font_ref_improve[f].median():.2f}",
        )
        plt.axvline(0, marker=".", c="black")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_wpm_improvement_from_{f}.png",
        )

        font_ref_improve = pd.DataFrame.from_dict(
            font_ref_improve, orient="index",
        ).T

        font_ref_improve.mean().sort_values().to_csv(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_wpm_improvement.csv",
        )

        plt.close("all")
        sns.kdeplot(data=font_ref_improve)
        for i, c in enumerate(font_ref_improve.columns):
            vline_pos = np.mean(font_ref_improve[c])
            plt.axvline(vline_pos, c="black")
            plt.text(vline_pos + 0.1, 0, c, rotation=90)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_wpm_improvement_from_all_fonts.png",
        )

        # distribution of wpm improvements
        plt.close("all")
        plt.figure(figsize=maj_figsize)
        plt.hist(df_improve["wpm_improve"], bins=50, alpha=0.8)
        plt.title(
            f"Distribution of WPM Improvement from Preferred, mean "
            f'{df_improve["wpm_improve"].mean():.2f}, median '
            f'{df_improve["wpm_improve"].median():.2f}',
        )
        plt.axvline(0, marker=".", c="black")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_wpm_improvement.png",
        )

        # distribution of wpm improvements from max
        tmp_font_wpm_improve = df_improve["pred_avg_wpm"] - \
            df_improve["max_wpm"]
        plt.close("all")
        plt.figure(figsize=maj_figsize)
        plt.hist(tmp_font_wpm_improve, bins=50, alpha=0.8)
        plt.title(
            f"Distribution of WPM Improvement from Max Speed, mean "
            f"{tmp_font_wpm_improve.mean():.2f}, median "
            f"{tmp_font_wpm_improve.median():.2f}",
        )
        plt.axvline(0, marker=".", c="black")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_wpm_improvement_from_max.png",
        )

        # print wpm range output
        np.mean(
            df_improve[
                df_improve["wpm_improve"]
                .between(-100, 100)
            ]["wpm_improve"],
        )
        np.median(
            df_improve[
                df_improve["wpm_improve"]
                .between(-100, 100)
            ]["wpm_improve"],
        )

        kept_list = df_improve.drop(
            [
                "toggle_id_user",
                "pred_avg_wpm",
                "wpm_improve",
                "wpm_improve_perc",
                "pred_font",
                "pref_font",
                "fastest_font",
            ],
            axis=1,
        ).columns
        print(kept_list)

    def plot_model_explain(self):
        """miscellaneous explainability functions"""
        X = self.final_train_lgb.get_data()

        plt.close("all")
        plt.hist(self.gbm.predict(X), bins=50)
        plt.title("Distribution of Ranking Prediction")
        plt.tight_layout()
        plt.savefig(
            f"{self.report_dir}/model_performance/"
            f"{self.objective}_perf_pred_distribution.png",
        )

        # feature importance
        plt.close("all")
        var_imp_df = pd.DataFrame(
            np.array(
                [
                    self.gbm.feature_name(),
                    self.gbm.feature_importance(importance_type="gain"),
                ],
            ).T,
            columns=["feature", "importance"],
        )
        var_imp_df["importance"] = var_imp_df["importance"].astype(float)
        var_imp_df["feature_clean"] = var_imp_df["feature"].apply(
            lambda s: feature_convert(s),
        )
        var_imp_df = var_imp_df.sort_values("importance", ascending=False)

        # ------------ shapley value
        # get a subset of features for shapley plots
        imp_toggle_cols = var_imp_df[
            (
                var_imp_df["importance"] > np.percentile(
                    var_imp_df["importance"], 5,
                )
            )
            & (var_imp_df["feature"].str.contains("^(toggle_|pref_font)"))
        ]["feature"]
        imp_user_cols = var_imp_df[
            (
                var_imp_df["importance"] > np.percentile(
                    var_imp_df["importance"], 5,
                )
            )
            & (var_imp_df["feature"].str.contains("rank_|pre_|toggle_|pref_font"))
        ]["feature"]

        # get shapley values
        explainer = shap.TreeExplainer(self.gbm)
        shap_values_explainer = explainer.shap_values(X)

        plt.close("all")
        shap.summary_plot(shap_values_explainer, X, show=False)
        plt.tight_layout()
        plt.savefig(f"{self.report_dir}/explain/summary_shapley.png")

        plt.close("all")
        for col_a, col_b in itertools.product(imp_user_cols, imp_user_cols):
            if col_a == col_b:
                continue
            idx_col_a = np.where(X.columns == col_a)[0][0]
            idx_col_b = np.where(X.columns == col_b)[0][0]

            tmp_group_df = pd.concat(
                [
                    X.iloc[:, idx_col_a].reset_index(drop=True),
                    X.iloc[:, idx_col_b].reset_index(drop=True),
                    pd.DataFrame(shap_values_explainer[:, idx_col_a]),
                ],
                axis=1,
            )
            tmp_group_df.columns = [col_a, col_b, "shap"]
            tmp_group_df = (
                tmp_group_df.groupby([col_a, col_b])[
                    "shap"
                ].mean().reset_index()
            )
            tmp_group_df_wide = tmp_group_df.pivot(col_a, col_b, "shap")

            plt.close("all")
            plt.figure(figsize=(12, 12))
            if (
                len(X[[col_a, col_b]].drop_duplicates()) <= 8
                and "toggle" in col_a
                and "toggle" in col_b
            ):

                # link font names for annotation
                tmp_font_info = self.df_full[[col_a, col_b, "toggle_font"]]
                tmp_font_info["toggle_font"] = tmp_font_info["toggle_font"].apply(
                    lambda s: s.replace("_", " ").title(),
                )
                print(col_a, col_b)
                try:
                    tmp_font_index = (
                        X[[col_a, col_b]]
                        .drop_duplicates()
                        .merge(tmp_font_info, on=[col_a, col_b])
                        .drop_duplicates()
                        .sort_values([col_a, col_b], ascending=[False, True])
                        .pivot(col_a, col_b, "toggle_font")
                    )
                    ax = sns.heatmap(
                        tmp_group_df_wide,
                        cmap="RdYlGn",
                        annot=tmp_font_index.values,
                        fmt="",
                        vmin=-np.nanmax(tmp_group_df_wide.values),
                        vmax=np.nanmax(tmp_group_df_wide.values),
                    )
                except ValueError:
                    continue
            else:
                ax = sns.heatmap(
                    tmp_group_df_wide,
                    cmap="RdYlGn",
                    vmin=-np.nanmax(tmp_group_df_wide.values),
                    vmax=np.nanmax(tmp_group_df_wide.values),
                )

            ax.invert_yaxis()
            plt.tight_layout()
            plt.xlabel(feature_convert(col_b))
            plt.ylabel(feature_convert(col_a))
            plt.savefig(
                f"{self.report_dir}/explain/heatmap_" f"{col_a}_{col_b}_lgbm.png",
            )

            tmp_group_df.to_csv(
                f"{self.report_dir}/explain/heatmap_" f"{col_a}_{col_b}_lgbm.csv",
            )

        # modifying for updated visuals
        plt.close("all")
        toggle_cols = var_imp_df["feature"][
            var_imp_df["feature"].str.contains("toggle_")
        ].tolist()
        if self.preference_mode == "font_char_diff":
            toggle_cols += var_imp_df["feature"][
                var_imp_df["feature"].str.contains("toggle_diff")
            ].tolist()
        toggle_cols = [
            t
            for t in toggle_cols
            if var_imp_df.query("feature==@t")["importance"].values[0] > 0
        ]
        user_char_cols = X.filter(regex="pre_|pref_", axis=1).columns

        if len(user_char_cols) > 8:
            figsize = (20, 30)
        else:
            figsize = None

        fig, axes = plt.subplots(
            len(user_char_cols), len(toggle_cols), figsize=figsize,
        )

        if self.objective == "preference":
            return

        vlim = np.max(
            (
                np.abs(np.nanpercentile(shap_values_explainer, 2.5)),
                np.abs(np.nanpercentile(shap_values_explainer, 97.5)),
            ),
        )
        for i, j in itertools.product(
            range(len(toggle_cols)), range(len(user_char_cols)),
        ):
            col_a, col_b = toggle_cols[i], user_char_cols[j]

            idx_col_a = np.where(X.columns == col_a)[0][0]
            idx_col_b = np.where(X.columns == col_b)[0][0]

            tmp_group_df = pd.concat(
                [
                    X.iloc[:, idx_col_a].reset_index(drop=True),
                    X.iloc[:, idx_col_b].reset_index(drop=True),
                    pd.DataFrame(shap_values_explainer[:, idx_col_a]),
                ],
                axis=1,
            )
            tmp_group_df.columns = [col_a, col_b, "shap"]

            tmp_group_df = (
                tmp_group_df.groupby([col_a, col_b])[
                    "shap"
                ].mean().reset_index()
            )
            tmp_group_df_wide = tmp_group_df.pivot(col_b, col_a, "shap")

            g = sns.heatmap(
                tmp_group_df_wide,
                cmap="RdYlGn",
                vmin=-vlim,
                vmax=vlim,
                ax=axes[j, i],
                cbar=False,
            )
            tmp_group_df_wide.to_csv(
                f"{self.report_dir}/explain/overall_shapley_{i}_{j}_{vlim}.csv",
            )
            g.invert_yaxis()

            if i != 0:
                g.set(ylabel=None)
                g.set(yticklabels=[])
                g.tick_params(left=False)
            else:
                axes[j, i].set_ylabel(
                    feature_convert(col_b).replace("User ", ""),
                    rotation=0,
                    horizontalalignment="right",
                )
            if j != len(user_char_cols) - 1:
                g.set(xlabel=None)
                g.set(xticklabels=[])
                g.tick_params(bottom=False)
            else:
                axes[j, i].set_xlabel(
                    feature_convert(col_a).replace("Font ", ""), rotation=90,
                )

        plt.tight_layout()

        for col_a in imp_toggle_cols:
            plt.close("all")
            _, ax = plt.subplots(figsize=(8, 6))

            ax.axhline(0, c="black", linestyle=":")

            # regression – lowess smoothing
            idx_col_a = np.where(X.columns == col_a)[0][0]
            lowess = sm.nonparametric.lowess(
                shap_values_explainer[
                    :,
                    idx_col_a,
                ], X.iloc[:, idx_col_a], frac=0.5,
            )
            ax.plot(
                *list(zip(*lowess)), color="green",
            )

            shap.dependence_plot(
                col_a,
                shap_values_explainer,
                X,
                interaction_index=None,
                # alpha=.3,
                ax=ax,
                xmin="percentile(1)",
                xmax="percentile(99)",
                show=False,
            )

            plt.xlabel(feature_convert(col_a))
            plt.ylabel(
                f"SHAP value for the '{feature_convert(col_a)}' feature",
            )

            # set y lim
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            plt.ylim(min(0.99 * y_min, 0), 0.99 * y_max)

            plt.tight_layout()
            plt.savefig(
                f"{self.report_dir}/explain/singleton_shapley_{col_a}.png",
            )

    def view_obs_comp_scatter(self):
        """one way to determine best comprehension cutoff is the elbow plot"""
        n_sampling_points = 51
        n_users = []
        for k in np.linspace(0, 1, n_sampling_points):
            n_users.append(
                len(
                    df_wpm[df_wpm["overall_error_rate"] < k][
                        "id_user"
                    ].drop_duplicates(),
                ),
            )

        kn = KneeLocator(
            np.linspace(0, 1, n_sampling_points),
            n_users,
            curve="concave",
            direction="increasing",
        )
        print(kn.knee)

        plt.close("all")
        sns.lineplot(
            np.linspace(0, 1, n_sampling_points), n_users, markers=True, dashes=False,
        )
        plt.axvline(kn.knee, color="black")
        plt.title("Elbow Method for Identifying Error Cutoff")
        plt.savefig(f"{self.report_dir}/input/error_cutoff.png")

    def evaluate(self):
        self.calculate_baseline_metrics()
        self.plot_font_features()
        self.plot_pred_dist()
        self.plot_model_explain()

        if self.objective == "performance" and self.label_type != "graded":
            self.plot_top1_proportion()
            self.plot_n_top_fonts()
            self.plot_wpm_improvement()
            self.view_obs_comp_scatter()
            if self.preference_mode != "exclude":
                self.calculate_prop_identified_top1()
        elif self.objective == "performance" and self.label_type == "graded":
            self.plot_wpm_improvement()


class Model(Diagnostic):
    def __init__(
        self,
        wpm_threshold=0.95,
        label_type="nonormalize",
        preference_mode="include",
        objective="performance",
        retrain=False,
        n_splits=50,
    ):
        self.wpm_threshold = wpm_threshold
        self.label_type = label_type
        self.preference_mode = preference_mode  # include,exclude, font_char
        self.retrain = retrain
        self.objective = objective
        self.n_splits = n_splits

        self.report_dir = (
            f"reports/{self.objective}_{str(self.label_type)}_"
            f"{str(self.preference_mode)}"
            f'{str(self.wpm_threshold).split(".")[1]}'
        )

        for d in [""]:
            os.makedirs(f"{self.report_dir}/input/{d}", exist_ok=True)
            os.makedirs(f"{self.report_dir}/explain/{d}", exist_ok=True)
            os.makedirs(
                f"{self.report_dir}/model_performance/{d}", exist_ok=True,
            )

        # get data
        self.data_obj = Data(
            report_dir="reports",
            wpm_threshold=self.wpm_threshold,
            label_type=self.label_type,
            preference_mode=self.preference_mode,
            objective=self.objective,
        )

        self.df_full, self.train_test_folds = self.data_obj.prep_data_and_split()

        # join font preference data, elo score, NOT directed graph
        df_preferred = pd.read_csv("data/interim/elo_transposed_new.csv")
        df_preferred = df_preferred[["id_user", "font", "elo"]]
        df_preferred["rank"] = df_preferred.groupby(["id_user"])["elo"].rank(
            method="dense", ascending=False,
        )
        self.df_preferred_wide = df_preferred.pivot(index="id_user", columns="font")[
            "rank"
        ]
        self.df_preferred_wide = self.df_preferred_wide.reset_index()
        self.df_preferred_wide.columns = ["toggle_id_user"] + list(
            "rank_" + self.df_preferred_wide.columns[1:],
        )

        # train model on a slice
        self.gbm = self.get_final_model(
            df_full=self.df_full, train_test_folds=self.train_test_folds,
        )

        #
        self.final_train_lgb, self.final_test_lgb = get_train_test_lgb_data(
            self.df_full, self.train_test_folds[0][0], self.train_test_folds[0][1],
        )
        self.final_train_lgb.construct()
        self.final_test_lgb.construct()

    def train_evaluate(self, search_params, df_full, train_test_folds, target_metric):
        """helper function for tuning"""

        def slice_score(fold):
            params = {**search_params, **FIXED_PARAMS}

            tmp_train_lgb, tmp_test_lgb = get_train_test_lgb_data(
                df_full, fold[0], fold[1], n_subsample=100,
            )
            gbm = lgb.train(
                params, tmp_train_lgb, valid_sets=tmp_test_lgb, verbose_eval=False,
            )

            score_dict = dict(gbm.best_score["valid_0"])

            if "toggle_font_rank" not in df_full:
                return {**score_dict}

            # slice wpm improvements
            tmp_test_df = df_full.iloc[fold[1], :]
            df_wpm_improve = tmp_test_df.merge(
                df_wpm,
                left_on=["toggle_id_user", "toggle_font"],
                right_on=["id_user", "font"],
            )

            df_wpm_improve["pred"] = gbm.predict(
                tmp_test_df[tmp_test_lgb.data.columns],
            )

            df_wpm_improve = df_wpm_improve.merge(
                df_wpm_improve.groupby("toggle_id_user")["avg_wpm"]
                .max()
                .reset_index(name="max_wpm"),
            )
            df_wpm_improve = df_wpm_improve.merge(
                df_wpm_improve.groupby("toggle_id_user")["avg_wpm"]
                .min()
                .reset_index(name="min_wpm"),
            )
            df_wpm_improve["wpm_range"] = df_wpm_improve.eval(
                "max_wpm-min_wpm",
            )

            df_wpm_improve = df_wpm_improve.drop(
                ["max_wpm", "min_wpm", "wpm_range"], axis=1,
            )

            df_wpm_improve["pred_rank"] = df_wpm_improve.groupby("toggle_id_user")[
                "pred"
            ].rank(ascending=False)

            wpm_improvement_dict = {}
            for i in range(2, 7):

                pred_wpm = (
                    df_wpm_improve.query("pred_rank<=@i")
                    .groupby(["toggle_id_user", "pre_age"])["avg_wpm"]
                    .median()
                    .reset_index(name="pred_wpm")
                )
                user_wpm = (
                    df_wpm_improve.query("toggle_font_rank<=@i")
                    .groupby(["toggle_id_user", "pre_age"])["avg_wpm"]
                    .median()
                    .reset_index(name="user_wpm")
                )
                rand_wpm = (
                    df_wpm_improve.groupby(["toggle_id_user", "pre_age"])
                    .sample(n=i, replace=True)
                    .groupby(["toggle_id_user", "pre_age"])["avg_wpm"]
                    .median()
                    .reset_index(name="rand_wpm")
                )

                delta_wpm = pred_wpm.merge(
                    user_wpm, on=["toggle_id_user", "pre_age"],
                ).merge(rand_wpm, on=["toggle_id_user", "pre_age"])

                # log various wpm improvements
                wpm_improvement_dict[f"wpm_improve_mean@{i}"] = np.mean(
                    delta_wpm.eval("pred_wpm-user_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_rand_mean@{i}"] = np.mean(
                    delta_wpm.eval("pred_wpm-rand_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_median@{i}"] = np.median(
                    delta_wpm.eval("pred_wpm-user_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_rand_median@{i}"] = np.median(
                    delta_wpm.eval("pred_wpm-rand_wpm"),
                )

                wpm_improvement_dict[f"wpm_improve_mean<=35@{i}"] = np.mean(
                    delta_wpm.query("pre_age<=35").eval("pred_wpm-user_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_rand_mean<=35@{i}"] = np.mean(
                    delta_wpm.query("pre_age<=35").eval("pred_wpm-rand_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_median<=35@{i}"] = np.median(
                    delta_wpm.query("pre_age<=35").eval("pred_wpm-user_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_rand_median<=35@{i}"] = np.median(
                    delta_wpm.query("pre_age<=35").eval("pred_wpm-rand_wpm"),
                )

                wpm_improvement_dict[f"wpm_improve_mean>35@{i}"] = np.mean(
                    delta_wpm.query("pre_age>35").eval("pred_wpm-user_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_rand_mean>35@{i}"] = np.mean(
                    delta_wpm.query("pre_age>35").eval("pred_wpm-rand_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_median>35@{i}"] = np.median(
                    delta_wpm.query("pre_age>35").eval("pred_wpm-user_wpm"),
                )
                wpm_improvement_dict[f"wpm_improve_rand_median>35@{i}"] = np.median(
                    delta_wpm.query("pre_age>35").eval("pred_wpm-rand_wpm"),
                )

            return {**score_dict, **wpm_improvement_dict}

        test_ndcg_list = Parallel(n_jobs=-1)(
            delayed(slice_score)(fold) for fold in train_test_folds
        )

        # log feature importance
        tmp_train_lgb, tmp_test_lgb = get_train_test_lgb_data(
            df_full, train_test_folds[0][0], train_test_folds[0][1], n_subsample=100,
        )

        gbm = lgb.train(
            {**search_params, **FIXED_PARAMS},
            tmp_train_lgb,
            valid_sets=tmp_test_lgb,
            verbose_eval=False,
        )

        # training were done with neptune metrics logging
        # log feature importance if on the first fold
        for idx, f in enumerate(gbm.feature_name()):
            # neptune.log_metric
            print(
                f"varimp/{f}", gbm.feature_importance(
                    importance_type="gain",
                )[idx],
            )

        # log other metrics
        for m in test_ndcg_list[0].keys():
            # neptune.log_metric
            print(
                f"eval/{m}", np.mean([metric[m] for metric in test_ndcg_list]),
            )

        if (
            np.mean([metric[target_metric] for metric in test_ndcg_list]) == 1
        ):  # in the case of graded labels
            alt_target_metric = "ndcg@2"
            return np.mean([metric[alt_target_metric] for metric in test_ndcg_list])
        else:
            return np.mean([metric[target_metric] for metric in test_ndcg_list])

    def model_tuning(
        self, df_full, train_test_folds, n_calls=100, n_random_starts=50,
    ):
        """tune and save best model parameters"""
        # neptune.create_experiment(
        #     "lgb-tuning",
        #     tags=["lgb-tuning"],
        #     params=SEARCH_PARAMS,
        #     upload_source_files="src/model/xgboost_lambdamart/*.py",
        # )

        @skopt.utils.use_named_args(SPACE)
        def obj_func(**params):
            return -1.0 * self.train_evaluate(
                params, df_full, train_test_folds, target_metric="map@8",
            )

        monitor = sk_utils.NeptuneMonitor()
        results = skopt.forest_minimize(
            obj_func,
            SPACE,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            callback=[monitor],
            random_state=1,
            n_jobs=-1,
        )
        sk_utils.log_results(results)

        best_search_params = {
            k: results["x"][i] for i, k in enumerate(SEARCH_PARAMS.keys())
        }

        # save and restore best parameters
        joblib.dump(
            best_search_params, f"{self.report_dir}/params_{self.objective}.joblib",
        )

        # neptune.stop()

    def get_final_model(self, df_full=None, train_test_folds=None):
        """obtain the trained best model"""
        params_file = f"{self.report_dir}/params_{self.objective}.joblib"
        if df_full is None and train_test_folds is None:
            df_full, train_test_folds = self.data_obj.prep_data_and_split()

        if not os.path.isfile(params_file) or self.retrain:
            print(f"--- retraining {self.objective} model")
            self.model_tuning(
                df_full, train_test_folds, n_calls=100, n_random_starts=50,
            )

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        best_search_params = joblib.load(params_file)
        with open(f"{self.report_dir}/model_param.json", "w") as fp:
            json.dump(best_search_params, fp, default=np_encoder)
        print(best_search_params)

        cv_metrics = []
        for fold in train_test_folds:
            final_train_lgb, final_test_lgb = get_train_test_lgb_data(
                df_full, fold[0], fold[1],
            )

            gbm = lgb.train(
                {**best_search_params, **FIXED_PARAMS},
                final_train_lgb,
                valid_sets=[final_test_lgb],
            )
            cv_metrics.append(gbm.best_score["valid_0"]["ndcg@2"])

        print(f"ndcg@2 from the best model is {np.mean(cv_metrics)}")

        return gbm


def main(argv):
    model = Model(
        wpm_threshold=FLAGS.wpm_threshold,
        label_type=FLAGS.label_type,
        preference_mode=FLAGS.preference_mode,
        retrain=FLAGS.retrain,
        objective=FLAGS.objective,
    )
    model.evaluate()


if __name__ == "__main__":
    # for pref_mode in ['include','font_char','exclude','font_char_diff']:
    # for pref_mode in ['exclude', 'include']:
    model = Model(
        wpm_threshold=0.9,
        label_type="nonormalize",
        retrain=False,
        preference_mode="exclude",
        objective="performance",
    )
    model.evaluate()

    # for pref_mode in ['include', 'exclude']:
    #     model = Model(wpm_threshold=.9, label_type='nonormalize',
    #                   retrain=False, preference_mode=pref_mode,
    #                   objective='performance')
    #     model.run_diagnostic()

    # model = Model(wpm_threshold=.9, label_type='graded',
    #               retrain=False, preference_mode='include',
    #               objective='performance')
    # model.run_diagnostic()

    # # need to include pref info b/c they'll be converted to outcome
    # model = Model(wpm_threshold=.9, label_type='nonormalize',
    #               retrain=False, preference_mode='include',
    #               objective='preference')
    # model.run_diagnostic()

    # app.run(main)
