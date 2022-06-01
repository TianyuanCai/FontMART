from __future__ import annotations

import janitor
import numpy as np
import pandas as pd
import pandas_flavor as pf
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold

from helper.data_build import make_name

np.random.seed(1)

GROUP_COL = "toggle_id_user"
EVENT_RENAME_FILE = "src/data/shorten_questions.yaml"
EVENT_RENAME_DICT = yaml.load(open(EVENT_RENAME_FILE), Loader=yaml.FullLoader)

# pairwise comparison raw data
fonts = [
    "arial",
    "avantgarde",
    "avenir_next",
    "calibri",
    "franklin_gothic",
    "garamond",
    "helvetica",
    "lato",
    "montserrat",
    "noto_sans",
    "open_sans",
    "oswald",
    "poynter_gothic_text",
    "roboto",
    "times",
    "utopia",
]

font_data_js = pd.read_csv("data/processed/font_metrics_js_combo.csv")
if "font-name" in font_data_js.columns:
    font_data_js = font_data_js.rename(columns={"font-name": "font"})
font_data_js.columns = [make_name(c) for c in font_data_js.columns]


def get_font_data(font):
    return font_data_js[font_data_js["font"] == font]


def find_index_col(data):
    index_cols = []
    for col in data.columns:
        if len(set(data[col])) == data.shape[0]:
            index_cols.append(col)

    return index_cols


def calculate_elo(df):
    users = []
    user_scores = {}

    # seed elo
    scores = {}
    elo = {}
    seed = 1500
    K = 64  # 32 is the default for chess, K is how quick the rating system
    # should react to results

    # df = pd.read_csv(filename)
    for f in sorted(df.font_a.unique()):
        elo[f] = seed
        scores[f] = []

    # formulas to compute elo
    def R_x(r):
        return pow(10, (r / 400))

    def E_1(R_1, R_2):  # expected score
        return R_1 / (R_1 + R_2)

    def E_2(R_1, R_2):  # expected score
        return R_2 / (R_1 + R_2)

    def r_x_new_elo(r, S, E):
        return r + K * (S - E)

    def compute_elo(uid, matchups):
        # compute elo from synthesized matchups
        for match, result in matchups.items():
            f1 = match[0]
            f2 = match[1]
            r_1 = elo[f1]
            r_2 = elo[f2]
            R_1 = R_x(r_1)
            R_2 = R_x(r_2)
            wn = result["winner"]
            if wn == f1:
                S_1 = 1
                S_2 = 0
            else:
                S_1 = 0
                S_2 = 1
            elo[f1] = r_x_new_elo(r_1, S_1, E_1(R_1, R_2))
            elo[f2] = r_x_new_elo(r_2, S_2, E_2(R_1, R_2))

        users.append(uid)
        user_scores[uid] = {}
        for f in sorted(elo.keys()):
            score = elo[f]
            scores[f].append(round(score))
            user_scores[uid][f] = score

    def run_elo_from_csv():
        for uid in df.id_user.unique():
            matches = {}
            for index, row in df.loc[df["id_user"] == uid].iterrows():
                match = (row["font_a"], row["font_b"])
                matches[match] = {"winner": row["winning_font"]}
            compute_elo(uid, matches)

    run_elo_from_csv()
    return user_scores


@pf.register_dataframe_method
def clean_df(data, verbose=False):
    # drop NA cols
    na_cols = data.columns[data.isna().sum() == len(data)]
    data = data.drop(na_cols, axis=1)

    # uninformative column
    useless_cols = list(data.columns[data.apply(lambda x: len(set(x))) == 1])
    useless_cols += list(data.filter(regex="^in_.*\\?", axis=1).columns)
    data = data.drop(useless_cols, axis=1)

    # print various processing logs
    if verbose:
        index_cols = find_index_col(data)
        if len(na_cols) > 0:
            print("---\nDropped NA columns:\n\t", ", ".join(na_cols))
        if len(useless_cols) > 0:
            print("Dropped uninformative columns:\n\t", ", ".join(useless_cols))
        if len(index_cols) > 0:
            print("Data unique by:\n\t", ", ".join(index_cols))
        else:
            print("No single unique columns")
        print(
            "Possible ID cols:\n\t",
            ", ".join(data.columns[data.columns.str.contains("id")]),
        )

    return data


class Data:
    def __init__(
        self,
        objective,
        report_dir,
        label_type,
        preference_mode,
        wpm_threshold,
        error_thresh=0.34,
    ) -> None:
        self.objective = objective
        self.report_dir = report_dir
        self.label_type = label_type
        self.preference_mode = preference_mode
        self.wpm_threshold = wpm_threshold
        self.error_thresh = error_thresh
        self.df_full = None

        self.df_toggle = pd.read_csv(
            "data/processed/new_df_toggle.csv",
        ).clean_df()
        self.df_wpm = pd.read_csv("data/processed/new_df_wpm.csv").clean_df()
        self.df_post = pd.read_csv(
            "data/raw/ML - 8 Fonts - Post-Survey (Responses) - Form Responses 1.csv",
        )
        self.df_pre = pd.read_csv(
            "data/raw/ML - 8 Fonts - Pre-Survey (Responses) - Form Responses 1.csv",
        )

        # data for majority vote update font data base table
        processed_font_matchup_df = self.get_toggle_data()
        processed_font_matchup_df.to_csv(
            "reports/input/detailed_matchup_df.csv", index=False,
        )
        self.get_new_model_data()

    def get_survey_data(
        self, df_pre, df_post,
    ):
        """Process the pre and post survey information.

        Args:
            df_pre ([pd.DataFrame]): PRE-survey data
            df_post ([pd.DataFrame]): POST-survey data

        Returns:
            [type]: Combined user characteristics dataframe
        """
        df_pre = (
            df_pre.clean_df()
            .rename(EVENT_RENAME_DICT["rename"], axis=1)
            .drop_duplicates("id_user", keep=False)
        )
        df_post = (
            df_post.clean_df()
            .rename(EVENT_RENAME_DICT["rename"], axis=1)
            .drop_duplicates("id_user", keep=False)
        )

        # combine user data
        df = df_pre.add_prefix("pre_").merge(
            df_post.add_prefix("post_"),
            left_on="pre_id_user",
            right_on="post_id_user",
            validate="1:1",
        )

        # remove participants with weird responses
        df = df[df["pre_under_influence"].astype(str).apply(len) < 100]
        df = df[
            ~df["pre_under_influence"].isin(
                ["tit", "iti", "o", "45", "lpsse"],
            )
        ]
        df = df[~df["pre_id_user"].astype(int).isin([2])]

        df = (
            df.clean_names(case_type="lower")
            .apply(lambda x: x.astype(str).str.lower() if (x.dtype == "object") else x)
            .replace(
                {
                    "pre_gender": {
                        "^(f|wom).*": "female",
                        "^m.*": "male",
                        "33|nan": "none",
                    },
                    "pre_language_other": {
                        "^(nil|none|noe|nan|)$": "none",
                        "^(English|english)$": "none",
                    },
                    "pre_vision_correction": {
                        ".*(normal|lasik|surgery|no|na|not|20|-|n/a).*": "none",
                        ".*contact.*": "contact",
                        ".*(glass|lens).*": "glasses",
                    },
                    "pre_under_influence": {
                        ".*(wine|adderal|beer|yes|weed|alcohol|vyvanse).*": "yes",
                        ".*(no|bo|n0|nan|zoloft|caff|pepsi|coffee|none|nicotine|n "
                        "o|well|good|not currently|glasses|na).*": "none",
                    },
                    "pre_medical_condition": {
                        "^(no|bo|n0|nan|n\\?a|never).*$": "none",
                        ".*anxiety disorder and depression.*": "anxiety disorder and "
                        "depression",
                        ".*anxiety$": "anxiety",
                        ".*diabetes.*": "diabetes",
                    },
                    "pre_reading_disability": {
                        "^(no|bo|n0|nan|n\\?a|never).*$": "none",
                        ".*do not.*": "none",
                    },
                },
                regex=True,
            )
            .fillna(
                {
                    "pre_language_other": "none",
                    "pre_vision_correction": "none",
                    "post_feedback": "none",
                    "pre_medical_condition": "none",
                    "pre_reading_disability": "none",
                    "pre_under_influence": "none",
                    "post_toggle_strategy": "none",
                    "pre_occupation": "none",
                    "pre_gender": "none",
                    "gender": "none",
                },
            )
        )

        df["pre_language_other"] = [
            1 if x != "none" else 0 for x in df["pre_language_other"]
        ]
        df["pre_language_native"] = [
            1 if "english" in x else 0 for x in df["pre_language_native"]
        ]

        df = df.drop(
            df.filter(
                regex="post_id_user|outlier|turk_id|timestamp",
            ).columns, axis=1,
        )

        # append dummy columns
        df = df[~df["pre_education_level"].isin(["prefer not to say"])]
        edu_map = {
            "less than high school": 0,
            "high school/ged": 1,
            "some college": 2,
            "associate's degree (2-years of college)": 3,
            "bachelor's degree (4-years of college)": 4,
            "professional degree": 5,
            "master's degree": 5,
            "doctoral degree": 6,
        }
        read_freq_map = {
            "less than once a month": 0,
            "once a month": 1,
            "once a week": 2,
            "2-3 times a week": 3,
            "everyday": 4,
        }

        df["pre_education_level"] = [
            edu_map[v]
            for v in df["pre_education_level"]
        ]

        # show relationship between age and user education
        # to make it more consistent, create right before outputting model data
        # drop education level afterwards since it's not used.
        df_age_edu = df[
            ["pre_age", "pre_education_level", "pre_id_user"]
        ].drop_duplicates()
        df_age_edu["pre_age_group"] = pd.qcut(df_age_edu["pre_age"], 4)
        sns.countplot(
            x="pre_age_group", hue="pre_education_level", data=df_age_edu,
        )
        plt.title("Age and College Degree")
        plt.savefig(f"{self.report_dir}/input/corr_age_education.png")

        # convert reading frequency to aggregate measures
        df["pre_reading4leisure_frequency"] = [
            read_freq_map[v] for v in df["pre_reading4leisure_frequency"]
        ]
        df["pre_reading4work_frequency"] = [
            read_freq_map[v] for v in df["pre_reading4work_frequency"]
        ]

        df["pre_reading_frequency"] = (
            df["pre_reading4leisure_frequency"] +
            df["pre_reading4work_frequency"]
        )

        df = df.drop(df.filter(regex="reading4", axis=1).columns, axis=1)
        # remove all read to children
        df = df.drop(
            df.filter(regex="pre_read_to_children", axis=1).columns, axis=1,
        )

        # dropping both because the device info is messy
        df = df.drop(
            df.filter(
                regex=r"pre_usual_working_device",
                axis=1,
            ).columns, axis=1,
        )
        df = df.drop(
            df.filter(
                regex="pre_usual_reading_device",
                axis=1,
            ).columns, axis=1,
        )

        # comfortable in english
        df = df[df["pre_reading_comfort"] == "very comfortable"]

        cat_cols = df.select_dtypes(include=object).columns

        # columns that require additional feature engineering
        feature_eng_cols = df[
            df[cat_cols].columns[df[cat_cols].describe().loc["unique", :] > 5]
        ].columns

        # option to keep font for performance
        cat_cols = set(cat_cols) - set(feature_eng_cols)

        df = pd.concat(
            [
                df.drop(list(cat_cols) + list(feature_eng_cols), axis=1),
                pd.get_dummies(
                    df[set(cat_cols)], prefix=set(cat_cols), prefix_sep=".",
                ),
            ],
            axis=1,
        )

        # filter those under influence
        df = df[df["pre_under_influence.none"] == 1]
        df = df.drop(
            df.filter(regex="pre_under_influence", axis=1).columns, axis=1,
        )

        df = df.drop(
            df.filter(regex="pre_normal_vision", axis=1).columns, axis=1,
        )

        # currently using desktop as baseline, exclude those using phones
        # df = df[df['pre_current_device.phone'] == 0]
        df = df.drop(
            df.filter(
                regex="pre_current_device.tablet",
                axis=1,
            ).columns, axis=1,
        )

        # no vision correction as baseline
        df = df.drop(
            df.filter(
                regex="pre_vision_correction.none",
                axis=1,
            ).columns, axis=1,
        )

        # female as baseline
        df = df.drop(
            df.filter(regex="pre_gender.female", axis=1).columns, axis=1,
        )

        # pre_education_level_less than high school as baseline
        df = df.drop(
            df.filter(
                regex="pre_education_level.less than high school", axis=1,
            ).columns,
            axis=1,
        )

        # other columns to drop
        df = df.drop(
            df.filter(regex="pre_reading_proficiency", axis=1).columns, axis=1,
        )

        df = df.drop(
            df.filter(regex="pre_reading_comfort", axis=1).columns, axis=1,
        )

        df = df.drop(
            df.filter(
                regex="pre_do_you_feel_your_reading_speed_could_be_improved_", axis=1,
            ).columns,
            axis=1,
        )

        df = df.drop(df.filter(regex="pre_gender", axis=1).columns, axis=1)

        df = df.drop(
            df.filter(regex="pre_current_device", axis=1).columns, axis=1,
        )

        return df

    def get_toggle_data(self, include_font_data=True):
        """
        Takes advantage of the existing toggle data
        and append font data to the winning and losing fonts respectively.
        """

        # adjustment factor
        df_toggle = self.df_toggle
        df_toggle["matchup"] = [
            ", ".join(
                sorted([df_toggle.loc[i, "font_a"], df_toggle.loc[i, "font_b"]]),
            )
            for i in range(len(df_toggle))
        ]

        # in cases of repeated matchup, last decision is used
        df_toggle = df_toggle.sort_values(["id_user", "timestamp"])
        df_toggle = df_toggle.drop_duplicates(
            ["id_user", "matchup"], keep="last",
        )

        # summarize data set
        df = (
            df_toggle.groupby(["id_user", "font_a", "font_b"])["winning_font"]
            .apply(lambda x: list(set(x)))
            .reset_index()
            .sort_values(["id_user", "font_a", "font_b"])
        )
        df = df[
            df["winning_font"].apply(
                lambda x: len(x) == 1,
            )
        ].reset_index(drop=True)
        df["win"] = df["winning_font"].apply(lambda x: x[0])
        df["lose"] = [
            df.loc[i, "font_b"]
            if df.loc[i, "font_a"] == df.loc[i, "win"]
            else df.loc[i, "font_a"]
            for i in range(len(df))
        ]

        df_time = (
            df_toggle.groupby(["id_user", "font_a", "font_b"])[
                "dwell_time_a", "dwell_time_b",
            ]
            .mean()
            .reset_index()
        )

        df = pd.merge(
            df, df_time, on=[
                "id_user", "font_a", "font_b",
            ], validate="1:1",
        )
        df["dwell_time_win"] = [
            df.loc[i, "dwell_time_a"]
            if df.loc[i, "win"] == df.loc[i, "font_a"]
            else df.loc[i, "dwell_time_b"]
            for i in range(len(df))
        ]
        df["dwell_time_lose"] = [
            df.loc[i, "dwell_time_a"]
            if df.loc[i, "lose"] == df.loc[i, "font_a"]
            else df.loc[i, "dwell_time_b"]
            for i in range(len(df))
        ]
        df["matchup"] = [
            ", ".join(sorted([df.loc[i, "font_a"], df.loc[i, "font_b"]]))
            for i in range(len(df))
        ]
        df = df.drop(
            ["font_a", "font_b", "dwell_time_a", "dwell_time_b", "winning_font"],
            axis=1,
        )

        # join font metrics data
        if not include_font_data:
            return df

        df_font = pd.DataFrame()
        for font in df["win"].unique():
            df_font = df_font.append(get_font_data(font))

        # convert panose features to dummies
        df_font = df_font.apply(lambda x: pd.to_numeric(x, errors="ignore"))
        df_font.to_csv("data/interim/df_font.csv")

        # join font data
        df["win"].unique()
        df = pd.merge(
            df, df_font.add_prefix("win_"), left_on="win", right_on="win_font",
        )
        df = pd.merge(
            df, df_font.add_prefix("lose_"), left_on="lose", right_on="lose_font",
        )

        return df

    def get_new_model_data(self):
        """Prepare model data, the first step before running any model
        This code generates a data frame object that includes all data needed for
        subsequent analyses
        """

        # # pre-post survey data #
        # remove users with invalid ids

        df_post = self.df_post[
            self.df_post["Please enter your Unique ID:"].apply(
                lambda x: len(x.split("_")),
            )
            == 5
        ]
        df_pre = self.df_pre[
            self.df_pre["Please enter your Unique ID:"].apply(
                lambda x: len(x.split("_")),
            )
            == 5
        ]

        df_pre["id_user"] = (
            df_pre["Please enter your Unique ID:"]
            .apply(lambda x: x.split("_")[4])
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        )
        df_pre = df_pre[~df_pre["id_user"].isna()]
        df_pre["id_user"] = df_pre["id_user"].astype(int)

        df_post["id_user"] = (
            df_post["Please enter your Unique ID:"]
            .apply(lambda x: x.split("_")[4])
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
        )
        df_post = df_post[~df_post["id_user"].isna()]
        df_post["id_user"] = df_post["id_user"].astype(int)

        df_user_full = self.get_survey_data(df_pre=df_pre, df_post=df_post)
        df_user_full.to_csv("reports/input/df_user.csv", index=False)
        df_user = df_user_full.filter(regex="^pre", axis=1)

        df_user = df_user.drop(
            ["pre_language_native", "pre_language_other", "pre_education_level"],
            axis=1,
        )

        # # user preference data #
        df_elo_wide = calculate_elo(self.df_toggle)
        df_elo_wide = pd.DataFrame.from_dict(df_elo_wide, orient="index")
        df_elo_wide = df_elo_wide.reset_index()
        df_elo_wide.columns = ["id_user"] + list(df_elo_wide.columns[1:])
        df_elo_long = df_elo_wide.melt(id_vars="id_user")
        df_elo_long.columns = ["id_user", "font", "elo"]

        # sorting expected behavior -- higher elo score should have
        # a better ranking (lower integer)
        df_elo_long["font_rank"] = (
            df_elo_long.groupby(["id_user"])["elo"].rank(
                ascending=False,
            ).astype(int)
        )

        # In our case, it may not be necessary to save. Direct passage between modules
        # would be safer
        df_elo_long.to_csv("data/interim/elo_transposed_new.csv", index=False)

        # # reading speed data #
        # obtain a subset of font data
        unique_fonts = set(
            self.df_toggle["font_a"].tolist(
            ) + self.df_toggle["font_b"].tolist(),
        )
        font_data_js_subset = font_data_js[
            font_data_js["font"].isin(
                unique_fonts,
            )
        ]

        # merge all data
        df_combined = self.df_wpm[self.df_wpm["avg_wpm"].between(100, 650)]
        df_combined = df_combined.merge(
            font_data_js_subset.add_prefix("toggle_"),
            left_on="font",
            right_on="toggle_font",
        )

        df_combined = df_combined.merge(
            df_user, left_on="id_user", right_on="pre_id_user",
        )
        df_combined = df_combined.merge(
            df_elo_long[["id_user", "font", "font_rank"]],
            left_on=["id_user", "toggle_font"],
            right_on=["id_user", "font"],
        )
        df_combined = df_combined.drop(["pre_id_user", "id_study"], axis=1)

        plt.close("all")
        df_combined["overall_error_rate"].hist()
        plt.title("Distribution of Comprehension Error Rate")
        plt.savefig(
            f"{self.report_dir}/input/{self.objective}_dist_error_rate.png",
        )

        # cleanup
        df_combined = df_combined[
            df_combined["overall_error_rate"]
            < self.error_thresh
        ]
        df_combined = df_combined.drop(
            ["font_x", "font_y", "overall_error_rate"], axis=1,
        )

        if self.objective == "performance":
            df_combined = df_combined.rename(
                columns={
                    "id_user": "toggle_id_user",
                    "fam_level": "toggle_fam_level",
                    "font_rank": "toggle_font_rank",
                    "avg_wpm": "outcome",
                },
            )
        else:
            df_combined = df_combined.drop("avg_wpm", axis=1)
            df_combined = df_combined.rename(
                columns={
                    "id_user": "toggle_id_user",
                    "fam_level": "toggle_fam_level",
                    "font_rank": "outcome",
                },
            )

        print(
            "--- # users left after filter:", df_combined["toggle_id_user"].nunique(
            ),
        )

        # outcome to range 0 to 1
        if self.objective == "performance":
            df_combined = df_combined.merge(
                df_combined.groupby(["toggle_id_user"])["outcome"]
                .max()
                .reset_index(name="max_wpm"),
                on="toggle_id_user",
            )
            df_combined = df_combined.merge(
                df_combined.groupby(["toggle_id_user"])["outcome"]
                .min()
                .reset_index(name="min_wpm"),
                on="toggle_id_user",
            )

            # outcome turn to binary relevancy label
            if self.label_type == "normalize":
                df_combined["outcome"] = (
                    df_combined.eval("(outcome-min_wpm)/(max_wpm-min_wpm)")
                    > self.wpm_threshold
                ).astype(int)
            elif self.label_type == "nonormalize":
                # approach 1: focus on users with more marginal benefits
                df_combined["outcome"] = (
                    df_combined["outcome"] /
                    df_combined["max_wpm"] > self.wpm_threshold
                ).astype(int)
            elif self.label_type == "graded":
                df_combined["outcome"] = (
                    10 * df_combined["outcome"] / df_combined["max_wpm"]
                ).astype(int)

            if self.preference_mode == "include":
                pass
            elif self.preference_mode == "exclude":
                df_combined = df_combined.drop("toggle_font_rank", axis=1)

            plt.close("all")
            plt.hist(df_combined["outcome"])
            plt.title("Distribution of Relevancy")
            plt.savefig(
                f"{self.report_dir}/input/dist_relevancy_outcome_labels.png",
            )

            df_combined = df_combined.drop(["max_wpm", "min_wpm"], axis=1)

        df_combined = pd.concat(
            [df_combined.drop("outcome", axis=1), df_combined[["outcome"]]], axis=1,
        )

        # dropping unnecessary or problematic columns
        df_combined = df_combined.drop(
            df_combined.filter(
                regex="gender|device_|rank_",
                axis=1,
            ).columns, axis=1,
        )

        df_combined = df_combined.drop(
            df_combined.filter(regex="_relative", axis=1).columns, axis=1,
        )

        df_user_full[
            df_user_full["pre_id_user"].isin(df_combined["toggle_id_user"])
        ].to_csv("reports/input/df_user.csv", index=False)

        plt.close("all")
        df_combined.hist(bins=20)
        plt.savefig(f"{self.report_dir}/input/train_full_hist.png")

        df_combined.to_csv(f"{self.report_dir}/input/train_full.csv")

        self.df_full = df_combined

    def prep_data_and_split(self):
        """Create k fold split

        Returns:
            [type]: [description]
        """

        if self.df_full is None:
            self.get_new_model_data()

        # train test split by group
        group_kfold = GroupKFold(n_splits=5)
        group_idx = self.df_full.groupby(GROUP_COL).ngroup()

        train_test_folds = group_kfold.split(
            self.df_full.drop(
                "outcome", axis=1,
            ), self.df_full["outcome"], group_idx,
        )

        return self.df_full, list(train_test_folds)


if __name__ == "__main__":
    data_obj = Data(
        report_dir="reports",
        wpm_threshold=0.95,
        label_type="nonormalize",
        preference_mode="include",
        objective="performance",
    )
    data_obj.get_new_model_data()
    df_full = data_obj.df_full
