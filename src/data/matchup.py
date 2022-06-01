from __future__ import annotations

import choix
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import trueskill
from sklearn.metrics import adjusted_mutual_info_score

tmp_df = pd.read_csv('data/processed/new_df_toggle.csv')
fonts = list(set(list(tmp_df['font_a']) + list(tmp_df['font_b'])))
n_fonts = len(fonts)


def rank_win_ratio(df_sum):
    # get toggle data
    from src.data import survey_data
    df_toggle = survey_data.get_toggle_data()

    df_sum_matchup = df_sum.groupby(['matchup', 'win', 'lose']).agg({
        'id_user':
            'count',
        'dwell_time_win':
            'mean',
        'dwell_time_lose':
            'mean',
    }).reset_index()

    df_sum_matchup_overall = df_sum.groupby(
        'matchup',
    )['id_user'].count().reset_index(name='encounters')
    df_sum_matchup = df_sum_matchup.merge(
        df_sum_matchup_overall,
        on='matchup',
        validate='m:1',
    )
    df_sum_matchup[
        'win_ratio'
    ] = df_sum_matchup['id_user'] / df_sum_matchup['encounters']
    df_sum_matchup_pivot = df_sum_matchup.pivot(
        index='win',
        columns='lose',
    )['win_ratio']

    plt.pcolor(df_sum_matchup_pivot)
    plt.yticks(
        np.arange(0.5, len(df_sum_matchup_pivot.index), 1),
        df_sum_matchup_pivot.index,
    )
    plt.xticks(
        np.arange(0.5, len(df_sum_matchup_pivot.columns), 1),
        df_sum_matchup_pivot.columns,
        rotation=90,
    )
    plt.ylabel('Winning fonts')
    plt.xlabel('Losing fonts')
    plt.title('Distribution of the Proportion of Times a font Wins')
    for i in range(len(fonts)):
        for j in range(len(fonts)):
            plt.text(
                j + .5,
                i + .5,
                int(100 * np.nan_to_num(df_sum_matchup_pivot.iloc[i, j])),
                ha='center',
                va='center',
                color='w',
            )
    plt.colorbar()
    plt.savefig('reports/exploratory_analysis/eda_win_ratio_heatmap.png')

    # distribution of win ratio for each font
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    for i in range(n_fonts):
        axes[i // 4, i % 4].bar(
            df_sum_matchup_pivot.loc[fonts[i], :].index,
            df_sum_matchup_pivot.loc[fonts[i], :].values,
        )
        axes[i // 4, i % 4].set_title(fonts[i])
        axes[i // 4, i % 4].axhline(0.5, linestyle='--')
        axes[i // 4, i % 4].tick_params(labelrotation=90)

    # distribution of dwell time

    divergence_df = pd.DataFrame(index=fonts, columns=fonts)
    for j in range(n_fonts):
        tmp_df_sum = df_sum[df_sum['matchup'].str.contains(fonts[j])]
        tmp_df_sum = tmp_df_sum[
            tmp_df_sum['dwell_time_win'].between(
                np.percentile(tmp_df_sum['dwell_time_win'], 5),
                np.percentile(tmp_df_sum['dwell_time_win'], 95),
            )
        ]
        tmp_df_sum = tmp_df_sum[
            tmp_df_sum['dwell_time_lose'].between(
                np.percentile(tmp_df_sum['dwell_time_lose'], 5),
                np.percentile(tmp_df_sum['dwell_time_lose'], 95),
            )
        ]

        # plot
        fig_dwell, axes_dwell = plt.subplots(
            4,
            4,
            figsize=(16, 12),
            sharex=True,
            sharey=True,
        )
        fig_dwell.suptitle(f'{fonts[j]}', fontsize=16, x=.5, weight='bold')

        for i in range(n_fonts):
            comp_font = fonts[i]
            winning_dwell_time = tmp_df_sum[
                tmp_df_sum['win'] ==
                comp_font
            ]['dwell_time_win']
            losing_dwell_time = tmp_df_sum[
                tmp_df_sum['lose'] ==
                comp_font
            ]['dwell_time_lose']
            sns.distplot(
                winning_dwell_time,
                ax=axes_dwell[i // 4, i % 4],
                label='win',
            )
            sns.distplot(
                losing_dwell_time,
                ax=axes_dwell[i // 4, i % 4],
                label='lose',
            )

            axes_dwell[i // 4, i % 4].set_title(fonts[i])
            axes_dwell[i // 4, i % 4].set_xlabel('')

            # compare the win rate
            min_len = min(len(winning_dwell_time), len(losing_dwell_time))
            divergence_df.loc[fonts[j], fonts[i]] = adjusted_mutual_info_score(
                winning_dwell_time.sample(min_len),
                losing_dwell_time.sample(min_len),
            )

        handles, labels = axes_dwell[i // 4, i % 4].get_legend_handles_labels()
        fig_dwell.legend(handles, labels, loc='lower right')
        plt.savefig(
            f'reports/exploratory_analysis/eda_dwell_time_{fonts[j]}.png',
        )
        plt.clf()
        plt.cla()

    time_df = np.concatenate((
        df_toggle[['win', 'dwell_time_win']].values,
        df_toggle[['lose', 'dwell_time_lose']].values,
    ))
    time_df = pd.DataFrame(time_df, columns=['font', 'time'])
    time_df['time'] = time_df['time'].astype(float)
    for f in fonts:
        sns.distplot(
            time_df[(time_df['font'] == f) & time_df['time'].between(
                np.percentile(time_df['time'], 5), np.percentile(
                    time_df['time'], 95,
                ),
            )]['time'],
            kde=True,
            hist=False,
            label=f,
        )
    plt.legend()
    plt.savefig('reports/exploratory_analysis/eda_win_duration_dist.png')
    return df_sum_matchup


def rank_true_skill(df_sum):
    # store the updated scores
    skill_dict = {}
    rating_dict = skill_dict.copy()
    for f in set(list(df_sum['win']) + list(df_sum['lose'])):
        skill_dict[f] = trueskill.Rating()
        rating_dict[f] = []

    for i in range(len(df_sum)):
        win = df_sum.loc[i, 'win']
        lose = df_sum.loc[i, 'lose']
        skill_dict[win], skill_dict[lose] = trueskill.rate_1vs1(
            skill_dict[win], skill_dict[lose],
        )
        rating_dict[win].append(skill_dict[win].mu)
        rating_dict[lose].append(skill_dict[lose].mu)

    # plot rating history to check convergence
    rating_history_array = []
    for x in rating_dict.values():
        rating_history_array.append(np.array(list(x)))
    max_len = max(len(x) for x in rating_history_array)
    rating_history_array = [
        np.pad(x, (0, max_len - len(x))) for x in rating_history_array
    ]
    rating_history_array = np.stack(rating_history_array, axis=0)
    rating_history_array[rating_history_array == 0] = np.nan

    plt.figure()
    plt.plot(rating_history_array.T)
    plt.legend(list(rating_dict.keys()), loc='upper left')
    plt.xlabel('iterations')
    plt.ylabel('mean score')
    plt.title('Rate of Convergence of Global TrueSkill Score')
    plt.savefig('reports/exploratory_analysis/eda_trueskill_convergence.png')

    # sorted scores
    rating_df = pd.DataFrame()
    rating_df['font'] = skill_dict.keys()
    rating_df['mu'] = [skill_dict[k].mu for k in skill_dict.keys()]
    rating_df['sigma'] = [skill_dict[k].sigma for k in skill_dict.keys()]
    rating_df = rating_df.sort_values('mu', ascending=False)

    return rating_df


# Preference Probability #
def rank_bradley_terry(df_sum):
    font2idx = {fonts[i]: i for i in range(n_fonts)}
    win_fonts_idx = df_sum['win'].apply(lambda x: font2idx[x])
    lose_fonts_idx = df_sum['lose'].apply(lambda x: font2idx[x])
    font_comparison = choix.lsr_pairwise(
        n_fonts, list(zip(win_fonts_idx, lose_fonts_idx)),
    )
    preference_prob = choix.probabilities(
        list(range(n_fonts)),
        font_comparison,
    )
    preference_prob = list(zip(fonts, preference_prob))
    preference_prob = pd.DataFrame(
        preference_prob,
        columns=[
            'fonts', 'prob',
        ],
    ).sort_values(
        'prob',
        ascending=False,
    )
    return preference_prob


# Graph – font ranking #
# create ranking data frame for all fonts
def rank_graph(df_sum):
    font_ranking_df = pd.DataFrame()
    for user in np.unique(df_sum['id_user']):
        user_pref_df = df_sum[df_sum['id_user'] == user].drop_duplicates(
            ['win', 'lose'],
        )

        user_graph = nx.from_pandas_edgelist(
            user_pref_df[['win', 'lose']],
            'win',
            'lose',
            create_using=nx.DiGraph,
        )

        leaves = [
            node for node in user_graph if user_graph.out_degree(node) == 0
        ]

        # find "source node" with the least number of edges
        for n_edges in range(n_fonts):
            if sum(
                user_graph.in_degree(node) == n_edges for node in user_graph
            ) == 0:
                continue

            path_data = [
                list(nx.all_simple_paths(user_graph, node, leaf))
                for node in user_graph if user_graph.in_degree(node) == n_edges
                for leaf in leaves
            ]
            break

        if len(path_data) == 0:
            continue

        flat_list = sorted(item for sublist in path_data for item in sublist)
        font_hierarchy = {'id_user': user}
        for f in fonts:
            idx_list = [
                font_list.index(f) for font_list in flat_list if f in font_list
            ]
            if len(idx_list) > 0:
                font_hierarchy[f] = max(
                    l.index(f) for l in flat_list if f in l
                )

        font_ranking_df = font_ranking_df.append(
            pd.DataFrame.from_dict(font_hierarchy, orient='index').T,
        )

    # missing is caused by user not going through the full list of fonts
    font_ranking_df = font_ranking_df.dropna()

    return font_ranking_df


if __name__ == '__main__':
    from src.data import survey_data

    df_sum = survey_data.get_toggle_data()
    df_graph = rank_graph(df_sum)
    df_bradley_terry = rank_bradley_terry(df_sum)
    df_true_skill = rank_true_skill(df_sum)
