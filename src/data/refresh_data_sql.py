from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine

# get toggle data
connection_string = os.environ['DATABASE_CONNECTION']
engine = create_engine(connection_string)

df_toggle = pd.read_sql(
    """
    SELECT *
    FROM reading.choices
    WHERE id_study IN (
        select id_study from reading.study where `end` IS NOT NULL and
        url =
        'ml_preference_effectiveness')
        and study_step = 'normal'
    """,
    engine,
)
df_toggle.to_csv('data/processed/new_df_toggle.csv', index=False)

# get wpm table
df_wpm = pd.read_sql(
    """
     with wpm as (
        select id_user, font, avg(wpm) as avg_wpm
        from (
                 SELECT *
                 FROM reading.speed
                 WHERE id_study IN (
                     select id_study
                     from reading.study
                     where `end` IS NOT NULL
                       and url = 'ml_preference_effectiveness')
                   AND font != 'comic_sans'
                   and study_step = 'normal'
                   and wpm >=100 and wpm <= 650
             ) base
        group by id_user, font
    ),
         comp as (
             select id_user,
                    sum(if(correct = 0, 1, 0))
                                                                 as
                                                                 n_errors,
                    count(distinct question)
                                                                 as
                                                                 n_questions,
                    sum(if(correct = 0, 1, 0)) / count(distinct
                                                       question) as
                                                                    overall_error_rate
             from (
                      select *
                      from reading.comprehension
                      where id_study in (
                          select id_study
                          from reading.study
                          where `end` is not null
                            and url = 'ml_preference_effectiveness'
                            and user_type = 'mturk')
                        and font != 'comic_sans'
                        and study_step = 'normal'
                        and question != 'Beethoven&#39;s first meeting
                        with Mozart was in:'
                  ) base
             group by id_user
         ),
         fam as (
             SELECT id_user, id_study, font, answer_num
             FROM reading.fontfamiliarity
             WHERE id_study IN (
                 select id_study
                 from reading.study
                 where `end` IS NOT NULL
                   and url = 'ml_preference_effectiveness')
         )
    select fam.id_study,
           fam.id_user,
           fam.font,
           answer_num as fam_level,
           avg_wpm,
           overall_error_rate
    from fam
             inner join wpm on wpm.id_user = fam.id_user and
                               wpm.font = fam.font
             inner join comp on comp.id_user = fam.id_user
   """,
    engine,
)
df_wpm.to_csv('data/processed/new_df_wpm.csv', index=False)
