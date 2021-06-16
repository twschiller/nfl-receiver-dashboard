# IMPORTS ----------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
import seaborn as sns
import statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess
import streamlit as st


_lock = RendererAgg.lock
plt.style.use('default')

# https://github.com/streamlit/release-demos/blob/master/0.65/demos/query_params.py
query_params = st.experimental_get_query_params()
default_player = query_params["player"][0] if "player" in query_params else ""


# SETUP ------------------------------------------------------------------------
st.set_page_config(page_title='Wide Receiver Dashboard',
                   page_icon='https://pbs.twimg.com/profile_images/'\
                             '1265092923588259841/LdwH0Ex1_400x400.jpg',
                   layout="wide")
# READ DATA --------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def get_pbp():

    #pbp data
    col_list1 = ['play_id','complete_pass','yards_gained','air_yards','touchdown',
                'epa','down','yardline_100','posteam','receiver','pass','play_type',
                'two_point_attempt','game_id','two_point_conv_result','fumble_lost',
                'week','rusher','receiver_id','success','punt_returner_player_name',
                'kickoff_returner_player_name','defteam']

    YEAR = 2020
    data_ = pd.read_csv('https://github.com/guga31bb/nflfastR-data/blob/master/data/' \
                         'play_by_play_' + str(YEAR) + '.csv.gz?raw=True',
                         compression='gzip', low_memory=False, usecols=col_list1)
    return data_

data = get_pbp()
#--------
@st.cache(allow_output_mutation=True)
def get_pd():
    #plyer data
    col_list2 = ['headshot_url','full_name', 'birth_date','height','weight',
                 'college','position','team']

    player_data_ = pd.read_csv('https://github.com/mrcaseb/nflfastR-roster/blob/'\
                                'master/data/seasons/roster_2020.csv?raw=True',
                                low_memory=False, usecols=col_list2)

    return player_data_

player_data = get_pd()
#--------

COLORS = {'ARI':'#97233F','ATL':'#A71930','BAL':'#241773','BUF':'#00338D',
          'CAR':'#0085CA','CHI':'#00143F','CIN':'#FB4F14','CLE':'#FB4F14',
          'DAL':'#7F9695','DEN':'#FB4F14','DET':'#046EB4','GB':'#2D5039',
          'HOU':'#C9243F','IND':'#003D79','JAX':'#136677','KC':'#CA2430',
          'LA':'#003594','LAC':'#2072BA','LV':'#343434','MIA':'#0091A0',
          'MIN':'#4F2E84','NE':'#0A2342','NO':'#A08A58','NYG':'#192E6C',
          'NYJ':'#203731','PHI':'#014A53','PIT':'#FFC20E','SEA':'#7AC142',
          'SF':'#C9243F','TB':'#D40909','TEN':'#4095D1','WAS':'#FFC20F'}

# CLEAN DATA -------------------------------------------------------------------

#clean air_yards
data['air_yards'] = (
    np.where(
    data['air_yards'] < -10,
    data['air_yards'].median(),
    data['air_yards'])
    )
#---------
#getting rid of suffixes
names = ['receiver','rusher', 'punt_returner_player_name',
         'kickoff_returner_player_name']

for i in names:
    data[i] = data[i].str.replace(" Sr.","")
    data[i] = data[i].str.replace(" Jr.","")
#---------
#filter data
df = data[
          (data['pass']==1) &
          (data['play_type']=='pass') &
          (data.two_point_attempt==0) &
          (data['epa'].isna()==False)
          ]

#weird aj brown fumble pruitt td recovery
df.at[32403, 'touchdown'] = 0
#---------
#fpts data
fantasy = data[
          (data.play_type.isin(['no_play','pass','run','punt','kickoff'])) &
          (data['epa'].isna()==False)
          ]
fantasy  = pd.get_dummies(fantasy, columns=['two_point_conv_result'])

fantasy['fpts_skill'] = (
    fantasy['yards_gained'] * 0.1 +
    fantasy['complete_pass'] * 1 +
    fantasy['touchdown'] * 6 +
    fantasy['two_point_conv_result_success'] * 2 +
    fantasy['fumble_lost'] * -2
    )

receiving_fpts = (fantasy.groupby(
    ['receiver','posteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'receiver':'player'}))

rushing_fpts = (fantasy.groupby(
    ['rusher','posteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'rusher':'player'}))

kr_fpts = (fantasy.groupby(
    ['kickoff_returner_player_name','posteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'kickoff_returner_player_name':'player'}))

#return team is defteam pn punts
punt_fpts = (fantasy.groupby(
    ['punt_returner_player_name','defteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'punt_returner_player_name':'player','defteam':'posteam'}))

fpts_skill = receiving_fpts.merge(
                rushing_fpts,on=['player','posteam','week'], how='outer'
                ).merge(
                    punt_fpts,on=['player','posteam','week'], how='outer'
                    ).merge(
                        kr_fpts,on=['player','posteam','week'], how='outer').fillna(0)

fpts_skill.columns = ['player','posteam','week','fpts_skill_x',
                      'fpts_skill_y','fpts_skill_z','fpts_skill_a']

fpts_skill['total_fpts'] = (
    fpts_skill['fpts_skill_x'] +
    fpts_skill['fpts_skill_y'] +
    fpts_skill['fpts_skill_z'] +
    fpts_skill['fpts_skill_a']
    )
#---------


st.write('Adapted from [Max Bolger\'s Streamlit app](https://share.streamlit.io/maxbolger/nfl-receiver-dashboard/main/receiver-dashboard.py)')


# ROW 2 ------------------------------------------------------------------------

name_col, week_col = st.beta_columns(2)

with name_col:
    options_p = df.groupby(['receiver','posteam'])[['play_id']].count().reset_index()
    options_p = options_p.loc[options_p.play_id>29]
    player_list = options_p['receiver'].to_list()

    player_data['pbp_name'] = [item[0] + '.'+ ''.join(item.split()[1:]) for item in player_data['full_name']]
    pd_filt = player_data.loc[(player_data['pbp_name'].isin(player_list)) &
                          ((player_data['position'] == 'WR') |
                          (player_data['position'] == 'TE'))]
    #Hardcode common nflfastR names due to id issues
    remove = ['Josh Smith','Jerome Washington','D.J. Montgomery','Jaron Brown',
              'Jalen Williams','Jaeden Graham','Hunter Bryant','Connor Davis',
              'Mike Thomas','Maxx Williams','Joe Reed','Marvin Hall','Ito Smith']

    pd_filt = pd_filt.loc[~pd_filt.full_name.isin(remove)]
    player_list = pd_filt['pbp_name'].to_list()
    pd_filt = pd_filt.filter(
                ['full_name','pbp_name','team']
                    ).dropna().reset_index(drop=True).sort_values('full_name')
    records = pd_filt.to_dict('records')

    if default_player:
        default_record_index = next((i for i, x in enumerate(records) if x["full_name"] == default_player), None)
    else:
        default_record_index = None
    
    selected_data = st.selectbox(
        'Select a Player',
        options=records,
        index=default_record_index or 0,
        format_func=lambda record: f'{record["full_name"]}'
    )

    player = selected_data.get('pbp_name')
    team = selected_data.get('team')


with week_col:
    start_week, stop_week = st.select_slider(
    'Select A Range of Weeks',
    options=list(range(1,22)),
    value=(1,21))



# ROW 2 ------------------------------------------------------------------------
st.write('')

def air_yards(player, team):
  '''
  This function returns an ay dist for the desired wr
  '''
  receiver=df.loc[(df.receiver==player) & (df.posteam==team) &
                  (df.week>= start_week) & (df.week<= stop_week)]

  fig1 = Figure()
  ax = fig1.subplots()
  sns.kdeplot(data=df['air_yards'], color='#CCCCCC',
                fill=True, label='NFL Average',ax=ax)
  sns.kdeplot(data=receiver['air_yards'], color=COLORS.get(team),
                fill=True, label=player,ax=ax)
  ax.legend()
  ax.set_xlabel('Air Yards', fontsize=12)
  ax.set_ylabel('Density', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  ax.set_xlim([-10,55])
  st.pyplot(fig1)

with _lock:
    st.subheader('Air Yards Distribution')
    air_yards(player,team)


def epa_chart(player, team):
    '''
    This function returns epa chart
    '''

    week_filter = df.loc[(df.week>= start_week) & (df.week<= stop_week) &
                         (df.receiver.isin(player_list))]
    epa = week_filter.groupby(['receiver','posteam']).agg(
            {'success':'mean','epa':'mean','play_id':'count'}
            ).reset_index()
    #error handling similar names due to id issues
    epa = epa.loc[~((epa['receiver'] == 'I.Smith') & (epa['posteam'] == 'ATL')) &
                  ~((epa['receiver'] == 'M.Brown') & (epa['posteam'] == 'LAR')) &
                  ~((epa['receiver'] == 'D.Johnson') & (epa['posteam'] == 'HOU'))]

    tgt_filt = stop_week - start_week
    epa = epa.loc[epa.play_id>tgt_filt]
    epa['color'] = '#EFEFEF'
    epa.loc[(epa.receiver==player) & (epa.posteam==team),'color'] = COLORS.get(team)
    epa = epa.sort_values(by='color',ascending=False)
    fig5 = Figure()
    ax = fig5.subplots()

    sns.scatterplot(x=epa.success, y=epa.epa,data=epa,
    color=epa.color,s=(epa.play_id * 2),ax=ax)

    ax.axhline(y=epa.epa.mean(),linestyle='--',color='black',alpha=0.2)
    ax.axvline(x=epa.success.mean(),linestyle='--',color='black',alpha=0.2)
    # ax.get_legend().remove()
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('EPA/Target', fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    st.pyplot(fig5)

with _lock:
    st.subheader('EPA/Target')
    epa_chart(player, team)
