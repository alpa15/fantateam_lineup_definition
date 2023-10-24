# LIBRARIES
from lineup_definition import *
from keras.models import load_model

#####################################################
### BEGIN OF PARAMETERS TO BE DEFINED BY THE USER ###
#####################################################

## DATA TO BE SAVED ##
# User defines if the dataframes with players votes and with day charts
# should be updated or not
obtain_players_day_votes_df = False
obtain_day_charts_df = False
obtain_merged_df = False
url_players_votes = 'https://fantacalcio.it/voti-fantacalcio-serie-a/'
url_day_charts = ('https://www.transfermarkt.it/serie-a/'
                'spieltagtabelle/wettbewerb/IT1?saison_id=&spieltag=')
url_prob_lineups = 'https://www.fantacalcio.it/probabili-formazioni-serie-a'
#url_prob_lineups_1 = 'https://app.fantaformazione.com/probabili-match/32/4'
#url_prob_lineups_2 = 'https://fantamaster.it/probabili-formazioni-serie-a-live'
first_year_train = 2021 #2015
last_year_train = 2023
first_day_train = 1
last_day_train = 5 #38

## MODEL PARAMETERS ##
# User defines the model chosen and if the model should be trained
model_type = 'nn' # Options: ['nn', 'xgb']
model_train_tf = True
input_variables = ['avg_votes', 'std_votes',
            'avg_votes_last4', 'std_votes_last4',
            'avg_fantavotes', 'std_fantavotes',
            'avg_fantavotes_last4', 'std_fantavotes_last4',
            'avg_points', 'std_points',
            'avg_points_last4', 'std_points_last4',
            'avg_goals_scored', 'std_goals_scored',
            'avg_goals_scored_last4', 'std_goals_scored_last4',
            'avg_goals_conceded', 'std_goals_conceded',
            'avg_goals_conceded_last4', 'std_goals_conceded_last4',
            'avg_opponents_points', 'std_opponents_points',
            'avg_opponents_points_last4', 'std_opponents_points_last4',
            'avg_opponents_goals_scored', 'std_opponents_goals_scored',
            'avg_opponents_goals_scored_last4', 'std_opponents_goals_scored_last4',
            'avg_opponents_goals_conceded', 'std_opponents_goals_conceded',
            'avg_opponents_goals_conceded_last4', 'std_opponents_goals_conceded_last4']
output_variable = 'fantavote_current_day'
# 'model_params' options:
# For 'nn': []
# For 'xgb': ['booster', 'silent', 'nthread', 'eta', 'max_depth', 'subsample'
#             'colsample_bytree', 'objective', 'eval_metric', 'num_class']
if model_type == 'xgb':
    model_params = {
        'booster': 'gbtree',
        'silent': 0,
        'nthread': 4,
        'eta': 0.1,
        'max_depth': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'num_class': 1
        }
elif model_type == 'nn':
    model_params = {
        'norm_type': 'min-max',
        'params_for_norm': ''
    }
else:
    raise NameError
save_model_out = False

## FANTATEAM PLAYERS LIST ##
# User definition of the name and role of every element of his own fantateam
players = {'mandas': 'GLK', 'provedel': 'GLK', 'sepe': 'GLK',
          'biraghi': 'DEF', 'bremer': 'DEF', 'd\'ambrosio': 'DEF',
          'darmian': 'DEF', 'juan jesus': 'DEF', 'mario rui': 'DEF',
          'marusic': 'DEF', 'schuurs': 'DEF',
          'adopo': 'MID', 'ikone\'': 'MID', 'koopmeiners': 'MID',
          'lazovic': 'MID', 'messias': 'MID', 'orsolini': 'MID',
          'politano': 'MID', 'zaccagni': 'MID',
          'bonazzoli': 'STK', 'chiesa': 'STK', 'colombo': 'STK',
          'immobile': 'STK', 'kouame\'': 'STK', 'piccoli': 'STK'}

## PREDICTIONS INFO
threshold_days = 4 # Usually 4, but for the first days it's better to lower it
current_year = 2023
first_day_pred = 1
current_day = 10
year = 2023
save_pred_out = True
###################################################
### END OF PARAMETERS TO BE DEFINED BY THE USER ###
###################################################


## DAY_VOTES_DF ##
day_votes_df_path = 'Fantateam lineup definition//day_votes_df.csv'
if os.path.exists(day_votes_df_path):
    if obtain_players_day_votes_df:
        day_votes_df = []
        url_players_votes_train = url_players_votes
    else:
        day_votes_df = pd.read_csv('Fantateam lineup definition//day_votes_df.csv')
        url_players_votes_train = ''
else:
    day_votes_df = []

## DAY_CHARTS_DF ##
day_charts_df_path = 'Fantateam lineup definition//day_charts_df.csv'
if os.path.exists(day_charts_df_path):
    if obtain_day_charts_df:
        day_charts_df = []
    else:
        day_charts_df = pd.read_csv('Fantateam lineup definition//' +
                                    'day_charts_df.csv')
        url_day_charts_train = url_day_charts
else:
    day_charts_df = []

## MERGED_DF ##
if obtain_merged_df:
    merged_df = ''
else:
    merged_df = pd.read_csv('Fantateam lineup definition//' +
                                    'merged_df.csv')

## MODEL TRAINING ##
if model_train_tf:
    training_data = obtain_training_data(url_players_votes_train,
                                day_votes_df, url_day_charts_train,
                                day_charts_df, first_year_train,
                                last_year_train, first_day_train,
                                last_day_train, obtain_players_day_votes_df,
                                obtain_day_charts_df, obtain_merged_df)
    train_model(training_data, input_variables, output_variable,
                model_type, model_params, save_model_out)
else:
    if model_type == 'nn':
        model = load_model('Fantateam lineup definition//models//nn_model.h5')
    elif model_type == 'xgb':
        model = xgb.Booster()
        model.load_model('Fantateam lineup definition//models//xgb_model.bin')

print(players.keys())

## PLAYERS DAILY FANTAVOTES PREDICTION ##
day_predicted_vote = predict_votes(players.keys(), url_players_votes,
                                    url_day_charts, first_day_pred,
                                    model, current_day, current_year,
                                    threshold_days, input_variables,
                                    save_pred_out)

prob_lineups_df = extract_prob_lineups_data(url_prob_lineups,
                                            players.keys())

votes_prob_df = pd.merge(day_predicted_vote, prob_lineups_df,
                         on=['player_name'])

votes_prob_df.loc[votes_prob_df['player_name'] == 'immobile',
                  'prob_of_play'] = 1
votes_prob_df.loc[votes_prob_df['player_name'] == 'chiesa',
                  'prob_of_play'] = 1

for i in range(0, len(votes_prob_df)):
    if votes_prob_df.loc[i, 'prob_of_play'] > 0:
        votes_prob_df.loc[i, 'prob_of_play'] = 1
    else:
        votes_prob_df.loc[i, 'prob_of_play'] = 0

votes_prob_df['score'] = votes_prob_df['predicted_fantavote'
                            ].multiply(votes_prob_df['prob_of_play'])

## PLAYERS LINEUP DEFINITION ##
scores = {}
for player in players.keys():
    player_row = votes_prob_df[votes_prob_df['player_name'] == player]
    if len(player_row) != 0:
        scores[player] = player_row['predicted_fantavote']
    else:
        scores[player] = 0
solution = solve_optimization_problem(scores, players)

print(votes_prob_df.sort_values(by=['predicted_fantavote'], ascending=False
                                ).reset_index().drop(columns=['index']))