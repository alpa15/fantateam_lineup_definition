# LIBRARIES
import numpy as np
import ast
import requests
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode
from pulp import *
from models import *

# GOAL: Starting from the url of the most probable lineups of the next
#       Serie A day, the probability of playing of all the 25 players
#       of the fantafootball team are svaed into a dataframe to be used
#       as the final multiplier to decide if a players should be inserted
#       in the day fantafootball lineup or not
def extract_prob_lineups_data(url, players_list):
    # Send a GET request to the URL and open save the web page content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the content of interest of the page
    names_list = soup.find_all('div', {'class': 'row col-sm'})
    probs_list = soup.find_all('div', {'class': 'row col-sm'})
    names = list()
    probs = list()
    assert len(names_list) == len(probs_list), """
        Error: Lists must have the same length.
        List 1 length: {}
        List 2 length: {}
        """.format(len(names_list), len(probs_list))
    # Loop to pass all the 10 matches in order to find all the 25 players
    for idx in range(0,len(names_list)):
        match_names = names_list[idx].find_all('a', {
            'class':'player-name'})
        match_probs = probs_list[idx].find_all('div', {
            'class':'progress-value'})
        # Every player of the selected match is evaluated
        for name, prob in zip(match_names, match_probs):
            name = unidecode(str(name).split('span')[1].replace(
                ">", "").replace("</", "")).lower()
            prob = int(str(prob).split('\n')[1].replace("%", ""))/100
            # If the player is in the list of the 25, he is appended
            if name in players_list:
                names.append(name)
                probs.append(prob)
    # The missing players are considered as injuried, so they are
    # associated to a probability of playing of 0
    missing_names = [x for x in players_list if x not in names]
    missing_probs = [0]*len(missing_names)
    names.extend(missing_names)
    probs.extend(missing_probs)
    # The dataframe is built and passed as output
    df = pd.DataFrame({'player_name': names, 'prob_of_play': probs})
    return df

# GOAL: Function which is called by 'extract_historical_votes_data' and
#       by 'extract_day_charts_data' in order to return averages and
#       standard deviations to be added to the respective
def obtain_avg_std(values, window, last = False):
    s = pd.Series(values)
    rolling_avg = s.rolling(window=window).mean()
    rolling_std = s.rolling(window=window).std()
    if last:
        rolling_avg = rolling_avg[len(rolling_avg)-1]
        rolling_std = rolling_std[len(rolling_std)-1]
    return rolling_avg, rolling_std

# GOAL: Starting from the url which contains all the votes and fantavotes
#       of every Serie A player, the dataframe with all the information
#       related to every player are saved
def extract_historical_votes_data(url, first_season, last_season, first_day, last_day,
                                  threshold_days, df_votes = [], df_day_votes = [],
                                  data_for_predict = False, save_out = False):
    if url == '':
        return df_votes, df_day_votes
    votes_df = pd.DataFrame(columns=['name', 'team', 'days_played',
                                     'votes_list', 'fantavotes_list',
                                     'season'])
    day_votes_df = pd.DataFrame(columns=['name', 'team', 'season', 'day',
                                    'avg_votes', 'std_votes',
                                    'avg_votes_last4', 'std_votes_last4',
                                    'avg_fantavotes', 'std_fantavotes',
                                    'avg_fantavotes_last4', 'std_fantavotes_last4',
                                    'fantavote_current_day'])
    seasons_no = last_season - first_season
    for i in range(0,seasons_no):
        analysed_season = first_season + i
        print("Currently analysed season: " + str(analysed_season))
        if data_for_predict:
            last = last_day
        else:
            last = last_day + 1
        for day in range(first_day,last):
            print("Currently analysed day: " + str(day))
            url_to_add = (str(analysed_season) + '-' +
                        str(analysed_season+1)[2:] + '/' + str(day))
            url_to_analyse = url + url_to_add
            # Send a GET request to the URL and open save the web page content
            response = requests.get(url_to_analyse)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find the content of interest of the page
            all_teams = soup.find_all('li', 
                                {'class': 'team-table'})
            for k in range(0, len(all_teams)):
                team_name = unidecode(all_teams[k].find_all('a',
                                    {'class': 'team-name team-link'}
                                    )[0].find_all('meta')[0].attrs['content']
                                    ).lower()
                players_names = all_teams[k].find_all('a', 
                                    {'class': 'player-name player-link'})
                all_votes = all_teams[k].find_all('span',
                                    {'class': 'player-grade'})[
                                        :len(players_names)*3]
                all_fanta_votes = all_teams[k].find_all('span',
                                    {'class': 'player-fanta-grade'})[
                                        :len(players_names)*3]
                for p in range(0,len(players_names)):
                    # Add to the dataframe all the information
                    player_name = unidecode(str(players_names[p].find_all('span')[0
                                    ]).split('>')[1].split('<')[0]).lower()
                    # Select the votes and the fantavotes
                    assert len(all_votes) == len(all_fanta_votes), """
                        Error: Lists must have the same length.
                        All votes list length: {}
                        All fantavotes list length: {}
                        """.format(len(all_votes), len(all_fanta_votes))
                    vote = float(str(all_votes[p*3+2]).split('data-value=\"'
                                )[1].split('\"')[0].replace(',', '.'))
                    fanta_vote = float(str(all_fanta_votes[p*3+2]).split(
                    'data-value=\"')[1].split('\"')[0].replace(',', '.'))
                    ## HARDCODED
                    if int(vote) == 55: # and player_name != "kouame\'":
                        continue
                    #if player_name == "kouame\'" and day == 6:
                    #    vote = 6
                    #    fanta_vote = 6
                    ## HARDCODED
                    if (len(votes_df[votes_df['name'] == player_name]) > 0):
                        if analysed_season in (votes_df[votes_df['name'
                                ] == player_name]['season'].tolist()):
                            # Add elements to the existing row
                            votes_df.at[np.where(votes_df['name'] == 
                                    player_name)[0][-1], 'days_played'
                                    ].append(day)
                            votes_df.at[np.where(votes_df['name'] == 
                                    player_name)[0][-1], 'votes_list'
                                    ].append(vote)
                            votes_df.at[np.where(votes_df['name'] == 
                                    player_name)[0][-1], 'fantavotes_list'
                                    ].append(fanta_vote)
                        else:
                            # Create the row
                            votes_df.loc[len(votes_df), 'name'
                                         ] = player_name
                            votes_df.loc[len(votes_df)-1, 'team'
                                         ] = team_name
                            votes_df.loc[len(votes_df)-1, 'days_played'
                                         ] = [day]
                            votes_df.loc[len(votes_df)-1, 'votes_list'
                                         ] = [vote]
                            votes_df.loc[len(votes_df)-1, 'fantavotes_list'
                                         ] = [fanta_vote]
                            votes_df.loc[len(votes_df)-1, 'season'
                                         ] = analysed_season
                    else:
                        # Create the row
                        votes_df.loc[len(votes_df), 'name'
                                         ] = player_name
                        votes_df.loc[len(votes_df)-1, 'team'
                                         ] = team_name
                        votes_df.loc[len(votes_df)-1, 'days_played'
                                         ] = [day]
                        votes_df.loc[len(votes_df)-1, 'votes_list'
                                        ] = [vote]
                        votes_df.loc[len(votes_df)-1, 'fantavotes_list'
                                        ] = [fanta_vote]
                        votes_df.loc[len(votes_df)-1, 'season'
                                        ] = analysed_season
                        
                    # It is necessary to have data of at least 5 days (so
                    # data of 4 days complete) in order to calculate the
                    # averages and the standard deviations
                    if day >= threshold_days+1:
                        ## PLAYER INPUTS
                        # The row of the currently analysed team is selected
                        # and the averages and standard deviations are calculated
                        player_row = (votes_df.loc[(votes_df['name'] == player_name) &
                                    (votes_df['season'] == analysed_season)]).copy()
                        ## VOTES
                        player_votes_list = player_row['votes_list'].tolist()[0]
                        if data_for_predict == False:
                            player_votes_list = player_votes_list[:-1]
                        if len(player_votes_list) < threshold_days:
                            continue
                        avg_votes_player, std_votes_player = obtain_avg_std(
                            player_votes_list,
                            len(player_votes_list), True)
                        avg_votes_last4_player, std_votes_last4_player = obtain_avg_std(
                            player_votes_list, threshold_days, True)
                        ## FANTAVOTES
                        player_fantavotes_list = player_row['fantavotes_list'].tolist()[0]
                        if data_for_predict == False:
                            player_fantavotes_list = player_fantavotes_list[:-1]
                        avg_fantavotes_player, std_fantavotes_player = obtain_avg_std(
                            player_fantavotes_list,
                            len(player_fantavotes_list), True)
                        avg_fantavotes_last4_player, std_fantavotes_last4_player = obtain_avg_std(
                            player_fantavotes_list, threshold_days, True)
                        
                        ## DAY_CHARTS_DF OWN TEAM DATA INSERTION
                        day_votes_df.loc[len(day_votes_df), 'name'] = player_name
                        day_votes_df.loc[len(day_votes_df)-1, 'team'] = team_name
                        day_votes_df.loc[len(day_votes_df)-1, 'season'] = analysed_season
                        day_votes_df.loc[len(day_votes_df)-1, 'day'] = day
                        day_votes_df.loc[len(day_votes_df)-1,
                            'avg_votes'] = avg_votes_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'std_votes'] = std_votes_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'avg_votes_last4'] = avg_votes_last4_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'std_votes_last4'] = std_votes_last4_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'avg_fantavotes'] = avg_fantavotes_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'std_fantavotes'] = std_fantavotes_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'avg_fantavotes_last4'] = avg_fantavotes_last4_player
                        day_votes_df.loc[len(day_votes_df)-1,
                            'std_fantavotes_last4'] = std_fantavotes_last4_player
                        if data_for_predict == False:
                            day_votes_df.loc[len(day_votes_df)-1,
                                'fantavote_current_day'] = player_row['votes_list'].tolist()[0][-1]
    if data_for_predict:
        day_votes_df = day_votes_df.drop(columns=['fantavote_current_day'])
    if save_out:
        votes_df.to_csv('Fantateam lineup definition//' +
                        'votes_df.csv', index=False)
        day_votes_df.to_csv('Fantateam lineup definition//' +
                        'day_votes_df.csv', index=False)
    return votes_df, day_votes_df

# GOAL: Starting from the url which contains all the daily charts of the
#       seasons of Serie A from 'first_season' to 'last_season', all the
#       information related to every team and day are saved into a dataframe
def extract_day_charts_data(url, first_season, last_season, first_day, last_day,
                            season_df = [], day_df = [],
                            data_to_predict = False, save_out = False):
    if url == '':
        return season_df, day_df
    season_charts_df = pd.DataFrame(columns = ['team', 'season', 'points_list',
                                    'goals_scored_list', 'goals_conceded_list',
                                    'opponents_list'])
    day_charts_df = pd.DataFrame(columns=['team', 'season', 'day',
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
            'avg_opponents_goals_conceded_last4', 'std_opponents_goals_conceded_last4'
            ])
    seasons_no = last_season - first_season
    for i in range(0,seasons_no):
        analysed_season = first_season + i
        print("Currently analysed season: " + str(analysed_season))
        url_to_analyse = (url.split("saison_id=")[0] + "saison_id=" + 
                str(analysed_season) + url.split("saison_id=")[1] + str(1))
        # Send a GET request to the URL and open save the web page content
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' +
                    ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110' +
                    ' Safari/537.36'}
        response = requests.get(url_to_analyse, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Definition of all the team names
        teams_list = []
        all_teams = soup.find_all('td', {'class': 'no-border-links hauptlink'})
        for idx in range(0,len(all_teams)):
            team = unidecode(str(all_teams[idx]).split('\">')[2
                                        ].split('</a>')[0]).lower()
            teams_list.append(team)
        for day in range(first_day,last_day+1):
            print("Currently analysed day: " + str(day))
            url_to_analyse = (url.split("saison_id=")[0] + "saison_id=" + 
                    str(analysed_season) + url.split("saison_id=")[1] + str(day))
            # Send a GET request to the URL and open save the web page content
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' +
                       ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110' +
                        ' Safari/537.36'}
            response = requests.get(url_to_analyse, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            all_matches = soup.find_all('td', {'class': 'no-border-links'})
            all_points = soup.find_all('td', {'class': 'zentriert'})[40:]
            k = 0
            while k < len(all_points):
                team = unidecode(str(all_matches[60+int(k/8)]).split("\">")[
                                        2].split("</a")[0]).lower()
                goals_scored = int(str(all_points[k+5]).split(':')[0
                                        ].split('\">')[1].split("<")[0])
                goals_conceded = int(str(all_points[k+5]).split(':')[1
                                                    ].split('<')[0])
                points = (int(str(all_points[k+2]).split('\">')[1
                                        ].split('<')[0])*3 +
                          int(str(all_points[k+3]).split('\">')[1
                                        ].split('<')[0])*1)
                for team_idx in range(0,len(all_matches),3):
                    if team in str(all_matches[team_idx]
                            ).split('title=\"')[-1].split('\"')[0].lower():
                        if team_idx % 6 == 0:
                            opponent_team = str(all_matches[team_idx+3]
                            ).split('title=\"')[-1].split('\"')[0].lower()
                            for element in teams_list:
                                if element in opponent_team:
                                    opponent_team = element
                                    break
                            break
                        else:
                            opponent_team = str(all_matches[team_idx-3]
                            ).split('title=\"')[-1].split('\"')[0].lower()
                            for element in teams_list:
                                if element in opponent_team:
                                    opponent_team = element
                                    break
                            break
                if (len(season_charts_df[season_charts_df['team'] == team]) > 0):
                    if analysed_season in (season_charts_df[
                        season_charts_df['team'] == team][
                            'season'].tolist()):
                        total_points_team = np.sum(season_charts_df[
                            season_charts_df['team'] == team][
                                'points_list'].tolist()[-1])
                        total_goals_scored_team = np.sum(season_charts_df[
                            season_charts_df['team'] == team][
                                'goals_scored_list'].tolist()[-1])
                        total_goals_conceded_team = np.sum(season_charts_df[
                            season_charts_df['team'] == team][
                                'goals_conceded_list'].tolist()[-1])
                        # Add elements to the existing row
                        season_charts_df.at[np.where(season_charts_df['team'] == 
                                team)[0][-1], 'points_list'
                                ].append(points-total_points_team)
                        season_charts_df.at[np.where(season_charts_df['team'] == 
                                team)[0][-1], 'goals_scored_list'
                                ].append(goals_scored-
                                            total_goals_scored_team)
                        season_charts_df.at[np.where(season_charts_df['team'] == 
                                team)[0][-1], 'goals_conceded_list'
                                ].append(goals_conceded-
                                            total_goals_conceded_team)
                        season_charts_df.at[np.where(season_charts_df['team'] == 
                                team)[0][-1], 'opponents_list'
                                ].append(opponent_team)
                    else:
                        total_points_team = 0
                        total_goals_scored_team = 0
                        total_goals_conceded_team = 0
                        # Create the row
                        season_charts_df.loc[len(season_charts_df), 'team'
                                        ] = team
                        season_charts_df.loc[len(season_charts_df)-1,
                            'points_list'] = [points-total_points_team]
                        season_charts_df.loc[len(season_charts_df)-1,
                            'goals_scored_list'] = [goals_scored-
                                            total_goals_scored_team]
                        season_charts_df.loc[len(season_charts_df)-1,
                            'goals_conceded_list'] = [goals_conceded-
                                            total_goals_conceded_team]
                        season_charts_df.loc[len(season_charts_df)-1,
                            'opponents_list'] = [opponent_team]
                        season_charts_df.loc[len(season_charts_df)-1, 'season'
                                        ] = analysed_season
                else:
                    total_points_team = 0
                    total_goals_scored_team = 0
                    total_goals_conceded_team = 0
                    # Create the row
                    season_charts_df.loc[len(season_charts_df), 'team'
                                    ] = team
                    season_charts_df.loc[len(season_charts_df)-1,
                        'points_list'] = [points-total_points_team]
                    season_charts_df.loc[len(season_charts_df)-1,
                        'goals_scored_list'] = [goals_scored-
                                        total_goals_scored_team]
                    season_charts_df.loc[len(season_charts_df)-1,
                        'goals_conceded_list'] = [goals_conceded-
                                        total_goals_conceded_team]
                    season_charts_df.loc[len(season_charts_df)-1,
                        'opponents_list'] = [opponent_team]
                    season_charts_df.loc[len(season_charts_df)-1, 'season'
                                    ] = analysed_season
                k += 8

                # It is necessary to have data of at least 5 days (so
                # data of 4 days complete) in order to calculate the
                # averages and the standard deviations
                if day >= 5:
                    ## OWN TEAM INPUTS
                    # The row of the currently analysed team is selected
                    # and the averages and standard deviations are calculated
                    team_row = (season_charts_df.loc[(season_charts_df['team'] == team) &
                                (season_charts_df['season'] == analysed_season)]).copy()
                    team_points_list = team_row['points_list'].tolist()[0]
                    team_points_list = team_points_list[:-1]
                    avg_points_team, std_points_team = obtain_avg_std(
                        team_points_list,
                        len(team_points_list), True)
                    avg_points_last4_team, std_points_last4_team = obtain_avg_std(
                        team_points_list, 4, True)
                    ## GOALS SCORED
                    team_goals_scored_list = team_row['goals_scored_list'].tolist()[0]
                    team_goals_scored_list = team_goals_scored_list[:-1]
                    avg_goals_scored_team, std_goals_scored_team = obtain_avg_std(
                        team_goals_scored_list, 
                        len(team_goals_scored_list), True)
                    avg_goals_scored_last4_team, std_goals_scored_last4_team = obtain_avg_std(
                        team_goals_scored_list, 4, True)
                    ## GOALS CONCEDED
                    team_goals_conceded_list = team_row['goals_conceded_list'].tolist()[0]
                    team_goals_conceded_list = team_goals_conceded_list[:-1]
                    avg_goals_conceded_team, std_goals_conceded_team = obtain_avg_std(
                        team_goals_conceded_list, 
                        len(team_goals_conceded_list), True)
                    avg_goals_conceded_last4_team, std_goals_conceded_last4_team = obtain_avg_std(
                        team_goals_conceded_list, 4, True)
                    
                    ## OPPONENT TEAM INPUTS
                    # The row associated to the opponent of this day of the
                    # current analysed team is selected and the averages and
                    # standard deviations of the previous days of the opponent
                    # are calculated
                    current_day_opponent = team_row['opponents_list'].tolist()[0][-1]
                    opponent_team_row = (season_charts_df.loc[
                        (season_charts_df['team'] == current_day_opponent) &
                        (season_charts_df['season'] == analysed_season)]).copy()
                    ## POINTS
                    opponents_points_list = opponent_team_row['points_list'].tolist()[0]
                    if len(opponents_points_list) == day:
                        opponents_points_list = opponents_points_list[:-1]
                    assert len(opponents_points_list) == len(team_points_list), """
                        Error: Lists must have the same length.
                        Opponents points list length: {}
                        Current team points list length: {}
                        """.format(len(opponents_points_list), len(team_points_list))
                    avg_points_opp, std_points_opp = obtain_avg_std(
                        opponents_points_list,
                        len(opponents_points_list), True)
                    avg_points_last4_opp, std_points_last4_opp = obtain_avg_std(
                        opponents_points_list, 4, True)
                    ## GOALS SCORED
                    opponents_goals_scored_list = opponent_team_row[
                        'goals_scored_list'].tolist()[0]
                    if len(opponents_goals_scored_list) == day:
                        opponents_goals_scored_list = opponents_goals_scored_list[:-1]
                    assert len(opponents_goals_scored_list) == len(team_goals_scored_list), """
                        Error: Lists must have the same length.
                        Opponents goals scored list length: {}
                        Current team goals scored list length: {}
                        """.format(len(opponents_goals_scored_list), len(team_goals_scored_list))
                    avg_goals_scored_opp, std_goals_scored_opp = obtain_avg_std(
                        opponents_goals_scored_list, 
                        len(opponents_goals_scored_list), True)
                    avg_goals_scored_last4_opp, std_goals_scored_last4_opp = obtain_avg_std(
                        opponents_goals_scored_list, 4, True)
                    ## GOALS CONCEDED
                    opponents_goals_conceded_list = opponent_team_row[
                        'goals_conceded_list'].tolist()[0]
                    if len(opponents_goals_conceded_list) == day:
                        opponents_goals_conceded_list = opponents_goals_conceded_list[:-1]
                    assert len(opponents_goals_conceded_list) == len(team_goals_conceded_list), """
                        Error: Lists must have the same length.
                        Opponents goals conceded list length: {}
                        Current team goals conceded list length: {}
                        """.format(len(opponents_goals_conceded_list), len(team_goals_conceded_list))
                    avg_goals_conceded_opp, std_goals_conceded_opp = obtain_avg_std(
                        opponents_goals_conceded_list, 
                        len(opponents_goals_conceded_list), True)
                    avg_goals_conceded_last4_opp, std_goals_conceded_last4_opp = obtain_avg_std(
                        opponents_goals_conceded_list, 4, True)
                    
                    ## DAY_CHARTS_DF OWN TEAM DATA INSERTION
                    day_charts_df.loc[len(day_charts_df), 'team'] = team
                    day_charts_df.loc[len(day_charts_df)-1, 'season'] = analysed_season
                    if data_to_predict:
                        day_charts_df.loc[len(day_charts_df)-1, 'day'] = day-1
                    else:
                        day_charts_df.loc[len(day_charts_df)-1, 'day'] = day
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_points'] = avg_points_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_points'] = std_points_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_points_last4'] = avg_points_last4_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_points_last4'] = std_points_last4_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_goals_scored'] = avg_goals_scored_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_goals_scored'] = std_goals_scored_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_goals_scored_last4'] = avg_goals_scored_last4_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_goals_scored_last4'] = std_goals_scored_last4_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_goals_conceded'] = avg_goals_conceded_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_goals_conceded'] = std_goals_conceded_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_goals_conceded_last4'] = avg_goals_conceded_last4_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_goals_conceded_last4'] = std_goals_conceded_last4_team
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_opponents_points'] = avg_points_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_opponents_points'] = std_points_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_opponents_points_last4'] = avg_points_last4_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_opponents_points_last4'] = std_points_last4_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_opponents_goals_scored'] = avg_goals_scored_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_opponents_goals_scored'] = std_goals_scored_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_opponents_goals_scored_last4'] = avg_goals_scored_last4_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_opponents_goals_scored_last4'] = std_goals_scored_last4_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_opponents_goals_conceded'] = avg_goals_conceded_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_opponents_goals_conceded'] = std_goals_conceded_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'avg_opponents_goals_conceded_last4'] = avg_goals_conceded_last4_opp
                    day_charts_df.loc[len(day_charts_df)-1,
                        'std_opponents_goals_conceded_last4'] = std_goals_conceded_last4_opp
    if save_out:
        season_charts_df.to_csv('Fantateam lineup definition//' +
                             'season_charts_df.csv', index=False)
        day_charts_df.to_csv('Fantateam lineup definition//' +
                             'day_charts_df.csv', index=False)
    return season_charts_df, day_charts_df

# GOAL: Taken the two tables saved of votes and charts, the merge
#       is obtained with this function, in order to have a complete
#       table for the training phase
def merge_data(day_votes_df, day_charts_df, save_out = False):
    merged_df = pd.merge(day_votes_df, day_charts_df, on=['team', 'season', 'day'])
    if save_out:
        merged_df.to_csv('Fantateam lineup definition//' +
                            'merged_df.csv', index=False)
    return merged_df

# GOAL: Starting from the url of the votes and of the Serie A charts,
#       the dataframes of the daily votes and of the daily charts
#       are prepared and then merged, in order to be used as input
#       for the training of the Xgboost model
def prepare_input(url_votes, url_day_charts, first_year, last_year,
                  first_day, last_day, threshold_days, players_list):
    _, day_votes_df = extract_historical_votes_data(url_votes, first_year, last_year,
                                first_day, last_day, threshold_days, [], [], True, False)
    _, day_charts_df = extract_day_charts_data(url_day_charts, first_year, last_year,
                                first_day, last_day, [], [], True, False)
    input_df = merge_data(day_votes_df, day_charts_df)
    if players_list != '':
        input_df = input_df[input_df['name'].isin(players_list)]
        input_df = input_df[input_df['day'] == last_day-1
                            ].reset_index().drop(columns=['index'])
    return input_df

# GOAL: When the training of the model must be performed, this function starts
#       from the urls of the votes and of the daily charts, in order to put in
#       tables all data and save them for the training phase
def obtain_training_data(url_votes, votes_df,
                        url_day_charts, day_charts_df,
                        first_year, last_year, first_day, last_day,
                        save_votes_out = False, save_day_chart_out = False,
                        save_merged_out = False):
    # Storing data of the players votes
    _, day_votes_df = extract_historical_votes_data(url_votes, first_year, last_year,
            first_day, last_day, 4, votes_df, votes_df, save_votes_out)
    # Storing data of the teams charts
    _, day_charts_df = extract_day_charts_data(url_day_charts, first_year, last_year,
            first_day, last_day, day_charts_df, day_charts_df, save_day_chart_out)
    # Returning the merged data after the call to the function 'merge_data'
    return merge_data(day_votes_df, day_charts_df, save_merged_out)

# GOAL: Starting from the type of model of interested, a model in the class
#       'NeuralNetwork' or in the class 'XgBoost' is created and it is trained
def train_model(training_data, input_variables, output_variable,
                model_type, model_params, save_model_out = False):
    if model_type == 'nn':
        model = NeuralNetwork()
        model.train(training_data, input_variables, output_variable,
                model_params, save_model_out)
    elif model_type == 'xgb':
        model = XgBoost()
        model.train(training_data, input_variables, output_variable,
                model_params, save_model_out)
    else:
        raise NameError

# GOAL: Predict the vote for every player in function of the team they will face
#       and depending on the probability of playing for that particular day
def predict_votes(players_list, url_players_votes, url_day_charts,
                  first_day, model, current_day, current_year,
                  threshold_days, input_variables, save_pred_out = False):
    input_df = prepare_input(url_players_votes, url_day_charts,
                             current_year, current_year+1, first_day,
                             current_day, threshold_days, players_list)
    predictions = model.predict(input_df, input_variables)
    predictions_df = pd.DataFrame({'player_name': np.array(input_df['name']),
                                   'predicted_fantavote': predictions}
                                   ).sort_values(by='predicted_fantavote',
                                                 ascending=False
                                   ).reset_index().drop(columns=['index'])
    if save_pred_out:
        predictions_df.to_csv('Fantateam lineup definition//' +
                            'predictions_day-' + str(current_day) +
                            '_year-' + str(current_year) + '.csv', index=False)
    return predictions_df

# GOAL: Starting from the predicted vote of every player for the current day,
#       an optimization problem is solved in order to find the best module and
#       the best players for the lineup of the fantafootball day       
def solve_optimization_problem(scores, roles):
    # Create a new optimization problem
    prob = LpProblem("Football Team Selection", LpMaximize)

    # Define the decision variables
    players = scores.keys()
    x = LpVariable.dicts("x", players, 0, 1, LpBinary)

    # Define the objective function
    prob += lpSum([scores[player] * x[player] 
                   for player in players]), "Total Score"

    # Define the constraints
    prob += lpSum([x[player] for player in players]) == 11, "Total Players"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'GLK']) == 1, "Goalkeeper"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'DEF']) >= 3, "Minimum Defenders"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'DEF']) <= 5, "Maximum Defenders"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'MID']) >= 3, "Minimum Midfielders"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'MID']) <= 5, "Maximum Midfielders"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'STK']) >= 1, "Minimum Strikers"
    prob += lpSum([x[player] for player in players if 
                   roles[player] == 'STK']) <= 3, "Maximum Strikers"

    # Solve the problem
    prob.solve()

    # Print the results
    print("Total Score: ", value(prob.objective))
    print("Selected Players:")
    for player in players:
        if x[player].value() == 1:
            print(player)

    return prob