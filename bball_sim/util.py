
# Packages
import numpy as np
import pandas as pd
import random

random.seed(100)



#~~~~~TEAM FUNCTIONS~~~~~

def get_team_data(data, teamList):
    """
    derives probability of player being selected to take free throw, 2 pointer, or three pointer
    returns a data frame of team stats and respective probabilities
    """

    # Filter data
    df = data[data['bbref_id'].isin(teamList)].copy()

    # Generate composite ranking
    df['2P_Rank'] = (df['2P%'].rank(ascending=False) * df['2P'].rank(ascending=False)).rank(method='first')
    df['3P_Rank'] = (df['3P%'].rank(ascending=False) * df['3P'].rank(ascending=False)).rank(method='first')
    df['FT_Rank'] = (df['FT%'].rank(ascending=False) * df['FT'].rank(ascending=False)).rank(method='first')

    # Get probability of being shooter
    probDict = {
        1.0 : 0.6,
        2.0 : 0.8,
        3.0 : 0.9,
        4.0 : 0.95,
        5.0 : 1.0
    }

    # map probabilities
    df['2P_prob'] = df['2P_Rank'].map(probDict)
    df['3P_prob'] = df['3P_Rank'].map(probDict)
    df['FT_prob'] = df['FT_Rank'].map(probDict)


    # Generate output
    df = df.loc[:,['Player', 'bbref_id', '2P%', '3P%', 'FT%', '2P_prob', '3P_prob', 'FT_prob', 'ORB', 'DRB']]

    return df


def calc_rebounds_pct(df1, df2):
    """
    calculates the probability of team1 and team2 getting an offensive rebound
    returns a tuple of offensive rebound probabilities
    """

    # make coppies
    df1_ = df1.copy()
    df2_ = df2.copy()

    # assign teams
    df1_['team'] = 1
    df2_['team'] = 2

    # create dataframe
    df = pd.concat([df1_, df2_])
    df = df.groupby(['team'])[['ORB', 'DRB']].sum()

    # calculate rebound probs
    team1_ORB_prob = df.iloc[0,0] / (df.iloc[0,0] + df.iloc[1,1])
    team2_ORB_prob = df.iloc[1,0] / (df.iloc[1,0] + df.iloc[0,1])

    return team1_ORB_prob, team2_ORB_prob



#~~~~~EVENT FUNCTIONS~~~~~

def timeTaken(timeLeft, mu, sigma):
    """
    calculates how much time is utilized for an event (shot) using normal distribution
    returns a float of seconds used
    """

    t = np.random.normal(mu, sigma)
    shotClock = 24
    minTime = 0.3

    # force t to be valid time
    if t < minTime:
        t = minTime
    elif t > timeLeft or t > shotClock:
        t = min(timeLeft, shotClock)
    
    return t


def takeShot(shotType, timeLeft, makePct=0.45, offRebPct=0.25, points=0, maximize='N'):
    """
    calculates the outcome of a shot
    returns a tuple of time remaining, points scored, and who has possession of the ball
    """

    # Exception handling
    if timeLeft == 0:
        # print('no time left')
        return 0, 0, 'opp'

    # Determine how much time is taken
    mu, sigma = 4, 1.5
    # if team wants to maximize time used
    if maximize == 'Y':
        mu = timeLeft - 1.5
    t = timeTaken(timeLeft, mu, sigma)


    # Determine if points scored
    rand = random.random()
    if rand <= makePct:
        points += shotType
        timeLeft -= t
        poss = 'opp'
        # print('made', str(shotType))
    else:
        timeLeft -= t

        # rebound
        rand = random.random()
        if rand <= offRebPct:
            poss = 'keep'
            # print('missed', str(shotType), 'w/ rebound')
        else:
            poss = 'opp'
            # print('missed', str(shotType), 'w/o rebound')

        # hardcoded rebound time
        timeLeft -= 0.5
        if timeLeft < 0:
            timeLeft = 0
        
    return timeLeft, points, poss


def foul(timeLeft, ftType=2, makePct=0.8, ftRebPct=0.15, points=0):
    """
    calculates the time it takes to foul and outcome of subsequent free throws
    returns a tuple of time left, points scored, and who has possession of the ball
    """

    # Determine how much time is taken
    mu, sigma = 2, 1.5
    t = timeTaken(timeLeft, mu, sigma)
    
    timeLeft -= t
    if timeLeft < 0:
        return 0, 0, 'opp'

    # Determine if points scored
    # first free throw
    rand = random.random()
    if rand <= makePct:
        points = 1
        # print('made ft 1')
    elif ftType==1:
        rand = random.random()
        if rand <= ftRebPct:
            poss = 'keep'
            # print('missed front 1:1 w/ rebound')
        else:
            poss = 'opp'
            # print('missed front 1:1 w/o rebound')

        # hardcoded rebound time
        timeLeft -= 0.5
        if timeLeft < 0:
            timeLeft = 0

    else:
        # print('missed ft 1')
        pass

    # if 1:1 free throw
    if ftType == 1:
        if points == 0:
            rand = random.random()
            if rand <= ftRebPct:
                poss = 'keep'
                # print('missed ft 2 w/ rebound')
            else:
                poss = 'opp'
                # print('missed ft 2 w/o rebound')
            return timeLeft, points, poss


    # second free throw
    rand = random.random()
    if rand <= makePct:
        points += 1
        poss = 'opp'
        # print('made ft 2')
    else:
        rand = random.random()
        if rand <= ftRebPct:
            poss = 'keep'
            # print('missed ft 2 w/ rebound')
        else:
            poss = 'opp'
            # print('missed ft 2 w/o rebound')
        
        # hardcoded rebound time
        timeLeft -= 0.5
        if timeLeft < 0:
            timeLeft = 0

    return timeLeft, points, poss


def calc_shot_prob(df, shot):
    """
    calculates the probability of a shot being made
    returns a float object of probability
    """

    df = df.copy()
    rand = random.random()

    # Calculate probability
    # free throw
    if shot == 1:
        # determine shooter
        df = df.sort_values(['FT_prob'])
        df['shooter'] = np.where(df['FT_prob']>=rand, 1, 0)
        df = df[df['shooter']==1].iloc[0]
        shooter = df.loc['bbref_id']

        # determine prob
        prob = df.loc['FT%']
        # print(shot, shooter, prob)

    # 2 pointer
    elif shot == 2:
        # determine shooter
        df = df.sort_values(['2P_prob'])
        df['shooter'] = np.where(df['2P_prob']>=rand, 1, 0)
        df = df[df['shooter']==1].iloc[0]
        shooter = df.loc['bbref_id']

        # determine prob
        prob = df.loc['2P%']
        # print(shot, shooter, prob)

    # 3 pointer
    elif shot == 3:
        # determine shooter
        df = df.sort_values(['3P_prob'])
        df['shooter'] = np.where(df['3P_prob']>=rand, 1, 0)
        df = df[df['shooter']==1].iloc[0]
        shooter = df.loc['bbref_id']

        # determine prob
        prob = df.loc['3P%']
        # print(shot, shooter, prob)

    return prob



#~~~~~SIMULATION FUNCTIONS~~~~~
def prepSim(team1, team2):
    """
    helper function to prepare initial data for simulations
    returns a tuple of dataframe object of stats for each team and float object of offensive rebound probability for each team
    """

    # Get data
    cols = ['Player', 'bbref_id', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB']
    data = pd.read_csv('./player_data.csv', usecols=cols)

    # Teams
    df1 = get_team_data(data, team1)
    df2 = get_team_data(data, team2)

    # Rebound probability
    rbPct1, rbPct2 = calc_rebounds_pct(df1, df2)

    return df1, df2, rbPct1, rbPct2


def runSim(strategy, df1, df2, rbPct1, rbPct2, timeLeftInitial=30, pointDiffInitial=-3, teamFouls1Initial=5, teamFouls2Initial=5, overtimeProb=0.5):

    # Initialize values
    timeLeft = timeLeftInitial
    pointDiff = pointDiffInitial
    # nba reach double if 2 fouls in last 2:00 of quarter otherwise 5
    teamFouls1 = max(teamFouls1Initial, 3) # ensures bonus after 2 fouls if value less than 5 selected
    teamFouls2 = max(teamFouls2Initial, 3)

    # Run simulation
    while timeLeft > 0:
        # our team
        poss = 'keep'
        # keep shooting while maintaining possession
        while poss == 'keep':
            # print('us', timeLeft, pointDiff)

            # losing or tied
            if pointDiff <= 0:
                # losing
                if pointDiff < 0:
                    # print('losing')
                    shotProb = calc_shot_prob(df1, strategy)
                    # print('shot prob', shotProb)
                    timeLeft, points, poss = takeShot(strategy, timeLeft, shotProb, rbPct1)
                    pointDiff += points
                    # print(timeLeft, pointDiff, poss)
                
                # tied
                else:
                    # print('tied and maximizing')
                    shotProb = calc_shot_prob(df1, 2)
                    # print('shot prob', shotProb)
                    timeLeft, points, poss = takeShot(2, timeLeft, shotProb, rbPct1, maximize='Y') # always take 2 when tied
                    pointDiff += points
                    # print(timeLeft, pointDiff, poss)

            # winning
            else:
                # print('winning and free throws')
                shotProb = calc_shot_prob(df1, 1)
                # print('shot prob', shotProb)
                
                # double bonus
                if teamFouls1 >= 5:
                    # print('2 FT')
                    timeLeft, points, poss = foul(timeLeft, 2, shotProb, rbPct1*0.8) # lowered due to ft rebounds being harder
                    pointDiff += points
                    teamFouls2 += 1
                    # print(timeLeft, pointDiff, poss)
                # 1 & 1
                else:
                    # print('1:1')
                    timeLeft, points, poss = foul(timeLeft, 1, shotProb, rbPct1*0.8)
                    pointDiff += points
                    teamFouls2 += 1
                    # print(timeLeft, pointDiff, poss)


        # print()

        # opponent
        # break loop if no time remaining
        if timeLeft == 0:
            break

        poss = 'keep'
        # keep shooting while maintaining possession
        while poss == 'keep':
            # print('them', timeLeft, pointDiff)

            # losing or tied
            if pointDiff >= 0:
                # losing
                if pointDiff > 0:
                    # print('losing')
                    shotProb = calc_shot_prob(df2, strategy)
                    # print('shot prob', shotProb)
                    timeLeft, points, poss = takeShot(strategy, timeLeft, shotProb, rbPct2)
                    pointDiff -= points
                    # print(timeLeft, pointDiff, poss)
                
                # tied
                else:
                    # print('tied and maximizing')
                    shotProb = calc_shot_prob(df2, 2)
                    timeLeft, points, poss = takeShot(2, timeLeft, shotProb, rbPct2, maximize='Y')
                    pointDiff -= points
                    # print(timeLeft, pointDiff, poss)

            # winning
            else:
                # print('winning and free throws')
                shotProb = calc_shot_prob(df1, 1)
                # print('shot prob', shotProb)

                # double bonus
                if teamFouls1 >= 5:
                    # print('2 FT')
                    timeLeft, points, poss = foul(timeLeft, 2, shotProb, rbPct2*0.8)
                    pointDiff -= points
                    teamFouls1 += 1
                    # print(timeLeft, pointDiff, poss)

                # 1 & 1
                else:
                    # print('1:1')
                    timeLeft, points, poss = foul(timeLeft, 1, shotProb, rbPct1*0.8)
                    pointDiff -= points
                    teamFouls1 += 1
                    # print(timeLeft, pointDiff, poss)

        # print()


    # Determine result
    if pointDiff > 0:
        result = 1
        overtime = 0
    elif pointDiff < 0:
        result = 0
        overtime = 0
    else:
        rand = random.random()
        if rand <= overtimeProb:
            result = 1
        else:
            result = 0
        overtime = 1

    return result, overtime, pointDiff





