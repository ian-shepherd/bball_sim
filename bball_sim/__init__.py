__version__ = '0.1.0'

import util
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



#~~~~~SIMULATION FUNCTIONS~~~~~

def baseSimulation(n, t, diff, fouls1, fouls2, ot_prob, plot=True, figname="N/A", verbose=False):
    """
    primary simulation to determine number of games won by each strategy
    returns a dataframe of strategy, result (number of won wins), number of sims, and mean point difference
    """

    # Generate empty lists
    simTypeList = []
    resultList = []
    overtimeList = []
    pointDiffList = []

    # Simulation
    for i in range(0, n):

        # 2 pt simulation
        result, overtime, pointDiff = util.runSim(2, 
                                                df1, 
                                                df2, 
                                                rbPct1, 
                                                rbPct2, 
                                                timeLeftInitial=t, 
                                                pointDiffInitial=diff, 
                                                teamFouls1Initial=fouls1, 
                                                teamFouls2Initial=fouls2, 
                                                overtimeProb=ot_prob)
        simTypeList.append('2pt')
        resultList.append(result)
        overtimeList.append(overtime)
        pointDiffList.append(pointDiff)

        # 3 pt simulation
        result, overtime, pointDiff = util.runSim(3, 
                                                df1, 
                                                df2, 
                                                rbPct1, 
                                                rbPct2, 
                                                timeLeftInitial=t, 
                                                pointDiffInitial=diff, 
                                                teamFouls1Initial=fouls1, 
                                                teamFouls2Initial=fouls2, 
                                                overtimeProb=ot_prob)
        simTypeList.append('3pt')
        resultList.append(result)
        overtimeList.append(overtime)
        pointDiffList.append(pointDiff)

        if verbose:
            print(i)

    # Output dataframe
    df = pd.DataFrame(zip(simTypeList, resultList, overtimeList, pointDiffList),
                columns=['Strategy', 'Result', 'Overtime', 'Point_diff'])
    df = df.groupby(['Strategy'])[['Result']].sum().reset_index()
    df['Sims'] = n

    if plot:
        # Generate plot
        # set plot style: grey grid in the background:
        sns.set(style="darkgrid")

        # set the figure size
        # plt.figure(figsize=(14, 10))
        plt.figure(figsize=(12, 8))

        # plot bars
        bar1 = sns.barplot(x='Strategy', y='Sims', data=df, estimator=sum, ci=None, color='lightcoral')
        bar2 = sns.barplot(x='Strategy', y='Result', data=df, color='dodgerblue')

        # legend
        top_bar = mpatches.Patch(color='lightcoral', label='Loss')
        bottom_bar = mpatches.Patch(color='dodgerblue', label='Win')
        plt.legend(bbox_to_anchor=(1,1), borderaxespad=0, frameon=False, ncol=2, handles=[bottom_bar, top_bar])

        # formatting
        plt.ylabel("# of Simulations")
        plt.title("Result of " + str(n) + " Simulations by Strategy")

        plt.savefig(figname)
        # plt.savefig('fig1 - base case')
        # plt.savefig('fig2 - shooters')
        # plt.savefig('fig3 - free throws')


    return df


def timeSimulation(t_start, t_end, n, diff, fouls1, fouls2, ot_prob, figname='N/A', verbose=False):
    """
    time remaining simulation to determine number of games won by each strategy by seconds remaining
    returns a dataframe of time remaining, strategy, result (number of won wins), number of sims, and % won
    """

    # Generate empty datafrmae
    df = pd.DataFrame(columns=['Time', 'Strategy', 'Result', 'Sims'])

    # Run simulations
    for t in range(t_start, t_end+1):
        df_ = baseSimulation(n, t, diff, fouls1, fouls2, ot_prob, plot=False)
        df_['Time'] = t
        df = pd.concat([df, df_])

        if verbose:
            print(t)

    # Prep dataframe
    df = df.reset_index(drop=True)
    df['Result'] = df['Result'].astype(int)
    df['Sims'] = df['Sims'].astype(int)
    df['Win %'] = df['Result'] / df['Sims']


    # Generate plot
    # set plot style: grey grid in the background:
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="Time", y="Win %", hue="Strategy")

    # formatting
    plt.xlim(t_start, t_end)
    plt.legend(bbox_to_anchor=(1,1), borderaxespad=0, frameon=False, ncol=2)
    plt.xlabel("Seconds Remaining")
    plt.title("Percentage of " + str(n) + " Simulations Won by Seconds Remaining")

    plt.savefig(figname)

    

    return df





#~~~~~TRIALS~~~~~

# Inputs
n = 100000 # number of simulations
t = 30 # time remaining
diff = -3 # point difference
fouls1 = 5 # number of fouls committed by team2 (winning)
fouls2 = 5 # number of fouls committed by team1 (losing)
ot_prob = 0.5 # overtime win probabiliy
t_start = 15 # start time for time based trial
t_end = 60 # end time for time based trial

# Base case
# teams
team1 = ['pg1', 'sg1', 'sf1', 'pf1', 'c1']
team2 = ['pg1', 'sg1', 'sf1', 'pf1', 'c1']
# get data
df1, df2, rbPct1, rbPct2 = util.prepSim(team1, team2)

# run sim
baseSimulation(n, t, diff, fouls1, fouls2, ot_prob, figname='fig1 - base case', verbose=True)
timeSimulation(t_start, t_end, n, diff, fouls1, fouls2, ot_prob, figname='fig4 - time effect', verbose=True)


# Shooters
# teams
team1 = ['curryse01', 'lillada01', 'hieldbu01', 'mccolcj01', 'capelca01']
team2 = ['pg1', 'sg1', 'sf1', 'pf1', 'c1']
# get data
df1, df2, rbPct1, rbPct2 = util.prepSim(team1, team2)
# run sim
baseSimulation(n, t, diff, fouls1, fouls2, ot_prob, figname='fig2 - shooters', verbose=True)


# Free throws
team1 = ['pg1', 'sg1', 'sf1', 'pf1', 'c1']
team2 = ['duranke01', 'leonaka01', 'butleji01', 'bealbr01', 'lillada01']
# get data
df1, df2, rbPct1, rbPct2 = util.prepSim(team1, team2)
# run sim
baseSimulation(n, t, diff, fouls1, fouls2, ot_prob, figname='fig3 - free throws', verbose=True)