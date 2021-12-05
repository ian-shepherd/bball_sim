# Packages
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import util

# Configure page
st.set_page_config(page_title='End of Game NBA Simulator',
                   page_icon='https://raw.githubusercontent.com/papagorgio23/Python101/master/newlogo.png',
                   layout="centered")


# Load data and convert to list of players
cols = ['Player', 'bbref_id']
players = pd.read_csv('./player_data.csv', usecols=cols)
playerList = players['Player'].tolist()


# Simulation function
def baseSimulation(n, t, diff, fouls1, fouls2, ot_prob):
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


    # Output dataframe
    df = pd.DataFrame(zip(simTypeList, resultList, overtimeList, pointDiffList),
                columns=['Strategy', 'Result', 'Overtime', 'Point_diff'])
    df = df.groupby(['Strategy'])[['Result']].sum().reset_index()
    df['Sims'] = n


    # Generate plot
    # set plot style: grey grid in the background:
    sns.set(style="darkgrid")

    # set the figure size
    # plt.figure(figsize=(14, 10))
    fig = plt.figure(figsize=(12, 8))

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

    st.pyplot(fig)

    # Print % of sims won
    st.write(str(round(df.loc[0,'Result'] / n * 100, 1)) + '% of 2pt strategy similations won')
    st.write(str(round(df.loc[1,'Result'] / n * 100, 1)) + '% of 3pt strategy similations won')

    return df


# Configure page
st.title("End of NBA Game Simulator")
st.subheader(
    "_Adjust the inputs in the sidebar and click apply to view the results of the simulation_"
)



# Configure sidebar
buton1 = st.sidebar.button("Run")

# game state inputs
n = st.sidebar.number_input("number of simulations", min_value=100, max_value=1000000, value=1000)
t = st.sidebar.number_input("seconds remaining", min_value=1, max_value=60, value=30)
diff = st.sidebar.number_input("point differential", min_value=-10, max_value=0, value=-3)
fouls1 = st.sidebar.number_input("fouls committed by leading team", min_value=0, max_value=10, value=5)
fouls2 = st.sidebar.number_input("fouls committed by trailing team", min_value=0, max_value=10, value=5)
ot_prob = st.sidebar.number_input("overtime win probability (%)", min_value=0, max_value=100, value=50) / 100

# trailing team players
st.sidebar.write("")
st.sidebar.write("Trailing Team")
player1 = st.sidebar.selectbox("player1", playerList, playerList.index("Kemba Walker\\walkeke02"))
player2 = st.sidebar.selectbox("player2", playerList, playerList.index("Marcus Smart\\smartma01"))
player3 = st.sidebar.selectbox("player3", playerList, playerList.index("Jaylen Brown\\brownja02"))
player4 = st.sidebar.selectbox("player4", playerList, playerList.index("Jayson Tatum\\tatumja01"))
player5 = st.sidebar.selectbox("player5", playerList, playerList.index("Grant Williams\\willigr01"))

# leading team players
st.sidebar.write("Leading Team")
player6 = st.sidebar.selectbox("player6", playerList, playerList.index("Ben Simmons\\simmobe01"))
player7 = st.sidebar.selectbox("player7", playerList, playerList.index("Seth Curry\\curryse01"))
player8 = st.sidebar.selectbox("player8", playerList, playerList.index("Danny Green\\greenda02"))
player9 = st.sidebar.selectbox("player9", playerList, playerList.index("Tobias Harris\\harrito02"))
player10 = st.sidebar.selectbox("player10", playerList, playerList.index("Joel Embiid\\embiijo01"))


# Run simulations
# if st.sidebar.button('Apply'):
if buton1:
    with st.spinner("Running simulations..."):
        team1 = [player1.rsplit('\\',1)[1], player2.rsplit('\\',1)[1], player3.rsplit('\\',1)[1], player4.rsplit('\\',1)[1], player5.rsplit('\\',1)[1]]
        team2 = [player6.rsplit('\\',1)[1], player7.rsplit('\\',1)[1], player8.rsplit('\\',1)[1], player9.rsplit('\\',1)[1], player10.rsplit('\\',1)[1]]
        df1, df2, rbPct1, rbPct2 = util.prepSim(team1, team2)
        baseSimulation(n, t, diff, fouls1, fouls2, ot_prob)


about = st.expander('Simulation Info')
with about:
    """
    This is an end of NBA game simulator based on player statistics for the 2020-2021 NBA season. You can select the same 
    player to both teams but you cannot put a player on the same team twice. There are also dummy players that act as a 
    representative player of that position. The simulator assumes the outcome of every possession is a made shot, missed 
    shot with the potential of a rebound, or intentional foul. It will not account for turnovers or blocks. The time taken 
    by each possession is based on a normal distribution accounting for what is in the best interest of the team. For example, 
    the simulation assumes the trailing team will take an average of 4 seconds but if the game is tied, that team will try 
    and maximize the amount of time taken so that mean is changed to the time remaining - 1.5 seconds. The shooter is also 
    determined by a composite rating that ranks players by number of that specific shot (free throw, 2 pt, 3 pt) taken per 
    game and their success rate. Players are then assigned a probability of being the selected shooter. Rebounds on the other 
    hand are determined by a team liklihood that compares the rebounding of the two teams to determine each team's liklihood 
    of successfully getting a rebound.
    """