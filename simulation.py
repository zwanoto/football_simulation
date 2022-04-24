import time
import os
import pandas as pd
import requests
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

### Data is scraped from eloratings net using tsv tables.
### Since the website is not static we let the scraper sleep for a moment
url = f"https://www.eloratings.net/2022_World_Cup.tsv?={int(time.time())}"
table = requests.get(url).content
with open("table_data.tsv", "wb") as f:
  f.write(table)
df = pd.read_csv("table_data.tsv", sep="\t", header=None)
df = df[[2,3]]
df.columns  = ['Team','Elo']
df2 = pd.DataFrame({'Team' : ['WA','PE','CR'], 'Elo': [1841,1856,1743]})
df = pd.concat([df,df2],axis=0)
df = df.sort_values(by='Elo',ascending=False).reset_index(drop=True)


for i in range(df.shape[0]):
  team = teams.loc[teams['Symbol'] == df['Team'][i]]['Team'].iloc[0]
  df['Team'][i] = team

  ### Download the list with team abbreviations
def simul():
   teams = pd.read_csv('/home/thomas/Downloads/en.teams.csv', sep='\t')
   countrylist = teams['Symbol']
   df = pd.read_csv("table_data.tsv", sep="\t", header=None)
   df = df[[2,3]]
   df.columns  = ['Team','Elo']
   df2 = pd.DataFrame({'Team' : ['WA','PE','CR'], 'Elo': [1841,1856,1743]})
   df = pd.concat([df,df2],axis=0)
   df = df.sort_values(by='Elo',ascending=False).reset_index(drop=True)
   for i in range(df.shape[0]):
      team = teams.loc[teams['Symbol'] == df['Team'][i]]['Team'].iloc[0]
      df.iloc[i,0] = team
   df['Group'] = ['G','D','C','F','E','B','H','E','A','D','H','G','D','F','C','G','B',
                     'A','B','B','H','C','F','E','E','F','A','A','C','G','D','H']
   df['Position'] = [1,1,1,1,1,1,1,3,4,3,3,3,2,4,3,2,4,2,3,2,4,4,2,4,2,3,3,1,2,4,4,2]
   groups = ['A','B','C','D','E','F','G','H']
   results = []
   for group in groups: 
      df_group = df[df['Group']==group]
      team = [df_group.loc[df_group['Position']==3,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==4,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==1,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==2,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==1,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==3,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==4,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==2,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==4,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==1,['Team']].iloc[0].iloc[0],   
         df_group.loc[df_group['Position']==2,['Team']].iloc[0].iloc[0],
         df_group.loc[df_group['Position']==3,['Team']].iloc[0].iloc[0]]
      elo = [df_group.loc[df_group['Position']==3,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==4,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==1,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==2,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==1,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==3,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==4,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==2,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==4,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==1,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==2,['Elo']].iloc[0].iloc[0],
            df_group.loc[df_group['Position']==3,['Elo']].iloc[0].iloc[0]]
      elodiff =  [0] * len(elo)
      goals = [0] * len(elo)
      goalsagainst = [0] * len(elo)
      x = [0.0006762, 0.00259055]
      for i in range(len(team)):
         if i % 2 == 0:
            elodiff[i] = elo[i] - elo[i+1]
            g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i])), 1)
            goals[i] = g[0]
         else: 
            elodiff[i] = elo[i] - elo[i-1]
            g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i])), 1)
            goals[i] = g[0]         
            goalsagainst[i] = goals[i-1]
            goalsagainst[i-1]  = goals[i]
      results_group = (pd.DataFrame([team,elo,elodiff,goals,goalsagainst]).transpose())
      if group =='A':
         results =results_group
      else:
         results = pd.concat([results,results_group],axis=0)
   conditions = [(results[3] > results[4]),(results[3] ==  results[4]),(results[3] < results[4])]
   # create a list of the values we want to assign for each conditio
   values = [3,1,0]
   # create a new column and use np.select to assign values to it using our lists as arguments
   results['result'] = np.select(conditions, values)
   results.columns = ['Team','Elo','Diff','Goals','Against','result']
   df['Games'] = [3] * df.shape[0]
   df['Wins']  = [0] * df.shape[0]
   df['Draws'] = [0] * df.shape[0]
   df['Losses']  = [0] * df.shape[0]
   df['Goals'] = [0] * df.shape[0]
   df['Against'] = [0] * df.shape[0]
   df['Diff'] = [0] * df.shape[0]
   df['Points'] = [0] * df.shape[0]
   for i in range(df.shape[0]):
     res_team =results[results['Team']==df['Team'][i]]
     df.iloc[i,5] = np.sum(res_team['result']==3)
     df.iloc[i,6] = np.sum(res_team['result']==1)
     df.iloc[i,7] = np.sum(res_team['result']==0)
     df.iloc[i,8] = np.sum(res_team['Goals'])
     df.iloc[i,9] = np.sum(res_team['Against'])
     df.iloc[i,10] = df['Goals'][i] - df['Against'][i]
     df.iloc[i,11] = np.sum(res_team['result'])
     df['Ranking'] = range(32)
   df = df.sort_values(['Points','Diff','Goals'],ascending=False)
   for group in groups: 
      df_group = df[df['Group']==group]
      df_group['Ranking'] = range(4)
      df[df['Group']==group] = df_group
   team = [df.loc[df['Group']=='A',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='B',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='C',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='D',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='E',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='F',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='G',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='H',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='A',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='B',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='C',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='D',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='E',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='F',['Team']].iloc[0].iloc[0],
           df.loc[df['Group']=='G',['Team']].iloc[1].iloc[0],
           df.loc[df['Group']=='H',['Team']].iloc[0].iloc[0]]
   elo = [df.loc[df['Group']=='A',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='B',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='C',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='D',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='E',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='F',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='G',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='H',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='A',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='B',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='C',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='D',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='E',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='F',['Elo']].iloc[0].iloc[0],
           df.loc[df['Group']=='G',['Elo']].iloc[1].iloc[0],
           df.loc[df['Group']=='H',['Elo']].iloc[0].iloc[0]]
   elodiff =  [0] * len(elo)
   goals = [0] * len(elo) 
   goalsagainst = [0] * len(elo)
   x = [0.0006762, 0.00259055]
   for i in range(len(team)):
      if i % 2 == 0:
         elodiff[i] = elo[i] - elo[i+1]
         g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i])), 1)
         goals[i] = g[0]
      else: 
         elodiff[i] = elo[i] - elo[i-1]
         g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i])), 1)
         goals[i] = g[0]
         goalsagainst[i] = goals[i-1]
         goalsagainst[i-1]  = goals[i]
   results_group = (pd.DataFrame([team,elo,elodiff,goals,goalsagainst]).transpose())
   conditions = [(results_group[3] > results_group[4]),
   (results_group[3] ==  results_group[4]),
   (results_group[3] < results_group[4])]
   # create a list of the values we want to assign for each condition
   values = [3,1,0]
   # create a new column and use np.select to assign values to it using our lists as arguments
   results_group['result'] = np.select(conditions, values)
   results_group.columns = ['Team','Elo','Diff','Goals','Against','result']
   results_group['overtime'] = 0
   results_group['overtime_against'] = 0
   for i in range(results_group.shape[0]):
         if results_group['result'][i] == 1:
            g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i]))/4, 1)
            results_group['overtime'][i] = g[0]
   for i in range(results_group.shape[0]): 
         if i % 2 == 1:
            results_group['overtime_against'][i] = results_group['overtime'][i-1]
            results_group['overtime_against'][i-1] = results_group['overtime'][i]
   conditions = [(results_group['Goals']+results_group['overtime'] > results_group['Against'] + results_group['overtime_against']),
   (results_group['Goals']+results_group['overtime'] == results_group['Against'] + results_group['overtime_against']),
   (results_group['Goals']+results_group['overtime'] < results_group['Against'] + results_group['overtime_against'])]
   # create a list of the values we want to assign for each condition
   values = [3,1,0]
   # create a new column and use np.select to assign values to it using our lists as arguments
   results_group['result'] = np.select(conditions, values) 
   for i in range(results_group.shape[0]):
      if results_group['result'][i] == 1 and i % 2 == 0:
         g= 3 * np.random.binomial(1, 0.5, 1)
         results_group['result'][i] = g
         results_group['result'][i+1] = 3-g
   results_group = results_group[results_group['result']==3]
   ###################### QUARTERFINALS
   for i in range(3):
      team = list(results_group['Team'])
      elo = list(results_group['Elo'])
      elodiff =  [0] * len(elo)
      goals = [0] * len(elo)
      goalsagainst = [0] * len(elo)
      x = [0.0006762, 0.00259055]
      for i in range(len(team)):
         if i % 2 == 0:
            elodiff[i] = elo[i] - elo[i+1]
            g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i])), 1)
            goals[i] = g[0]
         else: 
            elodiff[i] = elo[i] - elo[i-1]
            g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i])), 1)
            goals[i] = g[0]
            goalsagainst[i] = goals[i-1]
            goalsagainst[i-1]  = goals[i]
      results_group = (pd.DataFrame([team,elo,elodiff,goals,goalsagainst]).transpose())
      conditions = [(results_group[3] > results_group[4]),
             (results_group[3] ==  results_group[4]),
             (results_group[3] < results_group[4])]
      # create a list of the values we want to assign for each condition
      values = [3,1,0]
      # create a new column and use np.select to assign values to it using our lists as arguments
      results_group['result'] = np.select(conditions, values)
      results_group.columns = ['Team','Elo','Diff','Goals','Against','result']
      results_group['overtime'] = 0
      results_group['overtime_against'] = 0
      for i in range(results_group.shape[0]):
         if results_group['result'][i] == 1:
            g= np.random.poisson(np.exp(x[0]+x[1]*(elodiff[i]))/4, 1)
            results_group['overtime'][i] = g[0]
      for i in range(results_group.shape[0]): 
         if i % 2 == 1:
            results_group['overtime_against'][i] = results_group['overtime'][i-1]
            results_group['overtime_against'][i-1] = results_group['overtime'][i]
      conditions = [(results_group['Goals']+results_group['overtime'] > results_group['Against'] + results_group['overtime_against']),
           (results_group['Goals']+results_group['overtime'] == results_group['Against'] + results_group['overtime_against']),
           (results_group['Goals']+results_group['overtime'] < results_group['Against'] + results_group['overtime_against'])]
         # create a list of the values we want to assign for each condition
      values = [3,1,0]
         # create a new column and use np.select to assign values to it using our lists as arguments
      results_group['result'] = np.select(conditions, values)
      for i in range(results_group.shape[0]):
         if results_group['result'][i] == 1 and i % 2 == 0:
            g= 3 * np.random.binomial(1, 0.5, 1)
            results_group['result'][i] = g
            results_group['result'][i+1] = 3-g
      results_group = results_group[results_group['result']==3]
   winner = (results_group['Team'])
   return(winner.iloc[0])


#### Monte Carlo Simulation

def monte_carlo(n):
   winners = []
   for i in range(n):
      winner = simul()
      winners.append(winner)
   winners = pd.DataFrame(winners)
   winners.columns = ['winner']
   pltdata = winners['winner'].value_counts()
   pltdata.plot(kind='bar')
   plt.show()
   

#### Execute simulation
monte_carlo(1000)










