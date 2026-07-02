#%%
import pandas as pd
import numpy as np
import warnings
import math
import time
import ast
import os
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore', category=NoAuthWarning)

#%%
#-----------------------------------------
#     MAKE API CALLS TO GET DATA
#-----------------------------------------
def loadDataFromAPI(filePath):
    startTime = time.perf_counter()

    # Get all competitions
    competitions = sb.competitions()

    # Filter for adult male competitions
    adultMaleComps = competitions[(competitions['competition_youth'] == False) & (competitions['competition_gender'] == 'male')]

    # Declare shots dataframe
    allShots = pd.DataFrame()

    # Loop through all competitions
    for index, competition in adultMaleComps.iterrows():
        # Get all matches from the filtered competitions
        matches = sb.matches(competition_id = competition['competition_id'], season_id = competition['season_id'])
        # Loop through matches
        for matchID in matches['match_id']:
            # Get all events from match
            events = sb.events(match_id=matchID)
            # Filter events to just get shots
            shots = events[(events['type'] == 'Shot') & (events['shot_outcome'] != 'Blocked')]
            # Add shots to allShots dataframe
            allShots = pd.concat([allShots, shots])

    print(allShots)
    # Save data locally as a csv file
    allShots.to_csv(filePath, index=False)

    endTime = time.perf_counter()
    executionTimeForGatheringData = endTime - startTime

    print(f'Execution Time For Gathering Data: {executionTimeForGatheringData} seconds')
#%%
def IsDefenderBetweenShooterAndGoal(shotData, defenderXCoord, defenderYCoord):
    '''Returns whether the defender is between the shooter and the goal'''
    
    # Define coordinates
    goalXCoord = shotData['goal_coordinates'][0]
    postOneYCoord = 44
    postTwoYCoord = 36
        
    shooterXCoord = shotData['location'][0]
    shooterYCoord = shotData['location'][1]

    # Check whether the shooter is on the goal line, then return false if it's true to prevent a division by zero
    if shooterXCoord == goalXCoord:
        return False

    # Calculate slopes from shooter to posts
    shooterToPostOneSlope = (shooterYCoord-postOneYCoord)/(shooterXCoord-goalXCoord)
    shooterToPostTwoSlope = (shooterYCoord-postTwoYCoord)/(shooterXCoord-goalXCoord)

    # Define the bounds that the y coordinate of the defender can be if they're between the shooter and goal
    postOneYBound = shooterToPostOneSlope * (defenderXCoord - shooterXCoord) + shooterYCoord
    postTwoYBound = shooterToPostTwoSlope * (defenderXCoord - shooterXCoord) + shooterYCoord

    # Determine whether the defender is between the shooter and the goal
    defenderIsBetweenShooterAndGoal = False
    if ((abs(defenderXCoord - goalXCoord) < abs(shooterXCoord - goalXCoord)) 
        and ((postOneYBound < defenderYCoord < postTwoYBound) or 
             (postOneYBound > defenderYCoord > postTwoYBound))):
        defenderIsBetweenShooterAndGoal = True
    
    return defenderIsBetweenShooterAndGoal

#%%
def ParseShotFreezeFrame(shotData):
    #Extract freeze frame data
    freezeFrame = shotData['shot_freeze_frame']

    # If the shot doesn't have a freeze frame, return an empty dictionary
    if not isinstance(freezeFrame, list):
        return {}, 0
    
    # Create freeze frame dict
    freezeFrameDict = {}
    numDefenders = 0
    # Loop throught every player in the freeze frame
    for player in freezeFrame:
        #If the player isn't the shooter's teammate
        if player['teammate'] == False:
            # Handle position being either a dict or string
            position = player['position']
            if isinstance(position, str):
                position = ast.literal_eval(position)
            positionName = position.get('name')
            
            # If the player is the opposition's goalkeeper
            if positionName == 'Goalkeeper':
                freezeFrameDict['gk_x_coord'] = player['location'][0]
                freezeFrameDict['gk_y_coord'] = player['location'][1]
            else:
                # Define defender coordinates
                defenderXCoord = player['location'][0]
                defenderYCoord = player['location'][1]
                if IsDefenderBetweenShooterAndGoal(shotData, defenderXCoord, defenderYCoord):
                    freezeFrameDict[f'defender_{numDefenders}_x_coord'] = defenderXCoord
                    freezeFrameDict[f'defender_{numDefenders}_y_coord'] = defenderYCoord
                    numDefenders += 1
                
    return freezeFrameDict, numDefenders

#%%
def CalcShotAngle(row):
    shooterX = row['location'][0]
    shooterY = row['location'][1]
    goalX = row['goal_coordinates'][0]
    post1Y = 36
    post2Y = 44
    
    # Vectors from shooter to each post
    vec1 = [goalX - shooterX, post1Y - shooterY]
    vec2 = [goalX - shooterX, post2Y - shooterY]
    
    # Dot product and magnitudes
    dotProduct = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # Angle between the two vectors
    return math.acos(dotProduct / (mag1 * mag2))

#%%
#-----------------------------------------
#    LOAD DATA AND INITIALLY CLEAN IT 
#-----------------------------------------

#Load data
filePath = '/Users/jakeholfinger/Desktop/CC Analyst/Data/StatsBomb Shot Data.csv'

# If data isn't saved locally, download it from StatsBomb API
if not os.path.exists(filePath):
    loadDataFromAPI(filePath)

allShots = pd.read_csv(filePath)

# Remove columns that don't have any data
for colName in allShots.columns:
    if allShots[colName].isna().all():
        allShots = allShots.drop(columns=[colName])

# Convert the columns' data that should be lists into lists (they're strings right now)
for col in ['location', 'shot_end_location', 'shot_freeze_frame']:
    allShots[col] = allShots[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

#-----------------------------------------
#  CREATE COLUMNS TO USE FOR REGRESSION
#-----------------------------------------

# Create center of goal location column
xValueMidfield = 60
allShots['goal_coordinates'] = allShots['shot_end_location'].apply(lambda loc: [0,40] if loc[0]<xValueMidfield else [120,40])

# Calculate shot distance
allShots['shot_distance'] = allShots.apply(lambda row: math.sqrt((row['goal_coordinates'][0] - row['location'][0])**2 +
                                                                 (row['goal_coordinates'][1] - row['location'][1])**2), axis=1)

# Calculate shot angle
allShots['shot_angle'] = allShots.apply(CalcShotAngle, axis=1)

# Separate shot freeze frame
parsedFreezeFrameDF = pd.DataFrame()
numDefendersList = []

for index, shot in allShots.iterrows():
    
    # Parse shot freeze frame
    parsedFreezeFrameDict, numDefenders = ParseShotFreezeFrame(shot)

    # Add any keys as a column that aren't already a column in the dataframe
    for key in parsedFreezeFrameDict:
        if not key in parsedFreezeFrameDF.columns:
            parsedFreezeFrameDF[key] = None
            
    # Add parsed shot freeze frame list to parsed shots freeze frame dataframe
    parsedFreezeFrameDF.loc[len(parsedFreezeFrameDF)] = parsedFreezeFrameDict
    
    # Add numDefenders
    numDefendersList.append(numDefenders)
    
# Add parsed freeze frame dataframe to allShots dataframe
allShots = pd.concat([allShots, parsedFreezeFrameDF], axis=1)

# Add number of defenders list to allShots dataframe as a column
allShots['num_defenders'] = numDefendersList

# Create a binary goal column
allShots['goal'] = allShots['shot_outcome'] == 'Goal'

# Add gk distance column
allShots['gk_distance'] = allShots.apply(lambda row: math.sqrt((row['gk_x_coord']-row['location'][0])**2 + 
                                                               (row['gk_y_coord']-row['location'][1])**2), axis=1)

# Add defender distance columns
for colName in allShots.columns:
    if 'defender' in colName and 'coord' in colName:
        defenderNum = colName[9]
        allShots[f'defender_{defenderNum}_distance'] = allShots.apply(lambda row: math.sqrt((row[f'defender_{defenderNum}_x_coord']-row['location'][0])**2 + 
                                                               (row[f'defender_{defenderNum}_y_coord']-row['location'][1])**2), axis=1)

# Split 'location' column into 'location_x' and 'location_y'
allShots['location_x'] = allShots['location'].apply(lambda loc: loc[0])
allShots['location_y'] = allShots['location'].apply(lambda loc: loc[1])

allShots.info()

#%%
#-----------------------------------------
#          CLEAN DATA FURTHER
#-----------------------------------------

# Filter out shots that aren't wanted for regression (deflected, redirected, set pieces, and open goals)
allShotsRegression = allShots[(allShots['shot_deflected'].isna()) & (allShots['shot_redirect'].isna()) & (allShots['shot_type'] == 'Open Play') & (allShots['shot_open_goal'].isna())] 

# Fill empty values in 'shot_first_time' and 'shot_aerial_won' column with False
allShotsRegression['shot_first_time'] = allShotsRegression['shot_first_time'].fillna(False)
allShotsRegression['shot_aerial_won'] = allShotsRegression['shot_aerial_won'].fillna(False)
allShotsRegression['shot_one_on_one'] = allShotsRegression['shot_one_on_one'].fillna(False)
allShotsRegression['under_pressure'] = allShotsRegression['under_pressure'].fillna(False)
allShotsRegression['shot_follows_dribble'] = allShotsRegression['shot_follows_dribble'].fillna(False)
#allShotsRegression['shot_open_goal'] = allShotsRegression['shot_open_goal'].fillna(False)

# Drop rows with missing gk or defender distances
allShotsRegression = allShotsRegression.dropna(subset=['gk_distance'])
defenderMean = allShotsRegression[allShotsRegression['defender_0_distance'] < 35]['defender_0_distance'].mean()
allShotsRegression['defender_0_distance'] = allShotsRegression['defender_0_distance'].fillna(defenderMean)

#-----------------------------------------
#          LINEAR REGRESSION
#-----------------------------------------

# Reduce dataframe to only columns needed for linear regression
# location_x is included for depth model — closer shots force keepers further forward
# location_y is included for lateral model — keepers shift based on shooter's lateral position
allShotsLinearRegression = allShotsRegression[['shot_statsbomb_xg', 'shot_distance', 'shot_angle', 'defender_0_distance', 'num_defenders', 'location_x', 'location_y']]#, 'shot_open_goal']]

# Set up independent variables for depth model (no location_y since depth doesn't depend on lateral shooter position)
linearXAxisX = allShotsLinearRegression.drop('location_y', axis=1)
linearXAxisX = sm.add_constant(linearXAxisX)

# Set up independent variables for lateral model (includes location_y, excludes location_x)
linearXAxisY = allShotsLinearRegression.drop('location_x', axis=1)
linearXAxisY = sm.add_constant(linearXAxisY)

# Set up dependent variables
linearYAxisX = allShotsRegression['gk_x_coord']
linearYAxisY = allShotsRegression['gk_y_coord']

# Fit x model (gk depth)
linearXModel = OLS(linearYAxisX, linearXAxisX).fit()

# Fit y model (gk latitude)
linearYModel = OLS(linearYAxisY, linearXAxisY).fit()

#Print summaries
print("=== GK DEPTH (X) MODEL ===")
print(linearXModel.summary())
print()
print()
print("=== GK LATERAL (Y) MODEL ===")
print(linearYModel.summary())

#%%
#-----------------------------------------
#         CALCULATE RESDIDUALS
#-----------------------------------------

# Calculate residuals - predicted position - actual position
allShotsRegression['residual_x'] = linearXModel.predict(linearXAxisX) - linearYAxisX
allShotsRegression['residual_y'] =  linearYAxisY - linearYModel.predict(linearXAxisY)

bounds = {}
for residual in ['residual_x', 'residual_y']:
    # Get mean and std of residual
    mean = allShotsRegression[residual].mean()
    std = allShotsRegression[residual].std()
    bounds[residual] = (mean - 3*std, mean + 3*std)
    
# Apply both filters
allShotsRegression = allShotsRegression[
    (allShotsRegression['residual_x'] > bounds['residual_x'][0]) & 
    (allShotsRegression['residual_x'] < bounds['residual_x'][1]) &
    (allShotsRegression['residual_y'] > bounds['residual_y'][0]) & 
    (allShotsRegression['residual_y'] < bounds['residual_y'][1])
]

# Take absolute value of residual_y because direction doesn't matter
allShotsRegression['residual_y'] = allShotsRegression['residual_y'].abs()

#-----------------------------------------
#    CLEAN DATA FOR LOGISTIC REGRESSION
#-----------------------------------------

# Reduce dataframe to only columns needed for regression
allShotsRegression = allShotsRegression[['goal', 'shot_aerial_won', 'shot_body_part', 'shot_first_time', 'shot_technique', 'shot_one_on_one', 'shot_distance', 'shot_angle', 'shot_follows_dribble', 'under_pressure', 'residual_x', 'residual_y', 'defender_0_distance', 'num_defenders']]#, 'play_pattern, 'shot_open_goal']]

# Convert categorical columns into numerical columns
allShotsRegressionDummies = pd.get_dummies(allShotsRegression, columns=['shot_aerial_won', 'shot_body_part', 'shot_first_time', 'shot_technique', 'shot_one_on_one', 'shot_follows_dribble', 'under_pressure'], dtype=int, drop_first=True)#, 'shot_open_goal'], dtype=int, drop_first=True)

allShotsRegressionDummies.info()

# Add quadratic depth term
allShotsRegressionDummies['residual_x_squared'] = allShotsRegressionDummies['residual_x'] ** 2

# Add interaction terms to allow optimal residual to vary by shot context
allShotsRegressionDummies['residual_x_x_distance'] = allShotsRegressionDummies['residual_x'] * allShotsRegressionDummies['shot_distance']
allShotsRegressionDummies['residual_x_squared_x_distance'] = allShotsRegressionDummies['residual_x_squared'] * allShotsRegressionDummies['shot_distance']

#-----------------------------------------
#    COMPLETE LOGISTIC REGRESSION
#-----------------------------------------

# Define axes
dependentVar = allShotsRegressionDummies['goal']
independentVars = allShotsRegressionDummies.drop(columns=['goal'])

# Add constant
independentVars = sm.add_constant(independentVars)

# Fit logistic regression
model = sm.Logit(dependentVar, independentVars)
result = model.fit()

print(result.summary())

print(allShotsRegression['residual_y'].describe())

#%%
#------------------------------------------------------------
#  CREATE SYNTHETIC DATA FOR PLOT THAT ISOLATES GK DISTANCE
#------------------------------------------------------------

# Create dictionary for storing means (if its a numerical column) and most common values (if it was categorical)
meanCommonDict = {}
for colName in allShotsRegressionDummies.columns:
    col = allShotsRegressionDummies[colName]
    if colName in ['shot_distance', 'shot_angle', 'residual_x', 'residual_y', 'defender_0_distance', 'num_defenders']:
        meanCommonDict[colName] = allShotsRegressionDummies[colName].mean()
    elif colName != 'goal':
        meanCommonDict[colName] = allShotsRegressionDummies[colName].mode()[0]
meanCommonDict['shot_body_part_Right Foot'] = 1

# Calculate beta coefficients for optimal residual formula
# optimal_residual_x = -(beta1 + beta3 * distance) / (2 * (beta2 + beta4 * distance))
beta1 = result.params['residual_x']
beta2 = result.params['residual_x_squared']
beta3 = result.params['residual_x_x_distance']
beta4 = result.params['residual_x_squared_x_distance']

# Optimal residual at mean shot distance for reference
mean_distance = allShotsRegressionDummies['shot_distance'].mean()
optimalResidualX_mean = -(beta1 + beta3 * mean_distance) / (2 * (beta2 + beta4 * mean_distance))
print(f'Optimal residual_x at mean distance ({mean_distance:.1f}m): {optimalResidualX_mean:.3f}m')

# Marginalize over every real shot rather than a single modal row, so the average
# reflects the true joint distribution of technique, body part, distance, angle,
# defenders, pressure, etc. instead of a "right-footed normal-technique" stereotype.
baseShotsDF = allShotsRegressionDummies.drop(columns=['goal']).copy()

residuals = ['residual_x', 'residual_y']
residualNameDict = {'residual_x': 'Depth', 'residual_y': 'Lateral'}
for residual in residuals:
    # Clip to ±3 standard deviations
    std = allShotsRegressionDummies[residual].std()
    minCoord = max(allShotsRegressionDummies[residual].min(), -3 * std)
    maxCoord = min(allShotsRegressionDummies[residual].max(), 3 * std)

    coords = np.arange(minCoord, maxCoord, 0.1)
    marginalProbs = np.empty(len(coords))

    for i, coord in enumerate(coords):
        df = baseShotsDF.copy()
        df['residual_x'] = 0.0
        df['residual_y'] = 0.0
        df['residual_x_squared'] = 0.0
        df['residual_x_x_distance'] = 0.0
        df['residual_x_squared_x_distance'] = 0.0

        df[residual] = coord
        if residual == 'residual_x':
            df['residual_x_squared'] = coord ** 2
            df['residual_x_x_distance'] = coord * df['shot_distance']
            df['residual_x_squared_x_distance'] = coord ** 2 * df['shot_distance']

        df_const = sm.add_constant(df, has_constant='add')
        marginalProbs[i] = result.predict(df_const).mean()

    #-----------------------------------------------------
    #                   PLOT RESULTS
    #-----------------------------------------------------
    xAxis = coords
    yAxis = marginalProbs * 100

    plt.plot(xAxis, yAxis)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}%'))

    plt.xlabel(f'Distance from Expected {residualNameDict[residual]} Position (m)')
    plt.ylabel('Predicted Goal Probability')
    plt.title(f'Effect of {residualNameDict[residual]} Positioning Error on Goal Probability\nAveraged Over All Shots')

    if residual == 'residual_x':
        ax = plt.gca()
        ticks = ax.get_xticks()
        ax.set_xticklabels([f'{-t:.1f}' for t in ticks])

    plt.show()

#%%:
#-----------------------------------------------------
#   BAR CHART COMPARING RESIDUAL EFFECT SIZES
#-----------------------------------------------------

# Beta coefficients for context-aware optimal residual
beta1 = result.params['residual_x']
beta2 = result.params['residual_x_squared']
beta3 = result.params['residual_x_x_distance']
beta4 = result.params['residual_x_squared_x_distance']

# Optimal residual at mean shot distance
mean_distance = allShotsRegressionDummies['shot_distance'].mean()
optimal_residual_x = -(beta1 + beta3 * mean_distance) / (2 * (beta2 + beta4 * mean_distance))
print(f'Optimal Depth Residual at mean distance: {optimal_residual_x:.3f}m')

# Average predicted goal probability across every real shot, with the chosen
# residuals applied uniformly. Interaction terms use each shot's own distance.
def marginal_prob(residual_x_val, residual_y_val):
    df = baseShotsDF.copy()
    df['residual_x'] = residual_x_val
    df['residual_y'] = residual_y_val
    df['residual_x_squared'] = residual_x_val ** 2
    df['residual_x_x_distance'] = residual_x_val * df['shot_distance']
    df['residual_x_squared_x_distance'] = residual_x_val ** 2 * df['shot_distance']
    df_const = sm.add_constant(df, has_constant='add')
    return result.predict(df_const).mean()

optimal_prob = marginal_prob(optimal_residual_x, 0)

# Worst realistic depth positions (±3 std from zero)
depth_std = allShotsRegressionDummies['residual_x'].std()
worst_residual_x_positive = 3 * depth_std
worst_residual_x_negative = -3 * depth_std

worst_prob_positive = marginal_prob(worst_residual_x_positive, 0)
worst_prob_negative = marginal_prob(worst_residual_x_negative, 0)

# Use the higher of the two worst probabilities
worst_prob_depth = max(worst_prob_positive, worst_prob_negative)
depth_total_effect = worst_prob_depth - optimal_prob

# Lateral effect: hold depth at optimal, push lateral to ±3 std
lateral_std = allShotsRegressionDummies['residual_y'].std()
worst_residual_y = 3 * lateral_std

baseline_lateral_prob = marginal_prob(optimal_residual_x, 0)
worst_lateral_prob = marginal_prob(optimal_residual_x, worst_residual_y)

lateral_total_effect = worst_lateral_prob - baseline_lateral_prob

print(f'Optimal depth probability: {optimal_prob*100:.2f}%')
print(f'Worst depth probability: {worst_prob_depth*100:.2f}%')
print(f'Depth total effect: {depth_total_effect*100:.2f}%')
print(f'Lateral total effect: {lateral_total_effect*100:.2f}%')

# Set up bar chart data
labels = ['Depth Error', 'Lateral Error']
coef_values = [depth_total_effect, lateral_total_effect]

# Define bar colors
colors = ['#d73027', '#4575b4']

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(labels, coef_values, color=colors, width=0.4, zorder=3)

ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=2)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=1)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 100:.1f}%'))

# Add effect size labels — position above bar if positive, below bar if negative
# This prevents overlap with the zero axis line
for i, (bar, coef) in enumerate(zip(bars, coef_values)):
    # Determine vertical position and alignment based on sign
    if coef >= 0:
        ypos = coef + 0.0015
        va = 'bottom'
    else:
        ypos = coef - 0.0015
        va = 'top'

    # Determine horizontal position based on bar index
    if i == 1:  # lateral bar — place label on left side
        ax.text(bar.get_x() - 0.02, ypos, f'{coef*100:.2f}%',
                ha='right', va=va, fontsize=11)
    else:  # depth bar — place label on right side
        ax.text(bar.get_x() + bar.get_width() + 0.02, ypos, f'{coef*100:.2f}%',
                ha='left', va=va, fontsize=11)

# Add p-value annotations — position above bar top for positive, below bar bottom for negative
depth_ypos = coef_values[0] + 0.003 if coef_values[0] >= 0 else coef_values[0] - 0.003
lateral_ypos = coef_values[1] + 0.003 if coef_values[1] >= 0 else coef_values[1] - 0.003
depth_va = 'bottom' if coef_values[0] >= 0 else 'top'
lateral_va = 'bottom' if coef_values[1] >= 0 else 'top'

ax.text(0, depth_ypos, 'p < 0.001', ha='center', va=depth_va, fontsize=10, color='black')
ax.text(1, lateral_ypos, f'p = {result.pvalues["residual_y"]:.3f}',
        ha='center', va=lateral_va, fontsize=10, color='black')

ax.set_ylabel('Increase in Goal Probability\n(Optimal vs Worst Realistic Position)', fontsize=11)
ax.set_title('Total Effect of Positioning Errors on Goal Probability\nDepth vs Lateral', fontsize=14)
ax.set_ylim([min(coef_values) - 0.012, max(coef_values) + 0.012])

plt.tight_layout()
print('')
plt.show()
print('')

#%%
#-----------------------------------------------------
#   OPTIMAL GOALKEEPER POSITION HEATMAPS
#-----------------------------------------------------

# Retrain depth linear model without xG and with location_x for better R squared
# location_x captures that closer shots force keepers further from their line
linearXAxisX_noXG = allShotsLinearRegression[['shot_distance', 'shot_angle', 'defender_0_distance', 'num_defenders', 'location_x']]
linearXAxisX_noXG = sm.add_constant(linearXAxisX_noXG)
linearXModel_noXG = OLS(linearYAxisX, linearXAxisX_noXG).fit()
print(f'Depth model R² (with location_x, without xG): {linearXModel_noXG.rsquared:.3f}')

# Define cell dimensions
numCells = 7
xStart = 85
xEnd = 120
yStart = 15
yEnd = 65

# Define edges directly so right edge is exactly 120
shotXEdges = np.linspace(xStart, xEnd, numCells + 1)
shotYEdges = np.linspace(yStart, yEnd, numCells + 1)

# Cell centers are midpoints of edges
shotXRange = (shotXEdges[:-1] + shotXEdges[1:]) / 2
shotYRange = (shotYEdges[:-1] + shotYEdges[1:]) / 2

# Convert axes to distance from goal and distance from goal center
shotXDistEdges = 120 - shotXEdges
shotYCenterEdges = shotYEdges - 40

shotXDistRange = 120 - shotXRange
shotYCenterRange = shotYRange - 40

# Flip x so left = closer to goal, right = further from goal
shotXDistEdges = shotXDistEdges[::-1]
shotXDistRange = shotXDistRange[::-1]

shotXDistEdgeGrid, shotYCenterEdgeGrid = np.meshgrid(shotXDistEdges, shotYCenterEdges)
shotXDistGrid, shotYCenterGrid = np.meshgrid(shotXDistRange, shotYCenterRange)
shotXGrid, shotYGrid = np.meshgrid(shotXRange, shotYRange)

shotXCoords = shotXGrid.flatten()
shotYCoords = shotYGrid.flatten()

# Calculate shot distance and angle for each grid point
def calcDistance(shotX, shotY):
    return math.sqrt((120 - shotX)**2 + (40 - shotY)**2)

def calcAngle(shotX, shotY):
    vec1 = [120 - shotX, 36 - shotY]
    vec2 = [120 - shotX, 44 - shotY]
    dotProduct = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    cosAngle = max(-1, min(1, dotProduct / (mag1 * mag2)))
    return math.acos(cosAngle)

distances = np.array([calcDistance(x, y) for x, y in zip(shotXCoords, shotYCoords)])
angles = np.array([calcAngle(x, y) for x, y in zip(shotXCoords, shotYCoords)])

# Build context for depth linear model (no xG, includes location_x)
linearContextDepth = pd.DataFrame([{
    'const': 1,
    'shot_distance': d,
    'shot_angle': a,
    'defender_0_distance': meanCommonDict['defender_0_distance'],
    'num_defenders': meanCommonDict['num_defenders'],
    'location_x': x
} for d, a, x in zip(distances, angles, shotXCoords)])

# Build context for lateral linear model (includes location_y)
linearContextLateral = pd.DataFrame([{
    'const': 1,
    'shot_statsbomb_xg': allShotsLinearRegression['shot_statsbomb_xg'].mean(),
    'shot_distance': d,
    'shot_angle': a,
    'defender_0_distance': meanCommonDict['defender_0_distance'],
    'num_defenders': meanCommonDict['num_defenders'],
    'location_y': y
} for d, a, y in zip(distances, angles, shotYCoords)])

# Get predicted keeper depth position
predictedDepths = linearXModel_noXG.predict(linearContextDepth)

# Calculate context-aware optimal residual for each grid point
# optimal = -(beta1 + beta3 * distance) / (2 * (beta2 + beta4 * distance))
beta1 = result.params['residual_x']
beta2 = result.params['residual_x_squared']
beta3 = result.params['residual_x_x_distance']
beta4 = result.params['residual_x_squared_x_distance']

optimalResiduals = -(beta1 + beta3 * distances) / (2 * (beta2 + beta4 * distances))
print(f'Optimal residual range across shot contexts: {optimalResiduals.min():.3f}m to {optimalResiduals.max():.3f}m')

# Apply context-specific optimal residual to get optimal depth per cell
optimalDepths = predictedDepths.values + optimalResiduals
optimalDistancesFromGoal = 120 - optimalDepths

# Clip optimal distances so keeper cannot be inside the goal (distance < 0)
# Minimum realistic keeper position is on the goal line (0m) 
optimalDistancesFromGoal = np.maximum(optimalDistancesFromGoal, 0.0)

# Average keeper depth (residual = 0)
averageDistancesFromGoal = 120 - predictedDepths

# Get predicted keeper lateral position
optimalLateralPositions = linearYModel.predict(linearContextLateral)
optimalLateralDistancesFromCenter = optimalLateralPositions - 40
averageLateralDistancesFromCenter = optimalLateralDistancesFromCenter

# Reshape to 2D grids — flip x axis to match distance from goal orientation
optimalDistanceGrid = optimalDistancesFromGoal.reshape(shotXGrid.shape)[:, ::-1]
averageDistanceGrid = averageDistancesFromGoal.values.reshape(shotXGrid.shape)[:, ::-1]
optimalLateralGrid = optimalLateralDistancesFromCenter.values.reshape(shotXGrid.shape)[:, ::-1]
averageLateralGrid = averageLateralDistancesFromCenter.values.reshape(shotXGrid.shape)[:, ::-1]

print(f'Depth range: {optimalDistancesFromGoal.min():.2f}m to {optimalDistancesFromGoal.max():.2f}m from goal line')
print(f'Lateral range: {optimalLateralDistancesFromCenter.min():.2f}m to {optimalLateralDistancesFromCenter.max():.2f}m from center')

#-----------------------------------------------------
#   HELPER FUNCTION FOR FIELD LINES
#-----------------------------------------------------

def draw_field_lines(ax):
    line_color = 'white'
    line_alpha = 0.4
    line_width = 3

    # Goal posts
    ax.plot([0, 0], [-4, 4], color=line_color, linewidth=6,
            solid_capstyle='round', zorder=5, alpha=line_alpha)

    # 6 yard box
    ax.add_patch(Rectangle((0, -10), width=6, height=20, linewidth=line_width,
                            edgecolor=line_color, facecolor='none',
                            zorder=5, alpha=line_alpha))

    # 18 yard box
    ax.add_patch(Rectangle((0, -22), width=18, height=44, linewidth=line_width,
                            edgecolor=line_color, facecolor='none',
                            zorder=5, alpha=line_alpha))

    # Penalty spot
    ax.scatter(12, 0, color=line_color, s=100, zorder=5,
               marker='o', alpha=line_alpha)

#-----------------------------------------------------
#                  PLOT HEATMAPS
#-----------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

cmap_depth = LinearSegmentedColormap.from_list(
    'custom_dark',
    ['#2d8b2d', '#4a9e1a', '#8b8b00', '#b3a000', '#cc5500', '#b33000', '#8b0000'],
    N=256
)

cmap_lateral = LinearSegmentedColormap.from_list(
    'custom_diverging',
    ['#8b0000', '#b33000', '#cc5500', '#b3a000', '#1a6e1a', '#b3a000', '#cc5500', '#b33000', '#8b0000'],
    N=256
)

depth_vmin = min(optimalDistanceGrid.min(), averageDistanceGrid.min())
depth_vmax = max(optimalDistanceGrid.max(), averageDistanceGrid.max())
lat_abs_max = max(abs(optimalLateralGrid.min()), abs(optimalLateralGrid.max()),
                  abs(averageLateralGrid.min()), abs(averageLateralGrid.max()))

def annotate_cells(ax, grid, xRange, yRange, fmt):
    for i in range(len(yRange)):
        for j in range(len(xRange)):
            value = grid[i, j]
            label = fmt(value)
            ax.text(xRange[j], yRange[i], label,
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white', zorder=6)

def depth_fmt(v): return f'{v:.1f}m'
def lateral_fmt(v): return f'+{v:.1f}m' if v > 0 else f'{v:.1f}m'

# --- Top left: Optimal depth ---
heatmap1 = axes[0, 0].pcolormesh(
    shotXDistEdgeGrid, shotYCenterEdgeGrid, optimalDistanceGrid,
    cmap=cmap_depth, shading='flat',
    vmin=optimalDistanceGrid.min(), vmax=optimalDistanceGrid.max()
)
cbar1 = plt.colorbar(heatmap1, ax=axes[0, 0])
cbar1.set_label('Distance from Goal Line (m)', fontsize=11, labelpad=15)
annotate_cells(axes[0, 0], optimalDistanceGrid, shotXDistRange, shotYCenterRange, depth_fmt)
draw_field_lines(axes[0, 0])
axes[0, 0].set_xlabel('Shot Distance from Goal Line (m)', fontsize=11)
axes[0, 0].set_ylabel('Shot Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=10)
axes[0, 0].set_title('Optimal Goalkeeper Depth Position\nby Shot Location', fontsize=12)
axes[0, 0].set_xlim([shotXDistEdges.min(), shotXDistEdges.max()])
axes[0, 0].set_ylim([shotYCenterEdges.min(), shotYCenterEdges.max()])

# --- Top right: Optimal lateral ---
opt_lat_abs_max = max(abs(optimalLateralGrid.min()), abs(optimalLateralGrid.max()))
heatmap2 = axes[0, 1].pcolormesh(
    shotXDistEdgeGrid, shotYCenterEdgeGrid, optimalLateralGrid,
    cmap=cmap_lateral, shading='flat',
    vmin=-opt_lat_abs_max, vmax=opt_lat_abs_max
)
cbar2 = plt.colorbar(heatmap2, ax=axes[0, 1])
cbar2.set_label('Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=11, labelpad=15)
annotate_cells(axes[0, 1], optimalLateralGrid, shotXDistRange, shotYCenterRange, lateral_fmt)
draw_field_lines(axes[0, 1])
axes[0, 1].set_xlabel('Shot Distance from Goal Line (m)', fontsize=11)
axes[0, 1].set_ylabel('Shot Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=10)
axes[0, 1].set_title('Optimal Goalkeeper Lateral Position\nby Shot Location', fontsize=12)
axes[0, 1].set_xlim([shotXDistEdges.min(), shotXDistEdges.max()])
axes[0, 1].set_ylim([shotYCenterEdges.min(), shotYCenterEdges.max()])

# --- Bottom left: Average depth ---
heatmap3 = axes[1, 0].pcolormesh(
    shotXDistEdgeGrid, shotYCenterEdgeGrid, averageDistanceGrid,
    cmap=cmap_depth, shading='flat',
    vmin=depth_vmin, vmax=depth_vmax
)
cbar3 = plt.colorbar(heatmap3, ax=axes[1, 0])
cbar3.set_label('Distance from Goal Line (m)', fontsize=11, labelpad=15)
annotate_cells(axes[1, 0], averageDistanceGrid, shotXDistRange, shotYCenterRange, depth_fmt)
draw_field_lines(axes[1, 0])
axes[1, 0].set_xlabel('Shot Distance from Goal Line (m)', fontsize=11)
axes[1, 0].set_ylabel('Shot Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=10)
axes[1, 0].set_title('Average Goalkeeper Depth Position\nby Shot Location', fontsize=12)
axes[1, 0].set_xlim([shotXDistEdges.min(), shotXDistEdges.max()])
axes[1, 0].set_ylim([shotYCenterEdges.min(), shotYCenterEdges.max()])

# --- Bottom right: Average lateral ---
heatmap4 = axes[1, 1].pcolormesh(
    shotXDistEdgeGrid, shotYCenterEdgeGrid, averageLateralGrid,
    cmap=cmap_lateral, shading='flat',
    vmin=-lat_abs_max, vmax=lat_abs_max
)
cbar4 = plt.colorbar(heatmap4, ax=axes[1, 1])
cbar4.set_label('Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=11, labelpad=15)
annotate_cells(axes[1, 1], averageLateralGrid, shotXDistRange, shotYCenterRange, lateral_fmt)
draw_field_lines(axes[1, 1])
axes[1, 1].set_xlabel('Shot Distance from Goal Line (m)', fontsize=11)
axes[1, 1].set_ylabel('Shot Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=10)
axes[1, 1].set_title('Average Goalkeeper Lateral Position\nby Shot Location', fontsize=12)
axes[1, 1].set_xlim([shotXDistEdges.min(), shotXDistEdges.max()])
axes[1, 1].set_ylim([shotYCenterEdges.min(), shotYCenterEdges.max()])

plt.suptitle('Goalkeeper Positioning by Shot Location\nTop Row: Optimal   |   Bottom Row: Average (Residual = 0)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Calculate difference grid: optimal depth - average (expected) depth
# Positive = optimal is further from goal than average, Negative = optimal is closer to goal
depthDifferenceGrid = optimalDistanceGrid - averageDistanceGrid

# Use a diverging colormap centered at zero
cmap_diff = LinearSegmentedColormap.from_list(
    'custom_dark',
    ['#2d8b2d', '#4a9e1a', '#8b8b00', '#b3a000', '#cc5500', '#b33000', '#8b0000'],
    N=256
    )

diff_abs_max = max(abs(depthDifferenceGrid.min()), abs(depthDifferenceGrid.max()))

fig2, ax2 = plt.subplots(figsize=(10, 7))

heatmap_diff = ax2.pcolormesh(
    shotXDistEdgeGrid, shotYCenterEdgeGrid, depthDifferenceGrid,
    cmap=cmap_diff, shading='flat',
    vmin=-diff_abs_max, vmax=diff_abs_max
)

cbar_diff = plt.colorbar(heatmap_diff, ax=ax2)
cbar_diff.set_label('Optimal Depth minus Expected Depth (m)\n+ = optimal is further from goal, - = optimal is closer to goal',
                    fontsize=11, labelpad=15)

# Annotate cells with signed difference
for i in range(len(shotYCenterRange)):
    for j in range(len(shotXDistRange)):
        value = depthDifferenceGrid[i, j]
        label = f'+{value:.1f}m' if value > 0 else f'{value:.1f}m'
        ax2.text(shotXDistRange[j], shotYCenterRange[i], label,
                 ha='center', va='center', fontsize=9,
                 fontweight='bold', color='white', zorder=6)

draw_field_lines(ax2)

ax2.set_xlabel('Shot Distance from Goal Line (m)', fontsize=11)
ax2.set_ylabel('Shot Distance from Goal Center (m)\nPositive = toward top, Negative = toward bottom', fontsize=10)
ax2.set_title('Depth Positioning Adjustment Needed\n(Optimal minus Expected Keeper Position)', fontsize=13)
ax2.set_xlim([shotXDistEdges.min(), shotXDistEdges.max()])
ax2.set_ylim([shotYCenterEdges.min(), shotYCenterEdges.max()])

plt.tight_layout()
plt.show()

#%%
#--------------------------
#  POSSIBLE IMPROVEMENTS
#--------------------------
# - Include goalkeeper height (more control)
# - Include play pattern (more control)
# - Include open goals (could possibly make lateral positioning significant)
# - Include player position (more control - maybe defenders usually take closer shots)
# - Include number of attackers in attacking area
# - Factor in more locations of defenders and attackers