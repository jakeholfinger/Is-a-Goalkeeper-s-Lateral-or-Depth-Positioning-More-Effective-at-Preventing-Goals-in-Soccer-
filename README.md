# Is a Goalkeeper's Lateral or Depth Positioning More Effective at Preventing Goals in Soccer?

Research project by **Jake Holfinger** (Computer Science & Engineering and Data Analytics), prepared for the **OSU Sports Analytics Conference 2026**.

## Problem Statement

When an attacker takes a shot in soccer, the goalkeeper positions themselves to minimize the chance of a goal. In doing so, they must manage two key spatial dimensions: their **lateral position** and their **depth** from the goal line. It is unclear which one matters more for preventing goals.

This project builds a machine learning model in Python that estimates how goalkeeper lateral and depth positioning affect the probability of a goal. The results can inform goalkeeper coaching by identifying which positional direction goalkeepers should prioritize, and what their optimal position should be, in order to reduce goals conceded.

## Dataset

- Sourced from the [StatsBomb Open Data](https://github.com/statsbomb/open-data) GitHub repo.
- The dataset contains event and tracking data for thousands of games (only event data was used).
- ~47,000 shots were retrieved via API calls.

## Methodology

1. Retrieved shot data using API calls (~47,000 shots).
2. Cleaned and filtered the data, adding new variables.
3. Performed linear regressions between relevant variables and goalkeeper position.
4. Computed lateral and depth residuals to isolate goalkeeper positioning from confounding factors.
5. Fit logistic regression models using the residuals to estimate goal probability.
6. Generated synthetic data to isolate both residuals.
7. Plotted goal probability vs. residual, built a bar chart to visualize the impact of lateral vs. depth positioning, and created heatmaps to show optimal depth positions across shot locations.

## Results

- Goalkeeper **depth positioning** has a larger effect on goal probability than lateral positioning.
- The effect of depth positioning in the logistic regression is statistically significant (**p < 0.001**), while the effect of lateral position is not (**p = 0.859**).
- For an average shot (18.5 m from goal, 26° shooting angle), depth positioning can change goal probability by **16.69%** within plausible values.
- Lateral positioning, in contrast, can only change goal probability by **0.09%**, and this value is not statistically significant.
- The optimal goalkeeper depth for an average shot is **1.71 m** from the expected position.

## Implication

Goalkeepers and coaches should prioritize optimal **depth positioning** over lateral positioning, since depth has a far larger and statistically significant effect on preventing goals.

## Repository Contents

- `OSU Sports Analytics Research Poster Spring 2026 - Jake Holfinger.pdf` / `.jpg` — the research poster summarizing the problem, methodology, results, and implications.
- `2026 Sports Analytics Conference Program.pdf` — the program for the OSU Sports Analytics Conference 2026, where this research was presented.
- `IMG-Average_GK_Lateral_Position_Heatmap.png` — average goalkeeper lateral position by shot location.
- `IMG-Depth_Positioning_Adjustment_Heatmap.png` — depth adjustment needed (optimal minus expected keeper position) by shot location.
- `IMG-Effect_Of_Depth_Positioning_On_Goal_Probability_Line_Chart.png` — predicted goal probability as a function of depth positioning error, for an average shot.
- `IMG-Effect_Of_Positioning_Errors_On_Goal_Probability.png` — bar chart comparing the total effect of depth vs. lateral positioning errors on goal probability.
- `IMG-Optimal_GK_Depth_Position.png` — heatmap of optimal goalkeeper depth position by shot location.
- `requirements.txt` — Python dependencies for the analysis.
- `Code/` — analysis notebooks and scripts (the primary working files are excluded from version control via `.gitignore`).

## Run Instructions

1. Clone the repo and `cd` into it.
2. Create and activate a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the analysis:
   - As a notebook: `jupyter notebook` and open the `.ipynb` file in `Code/`.
   - As a script: `python "Code/Is_A_Goalkeepers_Lateral_or_Depth_Positioning_More_Effective_at_Preventing_Goals_in_Soccer_With_Calculator.py"`.

   Both pull shot data directly from the StatsBomb Open Data API via `statsbombpy`, so an internet connection is required and no local dataset needs to be downloaded beforehand.

## Sources

Jupyter Notebook (coding), GitHub (to access StatsBomb Open Data), Claude (helped overall process).
