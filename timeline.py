import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(TV_series="Seinfeld", Start='1989-07-05', Finish='1998-05-14'),
    dict(TV_series="Friends", Start='1994-09-22', Finish='2004-05-06'),
    dict(TV_series="The Office", Start='2005-03-24', Finish='2013-05-16'),
    dict(TV_series="The Big Bang Theory", Start='2007-09-24', Finish='2019-05-16')
])

fig = px.timeline(df, x_start="Start", x_end="Finish", text="TV_series", width=1000, height=500, title="Timeline of selected TV series")
fig.update_xaxes(gridcolor="black")
fig.update_yaxes(visible=False, autorange="reversed") # otherwise tasks are listed from the bottom up
fig.update_traces(insidetextanchor="middle")
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
# fig.show()
fig.write_image("figures/timeline.png", scale=2)