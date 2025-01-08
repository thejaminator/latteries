import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"


data = [
    {
        "Model": "Claude-3.5-sonnet",
        "Articulation Rate": 6.7,
        "Type": "Non-ITC",
        "Group": "Claude<br>Sonnet",
        "Error": 5.2,
    },
    {
        "Model": "GPT-4o",
        "Articulation Rate": 2.4,
        "Type": "Non-ITC",
        "Group": "GPT-4o",
        "Error": 2.4,
    },
    {
        "Model": "Llama<br>70b",
        "Articulation Rate": 2.4,
        "Type": "Non-ITC",
        "Group": "Llama<br>70b",
        "Error": 2.4,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 4.8,
        "Type": "Non-ITC",
        "Group": alibaba_group,
        "Error": 4.1,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 45.3,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": alibaba_group,
        "Error": 9.0,
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 14.8,
        "Type": "Non-ITC",
        "Group": google_group,
        "Error": 7.7,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 50.6,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": google_group,
        "Error": 11.2,
    },
]
df = pd.DataFrame(data)

# Create bar plot with error bars
fig = px.bar(
    df,
    x="Group",
    y="Articulation Rate",
    color="Type",
    error_y="Error",
    barmode="group",
    #  title='Articulation Rate by Model Type',
    hover_data=["Model"],
    text="Articulation Rate",
)

# add text labels by model
# fig.update_traces(textposition='outside', texttemplate='%{text:.1f}%', text=df['Articulation Rate'])
padd = "&nbsp;" * 8
trace_font = dict(size=20)
# don't show decimal places
fig.update_traces(textposition="outside", texttemplate=padd + "%{text:.0f}%", textfont=trace_font)


# Set white background and add spines
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
    ),
)

# Add secondary x-axis with model names
fig.update_layout(
    xaxis2=dict(
        ticktext=df["Model"],
        tickvals=list(range(len(df))),
        overlaying="x",
        side="bottom",
        title="Model",
        anchor="y",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
)

# remove x-axis "Group" and configure text display
fig.update_layout(
    xaxis_title=None,
    yaxis_title="",
    width=600,
    height=250,
    showlegend=False,
    font=dict(size=16),  # Set font size to 14
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "articulation_professor_more.pdf"
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
