from typing import Sequence
import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"
deepseek_group = "Deepseek"

"""
model	Professor
ITC: DeepSeek R1	59.4% (± 6.4%)
ITC: Gemini	68.2% (± 9.8%)
ITC: Qwen	46.9% (± 7.3%)
Claude-3.5-Sonnet	6.7% (± 5.2%)
Deepseek-Chat-v3	6.5% (± 3.6%)
GPT-4o	2.4% (± 3.3%)
Gemini-2.0-Flash-Exp	13.0% (± 6.2%)
Grok-2-1212	4.9% (± 4.2%)
Qwen-72b-Instruct	5.3% (± 3.6%)
"""

## Average per model
MANUAL_DATA = [
    {
        "Model": "Gemini<br>2.0",
        "Articulation Rate": 2.27,
        "Type": "Traditional Model",
        "Group": google_group,
    },
    {
        "Model": "Gemini-<br>Thinking",
        "Articulation Rate": 33.01,
        "Type": "Thinking Model",
        "Group": google_group,
    },
    {
        "Model": "DeepSeek-R1",
        "Articulation Rate": 24.67,
        "Type": "Thinking Model",
        "Group": deepseek_group,
    },
    {
        "Model": "DeepSeek-V3",
        "Articulation Rate": 1.37,
        "Type": "Traditional Model",
        "Group": deepseek_group,
    },
]


def plot_manual_stanford_fig_1(data: Sequence[dict] = []):
    if not data:
        data = MANUAL_DATA
    df = pd.DataFrame(data)

    # Create bar plot with error bars
    fig = px.bar(
        df,
        x="Model",
        y="Articulation Rate",
        color="Type",
        # error_y="Error",
        # barmode="group",
        #  title='Articulation Rate by Model Type',
        hover_data=["Model"],
        text="Articulation Rate",
    )

    # add text labels by model
    # fig.update_traces(textposition='outside', texttemplate='%{text:.1f}%', text=df['Articulation Rate'])
    # padd = "&nbsp;" * 8
    padd = ""

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
            range=[0, 35],  # Set y-axis range from 0 to 4
            # y ticks [0, 5, 10, 15, 20, 25, 30, 35]
            tickvals=[0, 5, 10, 15, 20, 25, 30, 35],
        ),
    )

    font_size = 24
    trace_font = dict(size=int(font_size * 0.8))
    # don't show decimal places
    fig.update_traces(textposition="outside", texttemplate=padd + "%{text:.0f}%", textfont=trace_font)
    fig.update_layout(
        yaxis_title=None,
        title={
            "text": "How often do models articulate a factor<br>that influences their decision? (%)",
            "x": 0.5,
            # 'y': 0.973,
            "xanchor": "center",
            "yanchor": "top",
            # 'bgcolor': '#FFFFFF'  # White background color
        },
        xaxis_title=None,
        title_font=dict(size=int(font_size * 1.3)),
        # xaxis_title="How often do models articulate a<br>factor that influences their decision? (%)",
        width=800,
        height=800,
        showlegend=True,
        font=dict(size=font_size),
        legend=dict(title="", yanchor="bottom", y=0.74, xanchor="left", x=0, font=dict(size=font_size)),
    )
    # add top margin
    fig.update_layout(margin=dict(t=200))

    fig.show()
    # hack to fix mathjax

    pio.kaleido.scope.mathjax = None
    pdf_name = "articulation_tweet.pdf"
    # remove margins
    fig.write_image(pdf_name)
    # write to png 1200 * 800
    png_name = "articulation_tweet.png"
    fig.write_image(png_name, width=800, height=800)


if __name__ == "__main__":
    plot_manual_stanford_fig_1()
