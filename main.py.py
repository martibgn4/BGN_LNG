import requests
import datetime
import pandas as pd
import plotly.express as px
import dash
from dash import Dash, dcc, html


grokn_token = "36Hl0cc4t8HQ1EjZLh9utMNTvdo_6CPQKyet16154uLDCE7c"


# Step 1: Fetch data from ZEMA API
def get_forward_curve(market):
    # url = "https://zema-api-endpoint/forwardcurve"
    # params = {"market": market, "curveType": "Forward", "dateRange": "latest"}
    # response = requests.get(url, params=params, auth=("user", "password"))
    # data = response.json()
    columns = ["TTF", "NBP", "THE"]
    dates = [datetime.date(2025, 3, i) for i in range(2, 8)]
    data = {
        "TTF": pd.DataFrame(data={"Date": dates, "Price": [10, 11, 12, 13, 14, 15], "Market": "TTF"}),
        "NBP": pd.DataFrame(data={"Date": dates, "Price": [20, 21, 22, 23, 24, 25], "Market": "NBP"}),
        "THE": pd.DataFrame(data={"Date": dates, "Price": [30, 31, 32, 33, 34, 35], "Market": "THE"})
    }
    if market not in data:
        raise ValueError(f"No such market '{market}' available")
    return data[market]

# Step 2: Prepare data
markets = ["TTF", "NBP", "THE"]
data_dict = {m: get_forward_curve(m) for m in markets}
# Combine all data into one DataFrame
df_all = pd.concat(data_dict.values())


# Step 3: Build dashboard
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Forward Curve Dashboard"),
    dcc.Dropdown(
        id="market-select",
        options=[{"label": m, "value": m} for m in markets],
        value=["TTF"],  # default selection
        multi=True  # Enable multi-select
    ),
    dcc.Graph(id="curve-chart")
])



@app.callback(
    dash.dependencies.Output("curve-chart", "figure"),
    [dash.dependencies.Input("market-select", "value")]
)
def update_chart(selected_markets):
    # df = data_dict[selected_market]
    # fig = px.line(df, x="Date", y="Price", title=f"{selected_market} Forward Curve")
    # return fig

    filtered_df = df_all[df_all["Market"].isin(selected_markets)]
    fig = px.line(filtered_df, x="Date", y="Price", color="Market",
                  title="Forward Curves", markers=True)
    return fig


if __name__ == "__main__":
    update_chart(["TTF", "NBP"])
    app.run(debug=True)