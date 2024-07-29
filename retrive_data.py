
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Title of the app
st.markdown(f"<h1 style=\"text-align: center;\">Welcome to demand zone scanner</h1>", unsafe_allow_html=True)
st.markdown('<hr style="border: 1px solid rainbow;">', unsafe_allow_html=True)    

# URL of the CSV data on GitHub
url = 'https://raw.githubusercontent.com/pk367/demand_zone_ohlc_data/main/ohlc_data.csv'

def plot_candlestick(Ticker, Zone_Date, Entry_Price, Stop_Loss, Time_Frame, Zone_Type, Close_Price, OHLC_Data):
    """Plot the candlestick chart for the given Ticker."""
    try:
        if OHLC_Data is None or OHLC_Data.empty:
            return None

        stock_data = []
        annotations = []

        # Determine date format based on Time_Frame
        date_format = '%Y-%m-%d' if Time_Frame in ['1d', '1wk', '1mo', '3mo'] else '%Y-%m-%d %H:%M:%S'

        # Collect data and create annotations
        for index, row in OHLC_Data.iterrows():
            formatted_date = index.strftime(date_format)
            stock_data.append({
                "time": formatted_date,
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
            })

            if index == OHLC_Data.index[9]:
                annotations.append({
                    'x': formatted_date,
                    'y': row['Low'],
                    'text': f"Zone Date <br> {Zone_Date}",
                    'showarrow': True,
                    'arrowhead': 2,
                    'arrowsize': 1,
                    'arrowwidth': 2,
                    'arrowcolor': "blue",
                    'ax': 20,
                    'ay': 30,
                    'font': dict(size=12, color="black"),
                    'bgcolor': 'rgba(255, 255, 0, 0.5)'  # light yellow
                })

        fig = go.Figure(data=[go.Candlestick(
            x=[item["time"] for item in stock_data],
            open=[item['open'] for item in stock_data],
            high=[item['high'] for item in stock_data],
            low=[item['low'] for item in stock_data],
            close=[item['close'] for item in stock_data],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            line_width=0.5)])

        # Add annotations
        fig.update_layout(annotations=annotations)

        # Additional calculations for high, low, and target
        index_9 = OHLC_Data.index[9]
        open_9 = OHLC_Data.loc[index_9, 'Open']
        close_9 = OHLC_Data.loc[index_9, 'Close']
        high_9 = OHLC_Data.loc[index_9, 'High']
        low_9 = OHLC_Data.loc[index_9, 'Low']

        if abs(open_9 - close_9) > 0.50 * (high_9 - low_9):
            high_price = max(open_9, close_9)
        else:
            high_price = high_9

        if OHLC_Data.loc[OHLC_Data.index[8], 'Open'] > OHLC_Data.loc[OHLC_Data.index[8], 'Close']:
            low_price = min(OHLC_Data.loc[OHLC_Data.index[8], 'Low'], low_9, OHLC_Data.loc[OHLC_Data.index[10], 'Low'])
        else:
            low_price = min(low_9, OHLC_Data.loc[OHLC_Data.index[10], 'Low'])

        total_risk = high_price - low_price
        minimum_target = (total_risk * 5) + high_price

        # Add rectangular shape
        shape_start = OHLC_Data.index[8].strftime(date_format)
        shape_end = OHLC_Data.index[-1].strftime(date_format)

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=shape_start,
            y0=low_price,
            x1=shape_end,
            y1=high_price,
            fillcolor="green",
            opacity=0.2,
            layer="below",
            line=dict(width=0),
        )

        # Add entry and stop loss annotations
        specified_index = OHLC_Data.index[min(17, len(OHLC_Data.index) - 1)].strftime(date_format)

        fig.add_annotation(
            x=specified_index,
            y=high_price,
            text=f"{Ticker} entry at: {high_price}",
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            align="center",
            valign="bottom",
            yshift=10,
        )

        fig.add_annotation(
            x=specified_index,
            y=low_price,
            text=f"Stoploss at : {low_price}",
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            align="center",
            valign="top",
            yshift=-10,
        )

        # Add annotation for minimum_target
        fig.add_annotation(
            x=OHLC_Data.index[-1].strftime(date_format),
            y=minimum_target,
            text=f"<b>1:5 Target: {minimum_target:.2f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=-20,
            ay=-30,
            font=dict(size=12, color="black"),
            bgcolor='rgba(0, 255, 0, 0.3)'  # Light green background for entry price
        )

        # Custom tick labels with line breaks
        custom_ticks = [item["time"].replace(' ', '<br>') for item in stock_data]

        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                type='category',
                ticktext=custom_ticks,
                tickvals=[item["time"] for item in stock_data],
                tickangle=0,
                tickmode='array',
                tickfont=dict(size=10)
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=100, b=40),
            legend=dict(x=0, y=1.0),
            font=dict(size=12),
            height=600,
            width=800,
            dragmode=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis_fixedrange=True,
            yaxis_fixedrange=True
        )

        # Add styled HTML text annotation
        header_text = (
            f"<span style='padding-right: 20px;'><b>📊 Chart:</b> {Ticker}</span>"
            f"<span style='padding-right: 20px;'><b>🕒 Close:</b> ₹ {Close_Price}</span>"
            f"<span style='padding-right: 20px;'><b>🟢 Zone:</b> {Zone_Type}</span>"
            f"<span><b>⏳ Time frame:</b> {Time_Frame}</span>"
        )

        fig.add_annotation(
            x=0.5,
            y=1.20,
            text=header_text,
            showarrow=False,
            align='center',
            xref='paper',
            yref='paper',
            font=dict(size=17, color='white'),
            bgcolor='rgba(0, 0, 0, 0.8)',
            borderpad=4,
            width=800,
            height=50,
            valign='middle'
        )     
        return fig
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def filter_dataframe(patterns_df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters")

    if not modify:
        return patterns_df

    df = patterns_df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

@st.cache_data(ttl=5)
def load_data():
    # Read the CSV data from the URL
    data = pd.read_csv(url)
    return data

# Load the data
data = load_data()

# Extract relevant information
pattern_data = []
for index, row in data.iterrows():
    # Check if 'OHLC Data' exists and is well-formed
    ohlc_data_str = row['OHLC Data']
    ohlc_df = pd.DataFrame(eval(ohlc_data_str))  # Convert string to DataFrame

    # Handle different keys for date and datetime
    if 'Date' in ohlc_df.columns:
        ohlc_df.rename(columns={'Date': 'Datetime'}, inplace=True)
    elif 'Datetime' not in ohlc_df.columns:
        st.warning(f"No 'Date' or 'Datetime' column found for {row['Ticker']}. Skipping this record.")
        continue
    
    ohlc_df.set_index('Datetime', inplace=True)
    ohlc_df.index = pd.to_datetime(ohlc_df.index)

    # Append the pattern data without OHLC Data
    pattern_data.append({
        "Ticker": row["Ticker"],
        "Close Price": row["Close Price"],
        "Zone Distance": row["Zone Distance"],
        "Time Frame": row["Time Frame"],
        "Zone Date": row["Zone Date"],
        "Entry Price": row["Entry Price"],
        "Stop Loss": row["Stop Loss"],
        "Zone Type": row["Zone Type"],
        "OHLC Data": ohlc_df  # Keep for plotting
    })

# Create a DataFrame from the summary data
patterns_df = pd.DataFrame(pattern_data)
patterns_df = patterns_df.sort_values(by='Zone Distance', ascending=True)
filtered_df = filter_dataframe(patterns_df)

# Create a DataFrame for the table view without OHLC Data
tab1, tab2 = st.tabs(["📁 Data", "📈 Chart"])

with tab1:
    st.markdown("**Table View**")
    filtered_dfff = filtered_df.drop(columns=['OHLC Data'])
    filtered_dfff['Zone Distance'] = filtered_dfff['Zone Distance'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    filtered_dfff = filtered_dfff.reset_index(drop=True)
    st.dataframe(filtered_dfff)

# Inside the tab for plotting
with tab2:
    st.markdown("**Chart View**")
    for index, row in filtered_df.iterrows():
        OHLC_Data = row['OHLC Data']

        # Call the plot function
        fig = plot_candlestick(
            Ticker=row['Ticker'],
            Zone_Date=row['Zone Date'],
            Entry_Price=row['Entry Price'],
            Stop_Loss=row['Stop Loss'],
            Time_Frame=row['Time Frame'],
            Zone_Type=row['Zone Type'],
            Close_Price=row['Close Price'],
            OHLC_Data=OHLC_Data
        )

        # Display the figure
        if fig is not None:
            st.plotly_chart(fig)
