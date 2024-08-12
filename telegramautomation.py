import pandas as pd
import json
import requests
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import yfinance as yf

# URL of the CSV data
csv_url = "https://raw.githubusercontent.com/pk367/my_demand_zone_ohlc_data/main/ohlc_data.csv"

# Load the data into a DataFrame
def load_data():
    return pd.read_csv(csv_url)

data = load_data()

# Function to fetch real-time stock price using yfinance
def get_real_time_price(ticker):
    ticker = ticker + '.ns'
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    return data['Close'].iloc[-1] if not data.empty else None

# Function to send message to Telegram channel
def send_telegram_message(bot_token, channel_username, message, image_path=None):
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto' if image_path else f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': channel_username,
        'caption': message,
        'parse_mode': 'Markdown'  # Specify Markdown for text formatting
    }
    
    if image_path:
        with open(image_path, 'rb') as image_file:
            requests.post(url, data=payload, files={'photo': image_file})
    else:
        requests.post(url, data=payload)

# Function to plot the candlestick chart
def plot_candlestick(ticker, zone_date, entry_price, close_price, ohlc_data):
    """Plot the candlestick chart for the given Ticker."""
    if ohlc_data is None or ohlc_data.empty:
        return None

    # Ensure the 'Date' column is in datetime format
    if 'Date' in ohlc_data.columns:
        ohlc_data['Date'] = pd.to_datetime(ohlc_data['Date'])
        ohlc_data.set_index('Date', inplace=True)
    elif 'Datetime' in ohlc_data.columns:
        ohlc_data['Date'] = pd.to_datetime(ohlc_data['Datetime'])
        ohlc_data.set_index('Date', inplace=True)

    # Create the candlestick chart
    mpf.plot(ohlc_data, type='candle', style='charles', title=ticker, ylabel='Price', savefig='candlestick_chart.png')

    # Mark the zone date on the chart
    plt.axvline(pd.to_datetime(zone_date), color='red', linestyle='--', label='Zone Date')
    plt.legend()
    plt.close()  # Close the plot to avoid displaying it in an interactive environment

# Replace with your bot token and channel username
bot_token = '7275933222:AAGJ47F5MV2Ard81sRddM7rs1dfrtRJPO3k'
channel_username = '@drravirkumar_scanner'

# Filter the data based on selected stocks
filtered_data = data[data['Zone Distance'] < 10]

for index, row in filtered_data.iterrows():
    ticker = row['Ticker']
    zone_date = row['Zone Date']
    entry_price = row['Entry Price']
    stop_loss = row['Stop Loss']
    zone_type = row['Zone Type']
    time_frame = row['Time Frame']
    
    # Get the current price
    current_price = get_real_time_price(ticker)
    if current_price is None:
        print(f"Could not fetch current price for {ticker}.")
        continue

    # Calculate price difference percentage
    price_difference = abs(current_price - entry_price) / entry_price * 100

    # Check if the price difference is less than or equal to 3%
    if price_difference <= 3:
        # Fix JSON loading
        try:
            # Replace single quotes with double quotes for valid JSON
            ohlc_data = row['OHLC Data'].replace("'", "\"")
            ohlc_data = json.loads(ohlc_data)
            ohlc_df = pd.DataFrame(ohlc_data)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for {ticker}: {e}")
            continue

        # Plot the candlestick chart using the custom function
        plot_candlestick(ticker, zone_date, entry_price, current_price, ohlc_df)

        # Prepare dynamic message
        if price_difference <= 1:
            price_comment = "📈 *Excellent opportunity!*"
        elif price_difference <= 2:
            price_comment = "📊 *Good opportunity, consider acting soon!*"
        elif price_difference <= 3:
            price_comment = "🔍 *Monitor closely, it's near the zone.*"
        else:
            price_comment = "⚠️ *Price is outside the ideal range.*"

        message = (
            f"{price_comment}\n"
            f"📈 *{ticker}* स्टॉक में {time_frame} के चार्ट टाइमफ्रेम पर {zone_date} डेटटाइम को {zone_type} डिमांड जोन का पैटर्न बना हुआ है।\n"
            f"📊 अभी इस स्टॉक का प्राइस *₹{current_price}* है, तथा हमारे जोन का एंट्री प्राइस *₹{entry_price}* है।\n"
            f"👉 यह स्टॉक डिमांड जोन से सिर्फ *{price_difference:.2f}%* दूरी पर है।\n"
            f"📉 अगर आपको इस जोन में Love at first side दिख रहा है यानी आपको लगता है कि यह जोन चलेगा तो आप *₹{stop_loss}* पर स्टॉपलॉस लगाकर इस ट्रेंड में एंट्री ले सकते हैं।\n"
            f"----------------------------------\n"
            f"📈 *संक्षिप्त में:* \n"
            f"📈 *Ticker:* {ticker}\n"
            f"🕒 *Zone Date:* {zone_date}\n"
            f"💰 *Entry Price:* ₹{entry_price}\n"
            f"🚨 *Stop Loss:* ₹{stop_loss}\n"
            f"⏳ *Time Frame:* {time_frame}\n"
            f"🟢 *Zone Type:* {zone_type}\n"
            "----------------------------------"
        )

        # Send the message and the chart to Telegram
        send_telegram_message(bot_token, channel_username, message, 'candlestick_chart.png')

        # Clean up the image file
        if os.path.exists('candlestick_chart.png'):
            os.remove('candlestick_chart.png')

print("✅ Finished checking stocks.")
