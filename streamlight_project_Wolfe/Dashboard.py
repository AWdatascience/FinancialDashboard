# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:26:41 2023

@author: awolfe
"""

###############################################################################
# FINANCIAL DASHBOARD DRAFT
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st    
from GoogleNews import GoogleNews #used in tab 5
from datetime import datetime, timedelta

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

##############################################################################
# HEADER
###############################################################################
def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    #st.image('https://github.com/AWdatascience/FinancialDashboard/blob/main/streamlight_project_Wolfe/Stocks.png')
    st.image('./img/Stocks.png')
    st.header('Girlboss. Gatekeep. Gaslight.')

    col1, col2, col3 = st.columns([1,4,2]) #width of columns and how many columns to make
    col1.write("Data source:")
    #col2.image('https://github.com/AWdatascience/FinancialDashboard/blob/main/streamlight_project_Wolfe/yahoo_finance.png', width=100)
    col2.image('./img/yahoo_finance.png', width=100)
    
    #update financial dashboard button
    if col3.button("Update data"):
       import yfinance as yf

 
 # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
 #[0] chooses the first table on the page
 #['Symbol'] chooses which column of table to return
 
 # Add the selection boxes
    col1, col2 = st.columns(2)  # Create 2 columns
 # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("My stock pick", ticker_list)

#set time period
    global period
    period = col2.selectbox("Set time period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "MAX"])
    
    
#==============================================================================
# Tab 1
#==============================================================================

#### NEED TO UPDATE SHAREHOLDER TABLE TO DATAFRAME- REMOVE INDEX ADD COLUMN NAMES
def render_tab1():
    """
    This function render the Tab 1 - Company Profile.
    """
    
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """
        return YFinance(ticker).info
    
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
        # Show the company description using markdown + HTML
        st.write('**Company tea:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        #Show the stock overview chart
    def GetStockData(ticker, period):
        stock_df = yf.Ticker(ticker).history(period=period)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df
    if ticker != '':
         stock_price = GetStockData(ticker, period)
    if ticker != '':
        st.write('**Stocks Chart**')       
        fig = go.Figure(data=[go.Scatter(
            x=stock_price.index,
            y=stock_price['Close'],
            mode='lines',
            fill='tozeroy',
            marker=dict(color='deeppink'))])
        st.plotly_chart(fig, use_container_width=True)
        
        
        #create columns for company statistic and profile tables
        col1, col2  = st.columns([1, 1])
        
        with col1:
            # Show some statistics as a DataFrame
            st.write('**Highlights:**')
            info_keys = {'previousClose':'Previous Close',
                         'open'         :'Open',
                         'bid'          :'Bid',
                         'ask'          :'Ask',
                         'volume'       :'Volume',
                         'averageVolume': 'Average Volume',
                         'marketCap'    :'Market Cap',
                         'pegRatio'     :'PEG Ratio'
                         }
            company_stats = {}  # Dictionary
            for key in info_keys:
                company_stats.update({info_keys[key]:info[key]})
            company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
            st.dataframe(company_stats)
            
        
        with col2:
        
            # Show company profile as a DataFrame
            st.write('**Profile:**')
            profile_keys = {'address1':'Street',
                         'city'         :'City',
                         'state'          :'State',
                         'zip'          :'Zip',
                         'country'    :'Country',
                         'phone'       :'Phone',
                         'website'      :'Website',
                         'industry'     :'Industry',
                         'sector'       :'Sector'}
            company_profile = {}  # Dictionary
            for key in profile_keys:
                company_profile.update({profile_keys[key]:info[key]})
            company_profile = pd.DataFrame({'Info':pd.Series(company_profile)})  # Convert to DataFrame
            st.dataframe(company_profile)

        
        col3, col4 = st.columns([2, 3])
        with col3:
            st.write('**Deep Dive Data:**')
            #create additional variables to pull into profile dataframe: DayRange, 52WeekRange, ExDividendDate  
            DayRange = round(info['dayHigh'] - info['dayLow'],2)
            
            fiftytwoRange = info['fiftyTwoWeekHigh'] - info['fiftyTwoWeekLow']
            
            ExDivDate = pd.to_datetime(info['exDividendDate'], unit='s')
            ExDivDate = ExDivDate.strftime('%Y-%m-%d') #drop time element, just keep date
           #create datarame to display variables
            company_data = {
                '     ': ['Day Range', '52 Week Range', 'Ex-Dividend Date', 'Dividend Yield','Trailing EPS'],
                'Value': [DayRange, fiftytwoRange, ExDivDate, info['dividendYield'], info['trailingEps']]
                        }
            company_df = pd.DataFrame(company_data)
            
            st.dataframe(company_df, use_container_width=True, hide_index = True)
        with col4:
            #display shareholder information
            st.write('**Shareholders:**')
            company=yf.Ticker(ticker)
            shares_df = company.major_holders
            shares_df.columns = ['  ', 'Value']
            st.dataframe(shares_df, use_container_width=True, hide_index = True)
            
#==============================================================================
# Tab 2
#==============================================================================
     
def render_tab2():
    """
    This function renders Tab 2 - company financials.
    """
    col1, col2 = st.columns([1, 1])
    sheet = col2.selectbox('Choose Financial Report',['Income Statement', 'Balance Sheet', 'Cash Flow Statement'])
    toggle = col1.select_slider('Choose Time Frame',['Annual', 'Quarterly'])
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """
        return YFinance(ticker).info
    @st.cache_data
    def GetCompanyHistory(ticker):
        """
        gets company history from yahoo finance
        """
        return yf.Ticker(ticker).history(period="1mo")
    
    # If the ticker is already selected
    if ticker != '':
    #get company info based on ticker value to access financial sheet info
        company=yf.Ticker(ticker)
        if toggle == 'Annual':
            if sheet == 'Income Statement':
            #annual income statement
                income = company.income_stmt
                st.subheader(f'{ticker} Annual Income Statement')
                st.write(income)
            elif sheet == 'Balance Sheet':
            #annual balance sheet 
                bs = company.balance_sheet
                st.subheader(f'{ticker} Annual Balance Sheet')
                st.write(bs)
            elif sheet == 'Cash Flow Statement':
            #annual cash flow statement 
                cf = company.cash_flow
                st.subheader(f'{ticker} Annual Cash FLow Statement')
                st.write(cf)
        if toggle == 'Quarterly':
            if sheet == 'Income Statement':
                #quarterly income statement
                qincome = company.quarterly_income_stmt
                st.subheader(f'{ticker} Quarterly Income Statement')
                st.write(qincome)
            elif sheet == 'Balance Sheet':
            #quarterly balance sheet 
                qbs = company.quarterly_balance_sheet
                st.subheader(f'{ticker} Quarterly Balance Sheet')
                st.write(qbs)
            elif sheet == 'Cash Flow Statement':
            #quarterly cash flow statement 
                qcf = company.quarterly_cash_flow
                st.subheader(f'{ticker} Quarterly Cash FLow Statement')
                st.write(qcf)



#==============================================================================
# Tab 3
#==============================================================================

     
def render_tab3():
     """
     This function render the Tab 3 - candlestick/line chart with volume and 50day moving average of the dashboard.
     """  
     
     # Add table to show stock data
     @st.cache_data
     def GetStockData(ticker, period):
         stock_df = yf.Ticker(ticker).history(period=period)
         stock_df.reset_index(inplace=True)  # Drop the indexes
         stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
         return stock_df
 
    #set up all the variables for user to interact with chart - will be called in function to create charts 
     col1, col2, col3, col4, col5 = st.columns(5)
     global start_date
     start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=365))
     global end_date
     end_date = col2.date_input("End date", datetime.today().date()) 
     global plot_type
     plot_type = col3.selectbox("Choose chart", ["Candle", "Line"])
     global interval
     interval = col4.selectbox("Choose interval", ["1D", "1M", "1Y"])
     global duration
     duration = col5.selectbox("Set duration", ["1M", "3M", "6M", "1y", "2y", "5y", "ytd", "MAX"])
    #button to actually run the function 
     if st.button("Click me to show chart"):
         plot_stock_chart(ticker, start_date, end_date, duration, interval, plot_type)
#create plots     
def plot_stock_chart(ticker, start_date, end_date, duration, interval, plot_type):
    # Fetch historical stock data with start and end date considered
        stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Get data based on the selected duration and interval
        stock_data_resampled = stock_data['Close'].resample(duration).ohlc()

    # based on global chart variable, create a candlestick chart or line chart
        if plot_type == 'Candle':
            trace = go.Candlestick(x=stock_data_resampled.index,
                                   open=stock_data_resampled['open'],
                                   high=stock_data_resampled['high'],
                                   low=stock_data_resampled['low'],
                                   close=stock_data_resampled['close'])
        else:
            trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price', marker=dict(color='deeppink')) #make pink to fit theme

        # Add volume bar chart
        trace_volume = go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', yaxis='y2',marker=dict(color='peachpuff'))
    
        # Add 50day moving average (MA) 
        stock_data['MA'] = stock_data['Close'].rolling(window=50).mean()
        trace_ma = go.Scatter(x=stock_data.index, y=stock_data['MA'], mode='lines', name='MA-50', line=dict(dash='dash'), marker=dict(color='pink'))
    
        # Create the layout   
     
        layout = go.Layout(title=f'{ticker} Stock Price Chart',
                       xaxis=dict(title='Date', rangeslider=dict(visible=False)),
                       yaxis=dict(title='Stock Price'),
                       yaxis2=dict(title='Volume', overlaying='y', side='right'),
                       legend=dict(x=0, y=1))

        # Plot the chart
        fig = go.Figure(data=[trace, trace_volume, trace_ma], layout=layout)
        st.plotly_chart(fig)


#==============================================================================
# Tab 4
#==============================================================================
def render_tab4():
    """
    This function renders Tab 4 - Monte Carlo Simulation.
    """
    col1, col2 = st.columns(2)
    
    #set time horizon dropdown
    thlist = (30,60,90)
    time_horizon = col1.selectbox("Time Horizon (days)", thlist)
    
    #set simulation dropdown
    simlist = (200,500,1000)
    n_simulation = col2.selectbox("Simulations", simlist)
    

    @st.cache_data
    def GetStockData(ticker, period):
        stock_df = yf.Ticker(ticker).history(period=period)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df
        
    
    if ticker != '':
         stock_price = GetStockData(ticker, period)
         
    # Calculate some financial metrics for the simulation
    close_price = stock_price['Close']
    # Daily return (of close price)
    daily_return = close_price.pct_change()
    # Daily volatility (of close price)
    daily_volatility = np.std(daily_return)
    
    

    # Run the simulation, set random seed for user
    np.random.seed(50)
    # Run the simulation
    simulation_df = pd.DataFrame()
    
    for i in range(n_simulation):
        
        # The list to store the next stock price
        next_price = []
        
        # Create the next stock price
        last_price = close_price.iloc[-1]
        
        for j in range(time_horizon):
            # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)
    
            # Generate the random future price
            future_price = last_price * (1 + future_return)
    
            # Save the price and go next
            next_price.append(future_price)
            last_price = future_price
        
        # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
            
        
    # Plot the Monte Carlo simulation
    fig, ax = plt.subplots()
    for i in range(n_simulation):
        ax.plot(simulation_df['sim' + str(i)], lw=0.8, alpha=0.7)
    
    ax.axhline(y=close_price.iloc[-1], color='red')
    ax.set_xlabel('Time Horizon (days)')
    ax.set_ylabel('Stock Price')
    ax.legend(['Current stock price is: ' + str(np.round(close_price.iloc[-1], 2))])
    ax.get_legend().legend_handles[0].set_color('red')
    #remove standard black border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #remove white background
    ax.set_facecolor('none')
    fig.set_facecolor('none')
    # Show the Matplotlib figure using st.pyplot
    st.pyplot(fig)
    
    #Show the Value At Risk
    #get ending price of current simulation
    ending_price = simulation_df.iloc[-1:, :].values[0, ]
    # Price at 95% confidence interval
    future_price_95ci = np.percentile(ending_price, 5)
    
   
#also display as a histogram for further clarity for the user
    st.subheader("Let's look at this another way...")
     
    fig2, ax2 = plt.subplots()
    ax2.hist(ending_price, bins=50, color = 'pink')
    ax2.axvline(x=close_price.iloc[-1], color='red')
    ax2.legend(['Current stock price is: ' + str(np.round(close_price.iloc[-1], 2))])
    ax2.get_legend().legend_handles[0].set_color('red')
    #remove standard black border
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #remove white background
    ax2.set_facecolor('none')
    fig2.set_facecolor('none')
     
    st.pyplot(fig2)
    
    # Value at Risk
   
    global VaR
    VaR = close_price.iloc[-1] - future_price_95ci
    st.subheader('Value at Risk at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD') 
   

#==============================================================================
# Tab 5
#==============================================================================

######♣ NEED TO FORMAT NEWS API!!!!!!!!!!!!!!!!

def render_tab5():
    """
    This function renders Tab 4 - additional analysis: my recommendation based on monte carlo output and recent company news
    """
    st.subheader("Wondering what you should do with all this information? Here's what I would do:")
    
    if VaR >= 10:
        st.markdown('<div style="text-align: center;">' + \
                    'BUY BUY BUY!!!' + \
                    '</div><br>',
                    unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;">' + \
            '<iframe src="https://giphy.com/embed/10nCUgty95Oa0U" width="480" height="312" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/pokemon-adorable-hearts-10nCUgty95Oa0U">via GIPHY</a></p>',
            unsafe_allow_html=True
                    )
    elif (VaR > 0) & (VaR < 10):
        st.markdown('<div style="text-align: center;">' + \
                    'Buy at your own risk' + \
                    '</div><br>',
                    unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;">' + \
            '<iframe src="https://giphy.com/embed/MziKDo6gO7x8A" width="480" height="360" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/pokemon-happy-smile-MziKDo6gO7x8A">via GIPHY</a></p>',
            unsafe_allow_html=True
                    )
    else:
        st.markdown('<div style="text-align: center;">' + \
                    'DO NOT BUY UNLESS YOU WANT TO LOSE MONEY' + \
                    '</div><br>',
                    unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;">' + \
            '<iframe src="https://giphy.com/embed/UsckMQeJIxpva08C9g" width="480" height="271" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/pokemon-anime-angry-meowth-UsckMQeJIxpva08C9g">via GIPHY</a></p>',
            unsafe_allow_html=True
                    )
    
    st.subheader("Don't take my word for it though, do more research!")

#display news articles about current selected ticker using GoogleNews
    @st.cache_data
    def GetStockData(ticker, period):
        stock_df = yf.Ticker(ticker).history(period=period)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df

##### NOTE USED STACKOVERFLOW TO FIGURE OUT getting data from GoogleNews: https://stackoverflow.com/questions/73759782/how-to-have-googlenews-package-return-a-list-of-news-articles-about-a-list-of-st
   
    now = datetime.today().date() 
    now = now.strftime('%m-%d-%Y')
    yesterday = datetime.today() - timedelta(days = 1)
    yesterday = yesterday.strftime('%m-%d-%Y')
    
    googlenews = GoogleNews(start=yesterday, end=now)
    googlenews.search(ticker)
    result = googlenews.result()
    
    #filter results to only show Title, Data and URL of articles pulled
    if result:
        articles = [{'Title': article['title'],
                     'Date': article['date'],
                     'URL': article['link']} for article in result]
        DFarticles = pd.DataFrame(articles)
        st.dataframe(DFarticles[['Title', 'Date', 'URL']],use_container_width=True, hide_index = True)
    else:
        st.write("Sorry, girlie! There aren't any news articles on f'{ticker}")

#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Company ☕", "Company ☕ ✨numbers ed.✨", "Stocks in Color", "Manifesting", "My Thoughts" ])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
