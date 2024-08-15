import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# Setting the working directory to the root of the project

current_dir = Path(__file__).parent
base_dir = current_dir.parent
data_dir = base_dir / "DataForStreamlit"


# Load the user profiles data
user_profiles = pd.read_excel(data_dir / "User_Profiles_Reports.xlsx")

# Load MCC mapping data
mcc_mapping = pd.read_excel(data_dir / "MCC_Details.xlsx")

# Convert MCC in mcc_mapping to string
mcc_mapping['MCC'] = mcc_mapping['MCC'].astype(str)

# Function to check if a column name is numeric (assuming MCC columns are numeric)
def is_numeric_column(col_name):
    try:
        int(col_name)
        return True
    except ValueError:
        return False

# Identify the MCC columns by checking if the column names are numeric
mcc_columns = [col for col in user_profiles.columns if is_numeric_column(col)]

mcc_amounts_cols = [col for col in user_profiles.columns.astype(str) if col.startswith('total_amount_mcc_')]

# Mapping dictionaries
day_mapping = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
    4: "Friday", 5: "Saturday", 6: "Sunday"
}

month_mapping = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

season_mapping = {
    1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"
}

#user interest score calculation function
def calculate_user_interest_score(user_id):
    user_data = user_profiles[user_profiles['FK_BusinessUserId'] == user_id]
    
    total_frequencies = user_data[mcc_columns].sum(axis=1).values[0]
    total_paid_amount = user_data[mcc_amounts_cols].sum(axis=1).values[0]
    
    mcc_scores = {}
    if total_frequencies > 0 and total_paid_amount > 0:
        for mcc in mcc_columns:
            frequency = user_data[mcc].values[0]
            amount_col = f'total_amount_mcc_{mcc}'
            if amount_col in user_data.columns:
                amount = user_data[amount_col].values[0]
                if frequency > 0 and amount > 0:
                    score = 5 * (frequency / total_frequencies) * (amount / total_paid_amount)
                    mcc_scores[mcc] = score
    
    return mcc_scores




# Function to display user profile information
def display_user_profile(user_id):
    user_data = user_profiles[user_profiles['FK_BusinessUserId'] == user_id].iloc[0]

    # Mapping numeric values to meaningful names
    most_active_day = day_mapping.get(user_data['most_common_day_of_week'], "Unknown")
    most_active_month = month_mapping.get(user_data['most_common_month'], "Unknown")
    most_common_season = season_mapping.get(user_data['most_common_season'], "Unknown")

    # Creating 3x3 grid for displaying the metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Transactions", value=user_data['total_transactions'])
        st.metric(label="Average Points Rewarded", value=f"{user_data['avg_points_rewarded']:.2f}")
        st.metric(label="Most Active Day", value=most_active_day)
    
    with col2:
        st.metric(label="Total Amount Spent (KWD)", value=f"{user_data['total_amount_spent']:.2f}")
        st.metric(label="Average Amount Spent (KWD)", value=f"{user_data['avg_amount_spent']:.2f}")
        st.metric(label="Most Active Month", value=most_active_month)
    
    with col3:
        st.metric(label="Recency (Days since last transaction)", value=user_data['recency'])
        st.metric(label="Most Common Season", value=most_common_season)
        st.empty()  # To maintain the 3x3 layout

def plot_frequent_mcc(user_id):
    user_data = user_profiles[user_profiles['FK_BusinessUserId'] == user_id]
    mcc_data = user_data[mcc_columns].sum(axis=0)
    mcc_data = mcc_data[mcc_data > 0]
    
    # Converting MCC to string for merging
    mcc_data.index = mcc_data.index.astype(str)
    
    # Merge with MCC mapping
    mcc_data = mcc_data.reset_index()
    mcc_data.columns = ['MCC', 'Frequency']
    mcc_data = mcc_data.merge(mcc_mapping, on='MCC', how='left')
    
    
    mcc_data['Label'] = mcc_data['MCC'] + ' - ' + mcc_data['Detailed MCC']
    
    fig = px.pie(
        values=mcc_data['Frequency'], 
        names=mcc_data['Label'], 
        title=f'Frequent MCC for User ID: {user_id}'
    )
    fig.update_layout(
        showlegend=True,
        width=1000,
        height=500
    )
    st.plotly_chart(fig)


def plot_top_mcc_by_amount(user_id):
    user_data = user_profiles[user_profiles['FK_BusinessUserId'] == user_id]
    # Select columns that represent total amounts spent in different MCCs
    mcc_amounts = user_data[mcc_amounts_cols].iloc[0]
    
    # Filter out zero values to keep only the MCCs with spending
    mcc_amounts = mcc_amounts[mcc_amounts > 0]
    
    # Convert the index to string to facilitate merging with MCC mapping
    mcc_amounts.index = mcc_amounts.index.str.replace('total_amount_mcc_', '').astype(str)
    
    # Convert the Series to a DataFrame
    mcc_amounts = mcc_amounts.reset_index()
    mcc_amounts.columns = ['MCC', 'Total_Amount']
    
    # Merge the MCC amounts with the MCC mapping data
    mcc_amounts = mcc_amounts.merge(mcc_mapping, on='MCC', how='left')
    
    # Create a label combining MCC and its description
    mcc_amounts['Label'] = mcc_amounts['MCC'] + ' - ' + mcc_amounts['Detailed MCC']
    
    # Create the pie chart using Plotly
    fig = px.pie(
        values=mcc_amounts['Total_Amount'], 
        names=mcc_amounts['Label'], 
        title=f'Top MCC by Total Amount for User ID: {user_id}'
    )
    fig.update_layout(
        showlegend=True,
        width=1000,
        height=500
    )
    st.plotly_chart(fig)

def plot_user_interest_score_by_mcc(user_id):
    mcc_scores = calculate_user_interest_score(user_id)

    if not mcc_scores:
        st.warning(f"User ID: {user_id} has no transactions or interest scores to display.")
        return

    # Merge MCC scores with MCC mapping
    mcc_scores_df = pd.DataFrame.from_dict(mcc_scores, orient='index', columns=['Score'])
    mcc_scores_df.reset_index(inplace=True)
    mcc_scores_df.columns = ['MCC', 'Score']
    mcc_scores_df['MCC'] = mcc_scores_df['MCC'].astype(str)  # Convert MCC to string
    mcc_scores_df = mcc_scores_df.merge(mcc_mapping, on='MCC', how='left')

    # Create labels combining MCC and description
    mcc_scores_df['Label'] = mcc_scores_df['MCC'] + ' - ' + mcc_scores_df['Detailed MCC']

    fig = px.pie(
        values=mcc_scores_df['Score'],
        names=mcc_scores_df['Label'],
        title=f'User Interest Score by MCC for User ID: {user_id}'
    )
    fig.update_layout(
        showlegend=True,
        width=1000,
        height=500
    )
    st.plotly_chart(fig)

def plot_top_mcc_by_segment(rfm_segment):
    user_profiles.columns = user_profiles.columns.astype(str)

    mcc_mapping['MCC'] = mcc_mapping['MCC'].astype(str)

    mcc_columns = [col for col in user_profiles.columns if col.startswith('total_amount_mcc_')]


    segment_data = user_profiles[user_profiles['RFM_Segment'] == rfm_segment]
    mcc_amounts = segment_data[mcc_columns].sum(axis=0)
    
    # Filter out zero values to keep only the MCCs with spending
    mcc_amounts = mcc_amounts[mcc_amounts > 0]
    
    # Convert the index to string to facilitate merging with MCC mapping
    mcc_amounts.index = mcc_amounts.index.str.replace('total_amount_mcc_', '').astype(str)
    
    # Convert the Series to a DataFrame
    mcc_amounts = mcc_amounts.reset_index()
    mcc_amounts.columns = ['MCC', 'Total_Amount']
    
    # Merge the MCC amounts with the MCC mapping data
    mcc_amounts = mcc_amounts.merge(mcc_mapping, on='MCC', how='left')
    
    # Create a label combining MCC and its description
    mcc_amounts['Label'] = mcc_amounts['MCC'] + ' - ' + mcc_amounts['Detailed MCC']
    
    # Create the pie chart using Plotly
    fig = px.pie(
        values=mcc_amounts['Total_Amount'], 
        names=mcc_amounts['Label'], 
        title=f'Top MCCs by Total Amount for RFM Segment: {rfm_segment}'
    )
    fig.update_layout(
        showlegend=True,
        width=1000,
        height=500
    )
    st.plotly_chart(fig)

# Streamlit app
st.title('User Profiles Analysis')

# Select user ID for analysis
user_id = st.selectbox('Select User ID', user_profiles['FK_BusinessUserId'].unique())

# Display user profile information
st.header(f'User Profile for User ID: {user_id}')
display_user_profile(user_id)

# Display pie chart for frequent MCC
st.header(f'Frequent MCCs for User ID: {user_id}')
plot_frequent_mcc(user_id)

# Display pie chart for top MCC by amount for suer
st.header(f'Top MCCs by amount for User ID: {user_id}')
plot_top_mcc_by_amount(user_id)


# Display pie chart for User Interest Score by MCC

st.header(f'User Interest Score by MCC for User ID: {user_id}')
plot_user_interest_score_by_mcc(user_id)

rfm_segment = st.selectbox('Select RFM Segment', user_profiles['RFM_Segment'].unique())

# Display pie chart for top MCC by amount for the selected RFM segment
st.header(f'Top MCCs by Amount for RFM Segment: {rfm_segment}')
plot_top_mcc_by_segment(rfm_segment)

