import pandas as pd
import streamlit as st
import requests
import os
import helper
import time
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_icon="🌍")#, initial_sidebar_state="collapsed")

def main():
    st.title("Economic Tracker Dashboard")
    
    st.markdown("""
    This dashboard allows you to explore economic data from various sources, visualize it on maps, and analyze historical trends through time series charts. You can select different features, counties, and layers to gain insights into the economic conditions of various regions.

    ### Data Sources
    The data used in this dashboard is sourced from Opportunity Insights and includes information on economic indicators, employment, education, and more. Location data for counties is retrieved from a public repository.

    ### How to Use
    - Use the navigation menu on the left to select different data sources and explore the maps and charts.
    - Each page represents a heatmap where you can choose from various features to overlay on the map.
    - You can also select a county and a specific feature to view the historical data as a time series chart.
                

    ### Explore and Analyze
    Feel free to interact with the different pages and layers to analyze economic trends and patterns. If you have any questions or need assistance, please don't hesitate to reach out.

    ### How to Update
    Click the 'Download Latest Data' button below to retrieve the most current data, ensuring that you have the latest information available for your analysis.
                        
    """)

data_links = {
    "Affinity - County - Daily": "https://github.com/OpportunityInsights/EconomicTracker/raw/main/data/Affinity%20-%20County%20-%20Daily.csv",
    "Employment - County - Weekly": "https://github.com/OpportunityInsights/EconomicTracker/raw/main/data/Employment%20-%20County%20-%20Weekly.csv",
    "Job Postings - County - Weekly": "https://github.com/OpportunityInsights/EconomicTracker/raw/main/data/Job%20Postings%20-%20County%20-%20Weekly.csv",
    "UI Claims - County - Weekly": "https://github.com/OpportunityInsights/EconomicTracker/raw/main/data/UI%20Claims%20-%20County%20-%20Weekly.csv",
    "Zearn - County - Weekly": "https://github.com/OpportunityInsights/EconomicTracker/raw/main/data/Zearn%20-%20County%20-%20Weekly.csv",
}

def clean_data_folder(data_folder="Datas/Raw Datas"):
    # Remove all files in the "Raw Datas" folder 
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        while os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                # Error occurred, wait for a short time and then retry
                time.sleep(1)  # Wait for 1 second
                st.error(f"Error while cleaning the data folder: {e}")
            else:
                # Deletion successful, exit the loop
                break


def download_all_data():
    
    # Clean the "Raw Datas" folder before downloading new data
    clean_data_folder()
    clean_data_folder(data_folder= "Datas/Maps")
    clean_data_folder(data_folder= "Datas/Chart Datas")

    # Create a progress bar
    progress_bar = st.progress(0)

    # Total number of data files to download
    total_files = len(data_links)

    # Counter to keep track of downloaded files
    completed_files = 0

    # Iterate through the data links and download the files
    for filename, link in data_links.items():
        destination_path = os.path.join("Datas/Raw Datas", filename + ".csv")
        response = requests.get(link, stream=True)

        if response.status_code == 200:
            with open(destination_path, "wb") as f:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
            completed_files += 1
            progress_bar.progress(completed_files / total_files)

    st.success("All data files downloaded successfully!")

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
     # Create the "Raw Datas" folder if it doesn't exist

    
    create_path("Datas/Raw Datas")
    create_path("Datas/Maps")
    create_path("Datas/Chart Datas")
    
    print(st.__version__)
    
    main()

    # Create a Streamlit button to clean and download data files
    if st.button("Download Latest Data"):
        download_all_data()
        zearn_df, unemep_df, job_df, emp_df, spend_df, loc = helper.read_data()
        # CLean Old Maps Under  Datas\Maps using os 
        
        helper.generate_and_save_heatmaps(zearn_df,unemep_df, job_df, emp_df, spend_df, loc)
        
        helper.prepare_and_save_chart_data(zearn_df,unemep_df, job_df, emp_df, spend_df, loc)
