import streamlit as st
import pandas as pd
import time
import plotly.express as px 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import numpy as np

def load_data():
    df = pd.read_csv('athlete_events.csv', encoding='unicode_escape')
    region_df = pd.read_csv('noc_regions.csv', encoding='unicode_escape')
    df = df[df['Season'] == 'Summer']
    merged_df = df.merge(region_df, on='NOC', how='left')
    merged_df.drop_duplicates(inplace=True)
    df = pd.concat([merged_df, pd.get_dummies(merged_df['Medal'])], axis=1)
    return df
def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, "Overall")

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, "Overall")
    return years, country
def fetch_medal_tally(df, years, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
 
    if years == "Overall" and country == "Overall":
        temp_df = medal_df
    if years == "Overall" and country != "Overall":
        flag=1
        temp_df = medal_df[medal_df['region'] == country]
    if years != "Overall" and country == "Overall":
        temp_df = medal_df[medal_df['Year'] == int(years)]
    if years != "Overall" and country != "Overall":
        temp_df = medal_df[(medal_df['Year'] == int(years)) & (medal_df['region'] == country)]
    

    if flag ==1:
        x = temp_df.groupby('Year').sum()[["Gold", "Silver", "Bronze"]].sort_values('Year', ascending=True).reset_index()
    else:
        x = temp_df.groupby('region').sum()[["Gold", "Silver", "Bronze"]].sort_values('Gold', ascending=False).reset_index()
    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')
    return x 
def data_over_time(df,col ):
    nations_over_time = df.drop_duplicates(['Year',col])['Year'].value_counts().reset_index().sort_values('index')
    nations_over_time.rename(columns = {'index': 'Editions' , 'Year': col}, inplace = True)
    return nations_over_time

def most_successful(df, sport):
    temp_df = df.dropna(subset=['Medal'])
    
    if sport !='Overall':
        temp_df= temp_df[temp_df["Sport"]==sport]
    x = temp_df['Name'].value_counts().reset_index().head(10).merge(df, left_on='index',right_on='Name', how='left')[['index','Name_x','Sport','region']].drop_duplicates('index')
    x.rename(columns ={'index':'Name', 'Name_x':'Medal', 'region' : 'Countries'}, inplace = True)
    return x 

def yearwise_medal_tally(df,country):
    temp_df=df.dropna(subset = ['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'], inplace = True)
    new_df=temp_df[temp_df['region']==country]
    f_df=new_df.groupby('Year').count()['Medal'].reset_index()

    return f_df

def country_heatmap(df,country):
    temp_df=df.dropna(subset = ['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'], inplace = True)
    new_df=temp_df[temp_df['region']==country]
    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count', fill_value=0).astype('int')

    return pt 

def most_successfully_player_country (df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df= temp_df[temp_df["region"]==country]
    x = temp_df['Name'].value_counts().reset_index().head(10).merge(df, left_on='index',right_on='Name', how='left')[['index','Name_x','Sport','region']].drop_duplicates('index')
    x.rename(columns ={'index':'Name', 'Name_x':'Medal', 'region' : 'Countries'}, inplace = True)
    return x 

def weightand_height(df, sport):
    athletes_df=df.drop_duplicates(subset=['Name','region'])
    athletes_df['Medal'].fillna('NO Medal', inplace = True)
    if sport !='Overall':
        temp=athletes_df[athletes_df['Sport']==sport]
        return temp
    else:
        return athletes_df
    
def men_vs_women(df):
    athletes_df=df.drop_duplicates(subset=['Name','region'])
    men = athletes_df[athletes_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athletes_df[athletes_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()
    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)
    final.fillna(0,inplace= True)

    return final

def athletes_df(df):
    athletes_df=df.drop_duplicates(subset=['Name','region'])
    return athletes_df


def main():
    st.sidebar.title('**Olympics Analysis**')
    st.sidebar.image("sochi-2014-g8880ee89a_640-e1645698211307.jpg", use_column_width=True)
    with st.spinner("Loading..."):
        time.sleep(5)
    st.success('WELCOME')
    df = load_data()


    user_menu = st.sidebar.radio(
        'Select an Option 👇 ',(':rainbow[Medal Tally]', ':rainbow[Overall Analysis]', ':rainbow[Country-Wise Analysis]', ':rainbow[Athlete Wise Analysis]')
    )

    background_image_url = "background.jpg"

    st.markdown(
    f"""
    <style>
        body {{
            background-image: url("{background_image_url}");
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
    if user_menu == ":rainbow[Medal Tally]":
        st.sidebar.header("Medal Tally")
        years,country =country_year_list(df)

        selected_year = st.sidebar.selectbox("Select Year",years)
        selected_country = st.sidebar.selectbox("Select Country", country)
        medal_tally =fetch_medal_tally(df,selected_year,selected_country)
 

        if selected_year == 'Overall' and selected_country == 'Overall':
            st.title("Overall Tally")
        if selected_year != 'Overall' and selected_country == 'Overall':
            st.title("Medal Tally in " + str(selected_year) + " Olympics")
        if selected_year == 'Overall' and selected_country != 'Overall':
            st.title(selected_country + " overall performance")
        if selected_year != 'Overall' and selected_country != 'Overall':
            st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
        st.table(medal_tally)


    if user_menu ==":rainbow[Overall Analysis]":
        editions = df['Year'].unique().shape[0] - 1
        cities = df['City'].unique().shape[0]
        sports = df['Sport'].unique().shape[0]
        events = df['Event'].unique().shape[0]
        athletes = df['Name'].unique().shape[0]
        nations = df['region'].unique().shape[0]

        st.title("Overall Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
        
            st.header("Editions")
            st.header(editions)
            
        with col2:
            st.header("Cities")
            st.header(cities)

        with col3:
            st.header("Sports")
            st.header(sports)

        col4, col5, col6 = st.columns(3)
        with col4:
            st.header("Events")
            st.header(events)

        with col5:
            st.header("Athletes")
            st.header(athletes)

        with col6:
            st.header("Nations")
            st.header(nations)
            
            
        nations_over_time=data_over_time(df, 'region') 
        fig = px.line(nations_over_time , x='Editions', y = 'region')
        st.header("Participating Nations over the years")
        st.plotly_chart(fig)

        Events_over_time= data_over_time(df,'Event') 
        fig2  = px.line(Events_over_time, x = 'Editions', y ='Event')
        st.header("Event over the years")
        st.plotly_chart(fig2)

        Athletes_over_time=data_over_time(df,'Name') 
        fig3  = px.line(Athletes_over_time, x = 'Editions', y ='Name')
        st.header("Athletes over the years")
        st.plotly_chart(fig3)

        sports_over_time=data_over_time(df,'Sport') 
        fig4  = px.line(sports_over_time, x = 'Editions', y ='Sport')
        st.header("Sports over the years")
        st.plotly_chart(fig4)

    
        st.header("Most successful Athletes")
        sport_list = df['Sport'].unique().tolist()
        sport_list.sort()
        sport_list.insert(0,'Overall')

        selected_sport = st.selectbox('Select a Sport 👇',sport_list)
        x = most_successful(df,selected_sport)
        st.table(x)
    
    if user_menu ==":rainbow[Country-Wise Analysis]":

        country_list = df ['region'].dropna().unique().tolist()
        country_list.sort()
        selected_country = st.sidebar.selectbox('Select a Country 👇',country_list)
        country_df =yearwise_medal_tally(df,selected_country)
        fig =  px.line(country_df, x = 'Year', y = 'Medal')
        st.header(selected_country  +  " Medal Tally over the year ")
        st.plotly_chart(fig)
    

        st.header(selected_country  +  " excel in the following sports")
        pt=country_heatmap(df , selected_country)
        fig , ax = plt.subplots(figsize=(20,20))
        ax=sns.heatmap(pt , annot = True)
        st.pyplot(fig)

        player =most_successfully_player_country(df, selected_country)
        st.header( " Top 10 player of  " + selected_country )
        st.table(player)
    

    if user_menu == ":rainbow[Athlete Wise Analysis]":
        ath=df.drop_duplicates(subset=['Name','region'])
        
        st.header('Age Distribution')
        fig = ff.create_distplot([ath['Age'].dropna()],['Age Distribution'])
        st.plotly_chart(fig, use_container_width=True, width=1000, height=800)
   

        x1 = ath['Age'].dropna()
        x2 = ath[ath['Medal']=='Gold']['Age'].dropna()
        x3 = ath[ath['Medal']=='Silver']['Age'].dropna()
        x4 = ath[ath['Medal']=='Bronze']['Age'].dropna()

        st.header("Age Distribution Across Medal Types")
        fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medal', 'Silver Medal', 'Bronze Medal'], show_hist=False, show_rug=False)
        st.plotly_chart(fig, use_container_width=True, width=1000, height=800)



        famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']


        athletes_df=df.drop_duplicates(subset=['Name','region'])
        sports_list=df['Sport'].drop_duplicates().tolist()
        x=[]
        name= []
        for sport in famous_sports :
            temp_df =athletes_df[athletes_df['Sport']==sport]
            x.append(temp_df[temp_df['Medal']=='Gold']['Age'].dropna())
            name.append(sport)
        st.header("Distribution of Age wrt Sports(Gold Medalist)")
        fig = ff.create_distplot(x ,name ,show_hist=False,show_rug=False)
        st.plotly_chart(fig, use_container_width=True, width=1000, height=800)


        st.header('Height VS Weight')
        sport_list = df['Sport'].unique().tolist()
        sport_list.sort()
        sport_list.insert(0,'Overall')
        selected_sport= st.selectbox('select a sport 👇', sport_list)
        temp =weightand_height(df, selected_sport)
        fig , ax = plt.subplots(figsize=(9,5))
        ax =sns.scatterplot(x= temp['Weight'],y= temp['Height'],hue=temp['Medal'], style=temp['Sex'])
        st.pyplot(fig)

        st.header('Men Vs Women Participation Over the Years')
        final =men_vs_women(df)
        fig = px.line(final, x='Year', y=['Male', 'Female'], title='Number of Male and Female Athletes Over the Years')
        fig.update_traces(name='Male', showlegend=True)
        fig.update_layout(xaxis_title='Year', yaxis_title='Count')
        st.plotly_chart(fig)

    
if __name__ == "__main__":
    main()