#!/usr/bin/env python
# coding: utf-8

# # Profitable App Profiles for the App Store & Google Play Markets
# This project has four parts:
# * I will start by by clarifying the goal of the project (business understanding)
# * Then I will collect relevant data and review it (data exploration)
# * Next I'll clean the data to prepare it for analysis (data preparation)
# * Finally I will analyze the cleaned data (data analysis)
# 
# **NOTE - this project is guided in spirit by Dataquest.io's guided project for the "Python for Data Science: Fundamentals" module. However, I am not using all of the methods they use - for example, I'm using pandas dataframes instead of lists of lists, and using code that makes sense with pandas dataframes instead of lists of lists, and so on.**
# 
# # Project objective (business understanding)
# This project looks at iOS and Android mobile apps from the perspective of an analyst for a company which builds free mobile apps and makes money from ad revenue on those mobile apps. To this end, I analyze free apps by number of users to determine which kinds of apps are likely to attract more users.
# 
# My goal in this project is to develop a profile or set of profiles for profitable apps on the App Store & Google Play markets. That way, the company's developers have data to inform what kind of apps they build.

# According to [Statista](https://www.statista.com/statistics/276623/number-of-apps-available-in-leading-app-stores/), in September 2018 there were approximately 2 million iOS apps on the App Store, and 2.1 million Android apps on Google Play:
# <img src='https://s3.amazonaws.com/dq-content/350/py1m8_statista.png'>
# 
# Since the data for all of those apps are not readily available, I will use two datasets which can function as samples of the data instead. There is [one data set](https://www.kaggle.com/lava18/google-play-store-apps/home) with approximately 10,000 Android apps from Google Play (collected Augist 2018) and [another](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps/home) with approximately 7,000 iOS apps from the App Store (collected July 2017).

# # Collecting and reviewing the data (data exploration)
# **Note: I've uploaded the data to [my GitHub](https://github.com/gmayock/profitable_app_profiles). I tried to access it directly from Kaggle's servers but was unsuccessful. Please give all credit where credit is due to the appropriate creators of the datasets, as linked above.**

# In[1]:


import pandas as pd
df_google_play = pd.read_csv('https://raw.githubusercontent.com/gmayock/profitable_app_profiles/master/googleplaystore.csv')
df_app_store = pd.read_csv('https://raw.githubusercontent.com/gmayock/profitable_app_profiles/master/AppleStore.csv', index_col=0)


# ### Identifying columns which could help with my analysis
# First I print the df shape along with a list of columns to see what they contain

# In[56]:


print("Google Play:",df_google_play.shape,"\n", list(df_google_play), "\n\nApp Store:",df_app_store.shape,"\n", list(df_app_store))


# Then I print the head of each dataframe to see what the data looks like. First Google:

# In[3]:


df_google_play.head()


# Then the App Store

# In[4]:


df_app_store.head()


# Now I'll print the unique values counts ([nunique](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.nunique.html)): 

# In[57]:


print("Google Play:\n",df_google_play.nunique(), df_google_play.shape,"\n\nApp Store:\n", df_app_store.nunique(), df_app_store.shape)


# ### Summary 
# Some features that stick out from Google Play are:
# * Category 
# * Genres
# * Rating
# * Reviews
# * Installs
# * Price
# 
# For the App Store:
# * prime_genre
# * rating_count_tot
# * user_rating
# * price

# # Cleaning the data for analysis (data preparation)
# The next step it to remove data which is not going to be relevant for this project. First I'll remove wrong information - bad lines etc - and then, since the goal is to build a free app in English, I will remove information which doesn't fit those parameters.
# 
# ## Google Play data
# ### Removing a row which was scraped wrong
# A quick look at the [discussion forum](https://www.kaggle.com/lava18/google-play-store-apps/discussion/81460) on Kaggle for the Google Play data set reveals a shift of the cells for index 10472 as well. Let's look at that first.

# In[6]:


df_google_play[10472:10473]


# It looks like the error is still in the dataset. I could try to fix the data, or simply delete it. Let's consider fixing it.
# 
# #### Why I can't fix the row with wrong values
# To fix it, I would need to shift the content of the cells over one to the left, then manually enter the value which was supposed to be in "Category", as that's the cell which is overwritten. 
# 
# From my data exploration above, it looks like the "Genres" feature maps to the "Category" feature. However, for this row, the "Genres" feature is blank. Therefore, I can't use that in this instance. Therefore I'll delete it.
# 
# #### Deleting it using loc
# I could in theory drop the row by the index. However, it's better with Jupyter Notebooks to drop it in a way that won't cause an error if the cell is ran more than once. The method I use can be ran multiple times without throwing an error, as you see below:

# In[7]:


df_google_play = df_google_play.loc[df_google_play['App'] != 'Life Made WI-Fi Touchscreen Photo Frame']
df_google_play.shape


# In[8]:


df_google_play = df_google_play.loc[df_google_play['App'] != 'Life Made WI-Fi Touchscreen Photo Frame']
df_google_play.shape


# ### Dropping duplicate rows
# #### Dropping duplicate rows using drop_duplicates()
# It looks like the discussion forum mentions some duplicate rows, as well, so we'll take care of that with [pandas.DataFrame.drop_duplicates()](https://pandas.pydata.org/pandas-docs/version/0.17/generated/pandas.DataFrame.drop_duplicates.html). However, it's important to note that these duplicates vary mostly in number of reviews. Therefore I will make sure the "Reviews" column does not factor in.

# In[9]:


col_list = list(df_google_play)
okay_to_drop_non_dupe = ['Reviews']
drop_dupe_list = [col for col in col_list if col not in okay_to_drop_non_dupe]
df_google_play_test = df_google_play.drop_duplicates(subset=drop_dupe_list)
df_google_play_test.shape


# That removes over a thousand rows, or nearly 10% of the data. But perhaps this isn't the best way to do this, for two reasons:
# 1. It drops the rows randomly
# 2. It doesn't necessarily result in a single row for each app
# 
# To point one: if the number of reviews is different, it likely indicates the information was pulled at different times, because logically the number of reviews should only be able to go up. Therefore, it's more logical to keep the row with the highest number of reviews.
# 
# To point two: there are 9660 unique values for the App, as seen above when I ran *df\_google\_play.nunique()*. This could be caused by any number of the other columns.
# 
# For example, since the rows may have been pulled on different dates, it's possible that "Last Updated" is causing duplicates which wouldn't be dropped by the above method. 
# 
# #### Dropping duplicate rows after sorting the dataframe
# Let's see what it looks like if we drop duplicates with the subset being only "App". In order to incorporate the learnings from point one, first I'll sort the dataframe by "Reviews" first.

# In[10]:


df_google_play_test = df_google_play.sort_values(by="Reviews", ascending=False)
df_google_play_test = df_google_play_test.drop_duplicates(subset="App")
df_google_play_test.shape


# 9,660 unique values, minus the one we dropped ('Life Made WI-Fi Touchscreen Photo Frame'), resulting in 9,659 unique rows. Wonderful. Now I'll assign this back to df_google_play.

# In[11]:


df_google_play = df_google_play_test.copy()
df_google_play.shape


# ### Removing non-free apps

# In[12]:


df_google_play = df_google_play.loc[df_google_play['Price'] == '0']
df_google_play.shape, df_google_play['Price'].value_counts()


# ### Removing non-English apps
# It's somewhat difficult to remove non-English apps from the Google Play data set as it does not explicitly state the language of the app. 
# 
# The suggested way to address this is to filter out apps with three or more non-standard (ordinal above 127) ASCII characters. This is used as a proxy for foreign language, although due to the pervasiveness of non-standard characters in modern app names - "Lep's World 3 🍀🍀🍀" comes to mind, or "► MultiCraft ― Free Miner! 👍" - it's not a perfect method.
# 
# Nevertheless, it is the method I will use at this time. Other methods which were considered but ultimately refused:
# 1. I could use the ASCII characters between 128 and 255 to denote non-English (thus omitting the emojis, etc)
# 2. I could find ranges for the ordinal of the emojis and use ASCII characters between 128 and the ordinal start of the emojis to denote non-English (thus omitting the emojis but including characters like kanji, hiragana, etc).
# 3. I could look for supplementary data sources

# In[13]:


import string

def nonEnglishCharacterCount(app_name):
    non_eng_char_ct = 0
    for character in app_name:
        if ord(character) > 127:
            non_eng_char_ct += 1
    return non_eng_char_ct


# In[14]:


df_google_play['num_non_eng_chars'] = [nonEnglishCharacterCount(i) for i in df_google_play['App']]


# In[15]:


df_google_play = df_google_play.loc[df_google_play['num_non_eng_chars'] <= 3]
df_google_play.shape


# ## App Store data
# ### Remove non-English apps

# In[16]:


df_app_store.shape


# In[17]:


df_app_store['num_non_eng_chars'] = [nonEnglishCharacterCount(i) for i in df_app_store['track_name']]
df_app_store = df_app_store.loc[df_app_store['num_non_eng_chars'] <= 3]
df_app_store.shape


# In[18]:


df_app_store.dtypes


# ### Removing non-Free apps

# In[19]:


df_app_store = df_app_store.loc[df_app_store['price'] == 0]
df_app_store.shape, df_app_store['price'].value_counts()


# ### Removing duplicates
# There are 7195 unique track_names, and 7197 rows. Let's see what the difference for those two are.

# In[20]:


dupe_list = ['Mannequin Challenge', 'VR Roller Coaster']
df_check1 = df_app_store.loc[df_app_store['track_name'] == dupe_list[0]]
df_check2 = df_app_store.loc[df_app_store['track_name'] == dupe_list[1]]
df_check1
# df_check2


# In[21]:


df_check2


# At a  glance, it looks like an old version is left in the list. There were no discussions about this at the source of the data, so I [started one](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps/discussion/90409). We'll see if anything comes of it.
# 
# In the meantime, let's drop_duplicates as before.

# In[22]:


df_app_store = df_app_store.sort_values(by='ver', ascending=False)
df_app_store = df_app_store.drop_duplicates(subset='track_name')
df_app_store.shape


# Alright, it worked.
# 
# # Understanding what data we need (business understanding)
# We have the data filtered to relevant information (free, English-language, non-duplicate apps), but what are we really looking for? 
# 
# The business objective is to develop a _successful_ app. Therefore, we should explore the data to determine which apps are successful on both Google Play and the App Store. 
# 
# The standard operating procedure for companies building this sort of app has three steps:
# 1. Build a minimal version of the app on Google Play
# 2. If the app has a good response, develop it further
# 3. If the app is profitable after a short time, port it to the App Store
# 
# The order of operating systems to develop can be switched depending on company competency, etc.
# 
# Nevertheless, it's important that we build a profile of apps which are successful on both Google Play _and_ the App Store, or we'll be leaving behind a large portion of the market ([about 45%](https://www.statista.com/statistics/266572/market-share-held-by-smartphone-platforms-in-the-united-states/)) which is accessible for presumably much less work than building a new app from scratch.

# # Building the profile of a successful app (data analysis)
# ## Genre
# First things first, let's explore which genres are most common in each market.

# In[23]:


# print(df_google_play['Genres'].value_counts().to_dict())
df_google_play['Genres'].value_counts(normalize=True)


# In[24]:


# print(df_google_play['Category'].value_counts().to_dict())
df_google_play['Category'].value_counts(normalize=True)


# In[25]:


# print(df_app_store['prime_genre'].value_counts().to_dict())
df_app_store['prime_genre'].value_counts(normalize=True)


# Games are by far the largest over the two, but what's this Family category on Google Play?

# In[60]:


df_family = df_google_play.loc[df_google_play['Category'] == 'FAMILY']
df_family['Genres'].value_counts(normalize=True)


# In[61]:


df_family_entertainment = df_family.loc[df_family['Genres'] == 'Entertainment']
df_family_entertainment = pd.DataFrame(df_family_entertainment['App'].value_counts()).reset_index().drop(columns='App')
list_family_entertainment = df_family_entertainment['index'].tolist()
print(list_family_entertainment)


# At least a quarter of the Family category look related to games (simulation, casual, puzzle, role playing, strategy, brain games, etc), but most are streaming devices or random entertainment. 
# 
# The two app stores look different. The App Store is largely dominated by games with more than half of the apps in our target market. Google Play has productivity-type apps - Tools, Business, Lifestyle, Productivity, etc - but still a large portion are games of one type or another.
# 
# My initial recommendation is moving forward with making a game. They are the most ubiquitous on the App Store, and close to the most on Google Play - certainly the most across both. However, I'm going to check two things first - the Genres to Category relationship, and the popularity of genres in each market.
# 
# ## 'Genres' to 'Category' relationship on Google Play
# At a glance, Genres looks more detailed than Category. We can check that each Genre is assigned to only one Category with a few commands.

# In[28]:


df_genre_category_relationship = df_google_play.loc[:,['Genres','Category']]
df_genre_category_relationship = df_genre_category_relationship.drop_duplicates()
len(df_genre_category_relationship)


# In[29]:


df_g_c_r_counts = pd.DataFrame(df_genre_category_relationship['Genres'].value_counts()).reset_index()
df_g_c_r_counts = df_g_c_r_counts.rename(columns={'index':'Genres','Genres':'Counts'})
df_g_c_r_counts = df_g_c_r_counts.loc[df_g_c_r_counts['Counts'] >= 2]
len(df_g_c_r_counts)


# It does look like the majority of Genres are only set to one Category, but 19 of 134 (14%) are assigned to two Categories. Let's take a look at those.

# In[30]:


check_list = list(df_g_c_r_counts['Genres'])
check_df = df_google_play.loc[df_google_play['Genres'].isin(check_list)]
check_df = check_df.loc[:,['Genres','Category']].drop_duplicates().sort_values('Genres')
check_df


# Lots of crossover between family and education, family and entertainment, and family and game, but not anything drastic enough to make me want to reconsider my recommendation to build a game.  
# 
# ## Popularity of genres
# We want our app to have a lot of installs. Therefore it's important to see the popularity of each. We can use the 'Installs' column directly for Google Play, and for the App Store we can use the rating_count_tot as a proxy. First we have to convert the text string install counts to int strings. The granularity isn't great but it's better than nothing.

# In[31]:


df_google_play['Installs_count'] = [i.replace(',','').replace('+','') for i in df_google_play['Installs']]
df_google_play['Installs_count'] = df_google_play['Installs_count'].astype(int)
# df_google_play['Installs_count'].value_counts()


# Now we can look at the average install count for the Genres and Categories.

# In[50]:


# Average installs (in millions) by Genre on Google Play
df_gp_avg_installs_g = df_google_play.groupby('Genres', as_index=False)['Installs_count'].mean().sort_values('Installs_count', 
                                                                                                          ascending=False)
df_gp_avg_installs_g['Installs_count'] = [i/1000000 for i in df_gp_avg_installs_g['Installs_count']]
df_gp_avg_installs_g = df_gp_avg_installs_g.rename(columns={'Installs_count':'Average_installs_count_in_millions'})
df_gp_avg_installs_g


# In[51]:


# Average installs (in millions) by Category on Google Play
df_gp_avg_installs_c = df_google_play.groupby('Category', as_index=False)['Installs_count'].mean().sort_values('Installs_count', 
                                                                                                          ascending=False)
df_gp_avg_installs_c['Installs_count'] = [i/1000000 for i in df_gp_avg_installs_c['Installs_count']]
df_gp_avg_installs_c = df_gp_avg_installs_c.rename(columns={'Installs_count':'Average_installs_count_in_millions'})
df_gp_avg_installs_c


# In[54]:


# Average number of ratings by prime genre on the App Store
df_a_avg_rat = df_app_store.groupby('prime_genre', as_index=False)['rating_count_tot'].mean().sort_values('rating_count_tot', ascending=False)
df_a_avg_rat['rating_count_tot'] = df_a_avg_rat['rating_count_tot'].astype(int)
df_a_avg_rat


# # Final Recommendation
# Based on the information above, I recommend building a game of the genre "action and adventure" or related. 
