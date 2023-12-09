# Finding Patterns in the Stream: A Machine Learning Analysis of Netflix Movie Data 
### 90803 Machine Learning Foundations with Python 
#### Spring 2023, Heinz College, Carnegie Mellon University
Authors: 

- Sajujya Gangopadhyay - sajujyag@andrew.cmu.edu
- Grace Eunji Kim - eunjik@andrew.cmu.edu
- Yi-Hsueh Yang - yihsuehy@andrew.cmu.edu


## Repository Structure
- Team8_DataCleaning.ipynb
- Team8_Prediction.ipynb
- Team8_ClusteringAnalysis.ipynb
- Team8_TopicModeling.ipynb
- Team8_AssociationRules.ipynb
- Team8_PresentationSlides.pdf


## Introduction

Like any other customer-centric application today, Netflix is fueled by a host of data science and analytics techniques, making the user experience smooth and glitch-free on the platform. Over the last 2 decades, the emergence of data science in e-commerce has accelerated growth in terms of both revenue generation and customer satisfaction. In this project, we are in the shoes of a group of consultants for Netflix, helping them discover their customers' reactions to their products and the patterns their customers have so as to keep the customers spending more time on their platform.

**Key Words**: Association Rule, Clustering, Topic Modeling, Rating Prediction, Movie Data

## Data Collection

The datasets for this project come from Netflix (Netflix Prize), the IMDb website, Github, and Wikipedia.

#### Datasets used in this project:

1. [Netflix Original Movie Rating Dataset](https://archive.org/download/nf_prize_dataset.tar)
	
	Source: Netflix Prize Dataset
	
	Netflix provides this dataset, and we will use two files: 'user\_ratings.txt' (100M, 4) and 'movie\_titles.txt' (17K, 4). The first file includes CustomerID, rating, and the date they rated the movie. The second file contains MovieID, Title, and YearOfRelease columns.

2. [IMDb Dataset of information on Netflix Movie](https://www.imdb.com/list/ls093264464/?sort=list_order,asc&st_dt=&mode=simple&page=1&ref_=ttls_vw_smp)
	
	Source: IMDb Website
	
	This dataset (3741, 14) from IMDb includes title, title type, IMDb Rating, Runtime (Minute), Year, Genres, Number of Votes, and Directors. Features like index, const, and URL are useless in our analysis, so they will be dropped.

3. [Netflix Movie Title Dataset](https://raw.githubusercontent.com/srikarthadaka/projects/main/netflix_recommendation_system/netflix_titles.csv)

	Source: Github/srikarthadaka
	
	This dataset (8807,12) includes more details about Netflix movies, including columns like type, title, director, cast, country, date\_added, release\_year, Rating, duration, listed\_in, and description. There is some overlap between the second and third datasets, which should be dropped, but we are using both for supplementary and feature multiplicity.
	
4. [Wikipedia Movie Description](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

	Source: Wikipedia

	This dataset has plot summary descriptions of movies scraped from Wikipedia. The dataset contains descriptions of 34,886 movies from around the world. Columns included are 'Release Year', 'Title', 'Origin/Ethnicity', 'Director', 'Plot', 'Genre', 'Wiki Page', and 'Plot'. We include this dataset for a longer description and more information about each movie for our topic modeling models. Therefore, we can compare the difference in insightfulness between the description on Netflix and Wikipedia.

## Procedure Graph

![alt text](https://drive.google.com/uc?id=139c3VbrwPttT04F9k4yby9-YialP9gKv)


## Analytical Questions

This project aims to analyze some common questions regarding Netflix and its users. 

We have 4 notebooks that do machine learning for this project. The first does clustering, the second makes IMDb rating prediction, the third does topic modeling, and the last one finds the association rule of the commonly-watched movie sets. 

1. As a customer, getting to know the rating or word of mouth from others is always helpful before we click on the first episode of any movie. However, it is considered biased or problematic to believe the ratings for movies or series on Netflix since many are Netflix Originals. Thus, a third-party impartial unit can play a crucial part, e.g., IMDb and Rotten Tomatoes, here for our reference. We find performing well on both platforms exciting and vital to be viewed as sound production. We are using an IMDb rating dataset combined with the customer rating dataset we got by merging previous movies and series only on Netflix to make predictions of their rating on IMDb. <u>**What is the IMDb rating for a Netflix movie, given its information and rating on Netflix?**</u> 

	The IMDb rating is our target variable for the prediction. The rest of the dataframe, including user Rating, number of votes, and genres, are all independent variables that will be fed into the model.
	- Models used for this question:
		-	Random Forest Regressor 
		-  Decision Tree Regressor
		-  Gradient Boosting Regressor
		-  XGBoost

2. As frequent users of Netflix, we often do some research before watching a new series or movie. We either consult our friends or look at the movie's description and trailer. The trailer is always a good source of information for us before making a decision. However, the description could supplement introducing the movie and 'selling' it to the customers. Hence, we came up with an idea to see whether the description of each Netflix movie is sufficient for us to acknowledge what we are actually consuming. The question will be: <u>**Is the Netflix movie description sufficient for us to know the topic of the movie?**</u> However, Netflix plot descriptions were too short in providing enough information for the models. Therefore, we decided to incorporate Wikipedia's description to see if it played a better role in the analysis.

	- Models used for this question:
		-  Latent Dirichlet Allocation (LDA)
		-  Latent Semantic Analysis (LSA)
		-  Non-Negative Matrix Factorization (NMF)
		-  Kmeans
		
		*TF-IDF preprocessing is used in all models*

		
3. For this question, we pretend to be the management at Netflix. Our question is: <u>**What are the features of the movies which receive the highest and lowest ratings from our customers?**</u> Answering this question will not only help us understand the movies which are current favorites for our audiences but will also serve as a guide to creating newer Netflix Originals in the future. 
	- We use an unsupervised machine-learning model to cluster the movies. After this, we look at the features of the movies in the different clusters to try and gain more information about these clusters. Initially, we aim to look at the box plots of the Customer Ratings in each of these clusters first to understand which cluster performs the best.
	- Once this is done, we will look at the other features of our best and worst-performing clusters to maximize our gains and minimize our losses in the future. 
	- Models used for this question:
		-  Kmeans
		-  Agglomerative Clustering
		-  Gaussian Mixture
		-  Spectral Clustering
		

4. For the business side of Netflix, the primary goal is to make our customers stick to our platform as long as possible so they will be less likely to churn. One way to make customers spend more and more time on Netflix is to keep giving them content that interests them. Therefore, we think of the pattern of how people watch shows these days; we take into account the advice of our friends. We tend to think the more people watch, the better this show is going to be. Hence we come up with the desire to know about: <u>**What movies do users frequently watch together?**</u>
	- Models used for this question:
		-  Apriori Algorithm
		-  FP Growth
			-  *PCA is used for plotting*


## **Running the Project / Getting Started**
#### Option 1: Run from start to end:
If you would like to run the whole process from data merging to modeling:
Make sure to download all three csv files from this [link](https://drive.google.com/drive/folders/10kHPT59Sy6S_yYXdIwCjmiRnZs1Wkkbr?usp=sharing) before running Team8_DataCleaning.ipynb:

1. _netflix\_user\_ratings.csv_
2. _movie\_titles.csv_
3. _imdb\_netflix\_movies.csv_
4. _netflix\_titles.csv_
	
When running 'Team8_DataCleaning.ipynb', the notebook runs preliminary data cleaning for merging purposes, and the last cell is going to generate the final cleaned dataset called _final\_netflix\_dataset.csv_. (this takes about 1 minute) 

#### Option 2: Skip Data Merging and jump straight into data cleaning and modeling:
This step assumes that you have the dataset _final\_netflix\_dataset.csv_ ready.
If you haven't, download the cleaned dataset from: [Merged dataset download link](https://drive.google.com/file/d/1ZGN9726dUz-x_AI5dD1XUfSRObs5--PN/view?usp=sharing)  

Afterward, you are ready to start the exploration with us! 

We have four notebooks for modeling and analysis: 'Team8\_Prediction.ipynb' for prediction, 'Team8\_ClusteringAnalysis.ipynb' for clustering analysis, 'Team8\_TopicModeling.ipynb' for topic modeling, and 'Team8\_AssociationRules.ipynb' for association rule. 

It doesn't matter which notebook you run afterward; they do different tasks separately. The only thing that needs to keep in mind is that before running the 'Team8\_TopicModeling.ipynb' notebook, please make sure to download the _wiki\_movie\_plots\_deduped.csv_ file from the [link](https://drive.google.com/drive/folders/10kHPT59Sy6S_yYXdIwCjmiRnZs1Wkkbr?usp=sharing).
	