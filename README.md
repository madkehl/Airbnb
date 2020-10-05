# Airbnb

The following code uses Kaggle's Boston Airbnb dataset to examine what features are most related to high Airbnb review scores and answer the following questions:  

* Using random forest regression feature ranking, what are the top predictors of overall review scores?
* When we use a multilevel model, which of these features are statistically significant?
* Out of the statistically significant features, which have large effect sizes? What are the directions?


It first uses a random forest regressor to rank features on aggregated data. Based on these rankings it selects features with > mean importance, and runs multilevel models on them to look at significance as well as directionality and effect size.   

# Installation
      
Clone repository: git clone https://github.com/madkehl/Airbnb

# Operating instructions

If the repo is cloned, the Jupyter notebook should run smoothly from start to finish, and contains all necessary code.

# Files included:

* **listings.csv and reviews.csv**:  These are taken directly from the above link
* **Udacity-1.ipynb**  This notebook standalone contains all the code necessary to run the project.  
* **txt_df**: is a file that contains adjective/adverb count and word count for the text columns specified in Udacity-1.  It is contained as a separate file because the code takes a long time to run, however the syntax exists in Udacity-1 to recreate it (cell 12).
* **Airbnb_***_features.png**: These are just pngs of the graphs contained in Udacity-1

# Current Requirements:
pandas, numpy, seaborn, re, statistics, sklearn, nltk, plotly, statsmodels

# Results:

The most important predictors of Airbnb scores tended to be amenities (WiFi, AC, Laptop-Friendly Workspace, Hair Dryer).  Location in Boston might be important (no individual neighborhoods were extremely significant, however based on latitude and longitude, perhaps south west neighborhoods receive higher reviews.  Hard to say with current info).  Aside from this superhosts tended to receive higher reviews.    


# Contact: 

Madeline Kehl (mad.kehl@gmail.com)

# Acknowledgements:

* Kaggle 
* Udacity Data Science Nanodegree



# MIT License

Copyright (c) 2020 Madeline Kehl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
