Exploring Tidymodels
================
registea
13/07/2020

# 

# 

# 

![](C:/Users/Anthony/Documents/Git/Project%20Portfolio/tidymodels_exploration/Churn_Graphics.png)

# 

# 

# 

<style type="text/css"> 

body{ /* Normal  */ 
      font-size: 16px; 
  } 
td {  /* Table  */ 
  font-size: 12px; 
} 
h1.title { 
  font-size: 38px; 
  color: Red; 
} 
h1 { /* Header 1 */ 
  font-size: 30px; 
  color: Red; 
} 
h2 { /* Header 2 */ 
    font-size: 26px; 
  color: Red; 
} 
h3 { /* Header 3 */ 
  font-size: 22px; 
  font-family: "Aerial", Times, serif; 
  color: Red; 
} 
code.r{ /* Code block */ 
    font-size: 12px; 
} 
pre { /* Code block - determines code spacing between lines */ 
    font-size: 14px; 
} 
</style>

# Introduction

The primary objective of this notebook is to explore the tidymodels
predictive model framework. I am familiar with the caret package but as
Max Kuhn has replaced caret with tidymodels and it has been available
for a couple of years, I thought it a good time to take it for a test
ride. To enable me to explore the framework a churn
[dataset](https://www.kaggle.com/shrutimechlearn/churn-modelling)
provided by **Shurti\_lyyer** on Kaggle will be used. The objective of
this analysis is to complete a binary classification to identify whether
a customer will leave the business.

From reading online, within the tidymodels framework the key packages
are:

  - rsample - Different types of re-samples
  - recipes - Transformations for model data pre-processing
  - parnip - A common interface for model creation
  - tune - Framework for hyperparameter tuning
  - dials - Specific hyperparameter tuning functions
  - yardstick - Measure model performance

<!-- end list -->

``` r
# Modelling Framework
library(tidymodels) # Predictive Framework
library(caret) # Predictive Framework

# Modelling AlgorithmsS
library(glmnet) # Glmnet regression
library(rpart) # Decision Trees
library(ranger) # Random Forests

# Formating, Visualisations and tables
library(scales) # Number formats
library(knitr) # Table
library(gridExtra) # multiplot
library(e1071) # Summary distribution
library(skimr) # Summarise dataframe
library(naniar) # Missing data summary
library(corrplot) # Correlation plot
library(probably) # Probability thresholds

# Data handling Packages
library(tidyverse) # Data handling/ Graphics
library(data.table) # Data handling
```

# Data loading

To kick start this project, the fread function from the data.table
package is used to load the data into memory.

``` r
# Load data
df_churn <- 
  data.table::fread("C:/Users/Anthony/Documents/Git/Project Portfolio/tidymodels_exploration/Churn_Modelling.csv")
```

# Exploratory Data Analysis

The skim function from the **skimr** package produces the summary output
below, showing that we have 10,000 observations and 14 variables. There
are 3 character variables and 11 numeric.

The character variables relate to gender, geography and customer’s
surname. The numerical variables have two references of Id (rows and
customer). Beyond this, data relates to age, tenure, credit, income and
our final variable of whether the customer churned (labelled as Exited).

``` r
# Summarise datafrmae
skim(df_churn)
```

|                                                  |           |
| :----------------------------------------------- | :-------- |
| Name                                             | df\_churn |
| Number of rows                                   | 10000     |
| Number of columns                                | 14        |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |           |
| Column type frequency:                           |           |
| character                                        | 3         |
| numeric                                          | 11        |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |           |
| Group variables                                  | None      |

Data summary

**Variable type: character**

| skim\_variable | n\_missing | complete\_rate | min | max | empty | n\_unique | whitespace |
| :------------- | ---------: | -------------: | --: | --: | ----: | --------: | ---------: |
| Surname        |          0 |              1 |   2 |  23 |     0 |      2932 |          0 |
| Geography      |          0 |              1 |   5 |   7 |     0 |         3 |          0 |
| Gender         |          0 |              1 |   4 |   6 |     0 |         2 |          0 |

**Variable type: numeric**

| skim\_variable  | n\_missing | complete\_rate |        mean |       sd |          p0 |         p25 |         p50 |         p75 |       p100 | hist  |
| :-------------- | ---------: | -------------: | ----------: | -------: | ----------: | ----------: | ----------: | ----------: | ---------: | :---- |
| RowNumber       |          0 |              1 |     5000.50 |  2886.90 |        1.00 |     2500.75 |     5000.50 |     7500.25 |    10000.0 | ▇▇▇▇▇ |
| CustomerId      |          0 |              1 | 15690940.57 | 71936.19 | 15565701.00 | 15628528.25 | 15690738.00 | 15753233.75 | 15815690.0 | ▇▇▇▇▇ |
| CreditScore     |          0 |              1 |      650.53 |    96.65 |      350.00 |      584.00 |      652.00 |      718.00 |      850.0 | ▁▃▇▇▃ |
| Age             |          0 |              1 |       38.92 |    10.49 |       18.00 |       32.00 |       37.00 |       44.00 |       92.0 | ▅▇▂▁▁ |
| Tenure          |          0 |              1 |        5.01 |     2.89 |        0.00 |        3.00 |        5.00 |        7.00 |       10.0 | ▇▆▆▆▅ |
| Balance         |          0 |              1 |    76485.89 | 62397.41 |        0.00 |        0.00 |    97198.54 |   127644.24 |   250898.1 | ▇▃▇▂▁ |
| NumOfProducts   |          0 |              1 |        1.53 |     0.58 |        1.00 |        1.00 |        1.00 |        2.00 |        4.0 | ▇▇▁▁▁ |
| HasCrCard       |          0 |              1 |        0.71 |     0.46 |        0.00 |        0.00 |        1.00 |        1.00 |        1.0 | ▃▁▁▁▇ |
| IsActiveMember  |          0 |              1 |        0.52 |     0.50 |        0.00 |        0.00 |        1.00 |        1.00 |        1.0 | ▇▁▁▁▇ |
| EstimatedSalary |          0 |              1 |   100090.24 | 57510.49 |       11.58 |    51002.11 |   100193.91 |   149388.25 |   199992.5 | ▇▇▇▇▇ |
| Exited          |          0 |              1 |        0.20 |     0.40 |        0.00 |        0.00 |        0.00 |        0.00 |        1.0 | ▇▁▁▁▂ |

The skim output shows that most of the variable names have a mixture of
upper and lower case characters. The set\_names functions coverts all
variable names to lower case. This makes the programming a little bit
easier. Further to this there is a variable ‘RowNumber’ and ‘customerid’
which just references the row of each observation and a unique
identifier for each customer. These can be removed from the analysis.

``` r
# Convert all names to lower case
df_churn <-
  df_churn %>%
    set_names(., tolower(names(.))) %>%
    select(-c(rownumber, customerid))
```

## Exploring Target Variable

The target variable in this dataset is **exited**, the chart below shows
that roughly 80% of customers remain while the remaining 20% of
customers churn. This is a fairly unbalanced data, so we might have to
treat that at another time. The target variable is stored as numeric, I
will convert it to a factor with levels relating to remain and churn.

<img src="tidymodels_exploration_files/figure-gfm/unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

## Exploring Categorical Variables

The chart below indicates that male customers have a lower risk of
churning that female customers. Looking at the Geography variable, there
doesn’t appear to be a difference between the underlying rates of churn
for customers from France and Spain. However, comparing the both of
those to Germany, they both have much lower likelihood of churn.

<img src="tidymodels_exploration_files/figure-gfm/unnamed-chunk-6-1.png" style="display: block; margin: auto;" />

The final categorical variable is the customer’s surname, of the 10,000
observations, there are just under 3,000 unique surnames. This indicates
that some customers come from families (some quite large) who all have
accounts with the same bank. The histogram indicates that the majority
of the bank customers are the only one from their family that bank
there. However, a number of customers do have multiple family members.
There appears to be a relationship that the larger the family size who
have an account the lower the churn rate. In the following section it
could be good to explore additional ways of including the surname into
the model.

``` r
# Land and structure based variables
grid.arrange(
             widths = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
             layout_matrix = 
                      rbind(c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                            c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                            c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                            c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                            c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                            c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
                            c(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
                            c(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
                            c(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
                            c(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
                            c(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
                            c(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)),

            # Unique surname
            df_churn %>%
              select(surname) %>%
              unique() %>%
              count() %>%
              ggplot(aes(x = "Unique Names", y = n)) +
              geom_col(fill = "blue") +
              scale_y_continuous(labels = scales::comma) +
              labs(y = NULL, x = NULL) +
              theme(plot.title = element_text(hjust = 0.5),
                    legend.position = "top") +
              ggtitle("Number of Unique Surnames"), 
            
            
            # Distribution of Family Names
            df_churn %>%
              group_by(surname, exited) %>%
              count() %>%
              spread(exited, n) %>%
              replace_na(list(Churn = 0, Remain = 0)) %>%
              mutate(Total = Churn + Remain) %>%
              arrange(desc(Total)) %>%
              ungroup() %>%
              ggplot(aes(x = Total)) +
                geom_histogram(fill = "red") +
                scale_x_continuous(breaks = seq(0,35,2)) +
                scale_y_continuous(labels = scales::comma) +
                labs(y = NULL, x = NULL) +
                theme(plot.title = element_text(hjust = 0.5),
                      legend.position = "top") +
                ggtitle("Distribution of Counts of Surnames"), 
            
            # Distribution of surname
            df_churn %>%
              group_by(surname, exited) %>%
              count() %>%
              spread(exited, n) %>%
              replace_na(list(Churn = 0, Remain = 0)) %>%
              mutate(Total = Churn + Remain,
                     churn_perc = Churn / Total) %>%
              ggplot(aes(x = Total, y = churn_perc)) +
              geom_point(col = "orange") +
              scale_y_continuous(labels = scales::percent) +
              labs(y = NULL, x = "Family members with an account") +
              theme(plot.title = element_text(hjust = 0.5),
                    legend.position = "top") +
              ggtitle("Churn Perc vs Total Occurance of Surname")
)
```

![](tidymodels_exploration_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## Exploring Numerical Variables

The plot looks at the relationship between each numeric variable and
churn. Two of the variables ‘hascrcard’ and ‘isactivemember’ are binary
variables and should be visualised in a different form. The remaining
variables can be summarised as:

  - Age: There appears to be a relationship that customers that churn
    are older on average than customers that don’t
  - Balance: The average balance of customers who churn and remain are
    similar. However, those who churn tend to have a higher balance
  - Credit score: There doesn’t appear to be much of a difference with
    regards to creditscore
  - Estimated salary: Again there doesn’t appear to be much of a
    difference with regard to salary
  - Number of products: On average customers that remain have more than
    one product, while customers that leave have only one
  - Tenure: There is a wider distribution of tenure within customers
    that churn, it appears that newer and older customers could have a
    higher likelihood of churning

<!-- end list -->

``` r
# Relationship with Churn and Numerical variables
df_churn %>%
  {bind_cols(select_if(., is.numeric),
             select_at(., "exited"))
  } %>%
  gather(-exited, key = "var", value = "value") %>%
  ggplot(aes(x = exited, y = value, fill = exited)) +
    geom_boxplot() +
    theme(legend.position = "none") + 
    facet_wrap(~ var, scales = "free")  +
    ggtitle("Numerical Variable Relationship with Churn")
```

<img src="tidymodels_exploration_files/figure-gfm/unnamed-chunk-8-1.png" style="display: block; margin: auto;" />

As mentioned above the ‘credit card’ and ‘Active member’ variables
shouldn’t be visualised using a boxplot. To generate the following plot,
the variables are firstly converted to factors and then the plot
explores whether there is a relationship with churn. Having a credit
card doesn’t appear to impact the customers likelihood to churn.
Alternatively, whether the customer is considered active, definately has
a relationship, ‘Active’ customers have much lower liklihood of
churning.

<img src="tidymodels_exploration_files/figure-gfm/unnamed-chunk-9-1.png" style="display: block; margin: auto;" />

The plot below indicates that the numerical predictors are not
particularly correlated, so no actions are required to address
multicollinariety.

``` r
# Create a variable of total family size
corrplot(cor(df_churn %>% select_if(is.numeric)))
```

![](tidymodels_exploration_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

# Feature Engineering

An important part of the model building process is to try and create new
predictors which will improve the accuracy of the model. The information
gained from the exploratory analysis, provides a couple of ideas as to
which variables can be generated.

### Surname

Given that the surname has circa 3k unique entries, it is unlikely that
the variable will be entered into the model. Instead some of the
information embedded will be used to generate new variables. The first
variable which will be created is a count of the total number of family
members with the same surname. As seen in the section above this
appeared to show a negatively correlated relationship.

``` r
# Create a variable of total family size
df_churn <-
  df_churn %>%
  left_join(df_churn %>%
              group_by(surname, exited) %>%
              count() %>%
              spread(exited, n) %>%
              replace_na(list(Churn = 0, Remain = 0)) %>%
              mutate(total_family = Churn + Remain) %>%
              select(surname, total_family),
            by = "surname")
```

Without commiting to thorough text analytics, there might be some value
in seeing if the attributes of a customer’s name has a relationship with
whether they churn. A simple approach is taken here, firstly 26 new
columns are created to capture each letter of the alphabet a:z. Then for
each surname those columns are populated with the counts of times the
letter appeared in the customer’s name. The final step is to try and
reduce the new 26 columns down, a Principle Component Analysis (PCA) was
conducted to achieve this. The output below shows that the first 10
principle components represent 75% of the variation in the letters
associated with customer’s names.

``` r
#Create a df with 26 columns
df_alphabet <-  
  cbind(
        df_churn %>%
          select(surname),
        matrix(0, nrow = nrow(df_churn), ncol = 26) %>%
        as.data.frame() %>%
        set_names(., c("a", "b", "c", "d", "e", "f", "g", "h", 
                       "i", "j", "k", "l", "m", "n", "o", "p", 
                       "q", "r", "s", "t", "u", "v", "w", "x", 
                       "y", "z"))
        )

# Fill df with counts of letters in customer's name
df_alphabet[2:ncol(df_alphabet)] <-
  sapply(names(df_alphabet[2:ncol(df_alphabet)]), function(x) {str_count(df_alphabet[,1], x)})
  

# Create PCA analysis and plot
pca <- prcomp(df_alphabet[2:ncol(df_alphabet)])
summary(pca) 
```

    ## Importance of components:
    ##                           PC1    PC2    PC3     PC4     PC5     PC6     PC7
    ## Standard deviation     0.7707 0.7152 0.6944 0.66855 0.62850 0.58522 0.56174
    ## Proportion of Variance 0.1188 0.1023 0.0965 0.08943 0.07904 0.06853 0.06314
    ## Cumulative Proportion  0.1188 0.2212 0.3177 0.40713 0.48617 0.55470 0.61784
    ##                            PC8     PC9    PC10    PC11    PC12    PC13    PC14
    ## Standard deviation     0.50341 0.46637 0.44802 0.41093 0.39631 0.36110 0.34823
    ## Proportion of Variance 0.05071 0.04352 0.04016 0.03379 0.03143 0.02609 0.02426
    ## Cumulative Proportion  0.66855 0.71207 0.75223 0.78602 0.81745 0.84354 0.86781
    ##                           PC15   PC16    PC17    PC18    PC19    PC20    PC21
    ## Standard deviation     0.33924 0.3209 0.29870 0.28208 0.26311 0.25269 0.22539
    ## Proportion of Variance 0.02303 0.0206 0.01785 0.01592 0.01385 0.01278 0.01016
    ## Cumulative Proportion  0.89083 0.9114 0.92929 0.94521 0.95906 0.97184 0.98201
    ##                           PC22    PC23    PC24    PC25    PC26
    ## Standard deviation     0.19837 0.18813 0.08883 0.08001 0.02989
    ## Proportion of Variance 0.00787 0.00708 0.00158 0.00128 0.00018
    ## Cumulative Proportion  0.98988 0.99696 0.99854 0.99982 1.00000

The plot below shows the distribution of churn and remain for the first
10 principle components just created. There doesn’t appear to be much
variation based on the name based components, however this will be
explored more rigourously during feature selection.

``` r
# Visualise relationship with churn
cbind(df_churn %>% select(exited), pca$x[,1:10]) %>%
  gather(-exited, key = "var", value = "value") %>%
  mutate(var = as.numeric(substr(var, 3,4))) %>%
  ggplot(aes(x = exited, y = value, fill = exited)) +
    geom_boxplot() +
    facet_grid(. ~ var) +
    theme(axis.text.x = element_text(angle = 90),
          legend.position = "none",
          plot.title = element_text(hjust = 0.5)) +
    ggtitle("Surname PCA and Churn/ Remain")
```

![](tidymodels_exploration_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
# Update churn dataframe
df_churn <-
  cbind(df_churn,
        pca$x[,1:10]) %>%
  rename_at(vars(starts_with('PC')), funs(paste0('Name', .)))

rm(df_alphabet, pca) # reduce memory size
```

#### Tenure

The exploratory analysis revealed that the distribution of tenure for
customers who left the bank was a lot wider than those who remained. It
suggested that maybe newer and longer standing customers were at higher
risk of churn, while medium term customers had a lower likelihood. As
such a new variable will be created which splits tenure into less than 3
being a new cusotmer and being longer than 7 being older customer.

``` r
# Create a variable
df_churn <-
  df_churn %>%
    mutate(tenure_fct = factor(case_when(tenure <= 3 ~ "New",
                                         tenure >= 7 ~ "Long",
                                         TRUE ~ "Medium"),
                               level = c("New", "Medium", "Long")))
```

# Model Building

Now that the data has been fully prepared we can progress to the model
building phase. This is where we can take the tidymodels suite of
packages for a spin and explore the functionality for model development.
A number of different models will be built to predict churn, where
necessary the models will tuned and evaluated using the ROC metric. The
following models which will be built and tuned using the tidymodel
framework:

  - GLM
  - GLMNET
  - Random Forest

The steps which will be explored in this section are:

  - Split the data into training and testing sets
  - Conduct feature selection to identify which variables to actually
    include in this analysis
  - Determine which pre-processing steps are required
  - Build, compare and choose best cross validated model based on AUROC
  - Calculate probability thresholds to determine optimal cut off
  - Apply best model to test data and calculate evaluation statistics
  - Compare standard cut off to calculated threshold

## Split Data

To ensure we fairly evaluate our model, we will utilise the
inital\_split function from rsample package. This function allows us to
randomly split the churn dataset into two, one for testing and one for
training. There is an option for stratified sampling along the target
variable to ensure that the distributions are the same in both training
and testing datasets. Once an index has been created the functions
training and testing can be used to create the two datasets. The table
below shows that the inital\_split function has achieved the same
distribution of churn and remain observations in the training and
testing datasets to 4dp.

``` r
# Create an index to split the data
set.seed(1989)
l_split_index <- 
  initial_split(data = df_churn,
                prop = 0.75,
                strata = exited
                )

# Create training and testing set using the index
df_train <- training(l_split_index)
df_test <- testing(l_split_index)
  
# Check same distribution of output
bind_rows(
          as.data.frame(round(prop.table(table(df_train$exited)),4)) %>% mutate(Data = "Train"),
          as.data.frame(round(prop.table(table(df_test$exited)),4)) %>% mutate(Data = "Test")
          ) %>%
  spread(Var1, Freq) %>%
  kable(style = "pandoc",
        align = c('c','c','c'))
```

| Data  | Churn  | Remain |
| :---: | :----: | :----: |
| Test  | 0.2037 | 0.7963 |
| Train | 0.2037 | 0.7963 |

## Feature Selection

Generally in feature selection, there would be a more formal assessment
of each feature with the target variable. However, in this analysis as
the focus is mostly on getting to know tidymodels this section will
merely be selecting between the existing and new variables created.

I will run a quick random forest model to determine the feature
importance of the variables. I couldn’t seem to find documentation
showing how to extract the variable importance when using tidymodels. I
decided to progress and use caret here, to keep the analysis moving.

The results from the random forest show that the model isn’t that great
at prediction but the following variables rank in importance:

  - keep tenure and drop tenure\_fct
  - Keep total family variable
  - PCA variables aren’t the best but aren’t the worst and can stay

<!-- end list -->

``` r
# Caret model
set.seed(1989)
invisible(
          feature_select_ranger <-
            train(exited ~ .,
                data = df_train %>% select(-surname),
                method = "ranger",
                metric = "ROC",
                importance = 'impurity',
                preProcess = c("center", "scale"),
                tuneLength = 3,
                trControl = trainControl(
                                         verboseIter = FALSE,
                                         savePredictions = TRUE,
                                         summaryFunction = twoClassSummary,
                                         classProbs = TRUE,
                                         index = createFolds(df_train[, "exited"], k = 5)
                                         )
                  )
          )

# view importance
varImp(feature_select_ranger) 
```

    ## ranger variable importance
    ## 
    ##   only 20 most important variables shown (out of 24)
    ## 
    ##                  Overall
    ## age              100.000
    ## numofproducts     92.463
    ## balance           32.226
    ## estimatedsalary   17.219
    ## isactivemember    17.043
    ## creditscore       17.031
    ## NamePC4           15.730
    ## total_family      15.405
    ## NamePC2           15.174
    ## NamePC7           14.512
    ## NamePC5           14.331
    ## NamePC8           14.300
    ## NamePC3           14.226
    ## NamePC9           13.878
    ## NamePC10          13.670
    ## NamePC1           13.664
    ## NamePC6           13.399
    ## tenure            12.836
    ## geographyGermany  10.429
    ## hascrcard          4.575

``` r
# Drop objects
rm(feature_select_ranger)
```

## Pre processing data

We can now use the recipes package to conduct preprocessing,
pre-processing allows us to conduct transformations of the data which
hopefully improves the models ability to predict. As the name
suggestions, recipes has been constructed as an emulation of the food
preparation process. The key functions are:

  - recipes: This allows us to determine what preprocessing we wish to
    apply to our data
  - prep: Estimates the parameters assocated with the chosen recipe
  - bake: Applies the actual two steps above

Using the recipes package, two different steps of pre-processing will be
used to compare prediction accuracy with and without PCA. Both receipes
will include the same steps except from the last which will be either
removing variables with near zero variance or applying principle
component analysis.

Using the recipes package,steps that will be applied are:

  - Remove variables from feature selection
  - Covert nomial variables to dummy variables
  - Centre and scale all numerical variables
  - Remove highly correlated variables
  - Remove variables with no variance
  - Remove variables with minimal variance

Once the recipes have been created, the bake function is used to apply
the transformations to the training and testing set and ultimately
replace the old dataframes.

``` r
# Preprocessing - With near zero inflation
recipe <-
  df_train %>%
    recipe(exited ~ .) %>%
    step_rm(surname, tenure_fct) %>%
    step_dummy(all_nominal(), -all_outcomes()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    step_corr(all_numeric(), -all_outcomes(), threshold = 0.9) %>%
    step_zv(all_predictors()) %>%
    step_nzv(all_predictors()) %>%
    prep()

# Apply processing to test and training data
df_baked_train <- recipe %>% bake(df_train) # Preprocessed training
df_baked_test <- recipe %>% bake(df_test) # Preprocessed testing

rm(df_train, df_test) # remove old data
```

## Cross Validation

Cross validation allows us to explore multiple cuts of our data and to
gain confidence that the model performs well on multiple slices of the
data, which reduces bias and overfitting. 5 fold cross validation will
be applied and for fair comparison across the various models, they will
be applied to the exact same splits of data.

The vfold\_cv function is used to create 5 equal splits in our baked
training data, again there is an option to stratify sampling along our
target variable. The object l\_cv is created to be used later in
upcoming sections.

``` r
# Cross validation
set.seed(1989)
l_cv <- vfold_cv(df_baked_train, v = 5, strata = "exited") # Cross validation
```

## Develop competing models with tuning where required

To test the flexibility of tidymodel, the following three models will be
used:

  - glm
  - glmnet
  - random forest

In each of the models mentioned above, 5 models will be built on the
datasplits created by rsamples for cross validation. glmnet and random
forest have hyper parameters, so functionality from tune and dials will
be used to explore the best hyperparameters with respect to roc.

### GLM

Logistic regression is a very basic but surprisingly effective binary
classification algorithm, when appropriately built. As this is an
exercise to explore tidy models, a quick logistic regression model will
be built and the variables which are deemed significant will be used as
the baseline model. The variables in the table below are the one’s which
will be used as part of the baseline model.

``` r
# Get features
logistic_reg(mode = "classification") %>%
  set_engine("glm") %>%
  fit(exited ~ ., data = df_baked_train) %>%
  tidy() %>%
  filter(p.value < 0.05) %>%
  kable()
```

| term               |    estimate | std.error |   statistic |   p.value |
| :----------------- | ----------: | --------: | ----------: | --------: |
| (Intercept)        |   1.6535658 | 0.0356813 |   46.342581 | 0.0000000 |
| creditscore        |   0.0635297 | 0.0311580 |    2.038949 | 0.0414551 |
| age                | \-0.7578371 | 0.0312033 | \-24.287055 | 0.0000000 |
| tenure             |   0.0734776 | 0.0312311 |    2.352709 | 0.0186372 |
| balance            | \-0.1658047 | 0.0369276 |  \-4.489997 | 0.0000071 |
| isactivemember     |   0.5111766 | 0.0330989 |   15.443899 | 0.0000000 |
| geography\_Germany | \-0.3357994 | 0.0338382 |  \-9.923678 | 0.0000000 |
| gender\_Male       |   0.2959252 | 0.0313505 |    9.439244 | 0.0000000 |

The function below makes use of purrr’s map2\_df function which allows
iteration over multiple factors and returns a dataframe. The map2\_df
function interates over the cross validation object ‘l\_cv’ which was
created above.

Within each iteration:

  - The analysis and assessment functions are used to split the data
    effectively into training and testing. Note the deliberate
    difference in the names of functions as supposed to training and
    testing
  - A GLM model is built on the analysis data, with the columns in the
    table above
  - The GLM model is used to predict the target variable on the
    assessment data, both the class and probability is retained in a
    table
  - The output is stored in a list to be used later in the analysis

<!-- end list -->

``` r
# glm - 5 fold CV
mod_glm <-
  list(parameters = NULL,
       df = map2_df(.x = l_cv$splits,
                    .y = l_cv$id,
                    function (split = .x, fold = .y) 
                       {
                         # Split the data into analysis and assessment tables
                         df_analysis <- analysis(split)
                         df_assessment <- assessment(split)
                         
                         # Build the model
                         mod <-
                          logistic_reg(mode = "classification") %>%
                          set_engine("glm") %>%
                          fit(exited ~ creditscore + age + tenure + 
                                       balance + isactivemember + 
                                       geography_Germany + gender_Male, 
                              data = df_analysis)
                         
                         # Summarise Predictions
                         table <- 
                           tibble(fold = fold,
                                  truth = df_assessment$exited,
                                  .pred_Churn = predict(mod, 
                                                        new_data = df_assessment, 
                                                        type = "prob")[[".pred_Churn"]],
                                  .pred_Remain = predict(mod, 
                                                         new_data = df_assessment, 
                                                         type = "prob")[[".pred_Remain"]],
                                  .pred_Class = predict(mod, 
                                                        new_data = df_assessment) %>% 
                                    unlist() %>% 
                                    as.character()
                                  ) %>%
                           mutate(.pred_Class = factor(.pred_Class))
                         }) 
       )
```

### GLMNET

The second model is an extension of the glm model created above and is
commonly refered to as elastic-net logistic regression. This algorithm
adds regularision to combine ridge and lasso regression, which
ultimately control the size of the model. This method has two
hyperparameters ‘penalty’ and ‘mixture’, meaning an additional step in
the process is required.

The additional steps are:

  - Within the model build, the penalty and mixture features are equated
    with tune functions. This indicates that some form of search will be
    conducted across these parameters
  - The second step is to use the tune grid, parameters and
    grid\_max\_entropy functions from the tune and dials packages.
      - Grid\_max\_entropy creates a grid of size 50 in this case of
        random parameters which cover the parameter search space
      - The tune\_grid function takes a model formula, the cross
        validation and a metric to optimse (roc in this case). The
        function uses the cross validation to estimate the roc for the
        different parameters in the grid
      - The select\_best function from the tune package is used to
        retrieve the hyperparameter values which yield the best results

Once the model is built it is re-applied to the cross validated data and
the predictions and parameters are stored in a list for a later stage.

``` r
# Set the model engine
mod <-
  logistic_reg(mode = "classification",
    penalty = tune(), 
    mixture = tune()) %>%
    set_engine("glmnet") 

# Build initial model with varying parameters and cross validation
set.seed(1989)
mod_results_tbl <- 
  tune_grid(
            formula   = exited ~ .,
            model     = mod,
            resamples = l_cv,
            grid      = grid_max_entropy(parameters(penalty(), 
                                                    mixture()), 
                                         size = 50),
            metrics   = metric_set(roc_auc),
            control   = control_grid(verbose = FALSE)
      )

# Store the parameters
df_parameter <- mod_results_tbl %>% select_best("roc_auc")

# glmnet - 5 fold CV
mod_glmnet <-
  list(parameters = df_parameter, 
       df = map2_df(.x = l_cv$splits, # Add predictions and find c
                    .y = l_cv$id,
                    function (split = .x, fold = .y)
                     {
                       # Split the data into analysis and assessment tables
                       df_analysis <- analysis(split)
                       df_assessment <- assessment(split)
  
                       # Build the model
                       mod_2 <-
                        logistic_reg(mode = "classification",
                                     penalty = as.numeric(df_parameter["penalty"]),
                                     mixture = as.numeric(df_parameter["mixture"])
                                     ) %>%
                         set_engine("glmnet") %>%
                         fit(exited ~ ., data = df_analysis)
  
                       # Summarise Predictions
                       table <-
                         tibble(fold = fold,
                                truth = df_assessment$exited,
                                .pred_Churn = predict(mod_2,
                                                      new_data = df_assessment,
                                                      type = "prob")[[".pred_Churn"]],
                                .pred_Remain = predict(mod_2,
                                                       new_data = df_assessment,
                                                       type = "prob")[[".pred_Remain"]],
                                .pred_Class = predict(mod_2, new_data = df_assessment) %>%
                                  unlist() %>%
                                  as.character()
                                ) %>%
                         mutate(.pred_Class = factor(.pred_Class))
                      })
    )
          
rm(mod, mod_results_tbl, df_parameter) # Clear memory
```

### Random Forest

The final model variant to be explored is random forest which is a tree
based model and differs from the two models explored thus far. Again
this model has hyperparameter and the tidymodel packages will be used to
aid in tuning.

The steps in the build are:

  - Create a model object, setting the hyperparameters to tune and
    setting the engine to ranger. This means using the package ranger’s
    implementation of random forest
  - Secondly, the grid\_random function is used to generate 50 random
    combinations of parameters mtry, trees and min\_n. mtry requires an
    upper bound to be set, which in this case is set at the number of
    predictors
  - The best parameters are stored and applied in the cross validation
    as done in the previous step
  - The output is stored in a list for later use

<!-- end list -->

``` r
# Set the model engine
mod <-
 rand_forest(mode = "classification",
             mtry = tune(),
             trees = tune(),
             min_n = tune()) %>%
set_engine("ranger") 

# Build initial model with varying parameters and cross validation
set.seed(1989)
mod_results_tbl <- 
  tune_grid(
    formula   = exited ~ .,
    model     = mod,
    resamples = l_cv,
    grid      = grid_random(parameters(mtry(c(1, 22)), 
                                       trees(), 
                                       min_n()), 
                            size = 50),
    metrics   = metric_set(roc_auc),
    control   = control_grid()
      )

# Store the parameters
df_parameter <- mod_results_tbl %>% select_best("roc_auc")

# random forest - 5 fold CV
mod_rf <-
  list(parameters = df_parameter,
       df = map2_df(.x = l_cv$splits,
                    .y = l_cv$id,
                    function (split = .x, fold = .y)
                     {
                       # Split the data into analysis and assessment tables
                       df_analysis <- analysis(split)
                       df_assessment <- assessment(split)
  
                       # Build the model
                       mod_2 <-
                        rand_forest(mode = "classification",
                                     mtry = as.numeric(df_parameter["mtry"]),
                                     trees = as.numeric(df_parameter["trees"]),
                                     min_n = as.numeric(df_parameter["min_n"])
                                     ) %>%
                         set_engine("ranger") %>%
                         fit(exited ~ ., data = df_analysis)
  
                       # Summarise Predictions
                       table <-
                         tibble(fold = fold,
                                truth = df_assessment$exited,
                                .pred_Churn = predict(mod_2,
                                                      new_data = df_assessment,
                                                      type = "prob")[[".pred_Churn"]],
                                .pred_Remain = predict(mod_2,
                                                       new_data = df_assessment,
                                                       type = "prob")[[".pred_Remain"]],
                                .pred_Class = predict(mod_2, new_data = df_assessment) %>%
                                  unlist() %>%
                                  as.character()
                                ) %>%
                         mutate(.pred_Class = factor(.pred_Class))
                        })
          )

rm(mod, mod_results_tbl, df_parameter) # Clear memory
```

## Model Evaluation metrics

This section summarises the performance of the three models from the
cross validation and hyperparameter tuning section above. The evaluation
metrics which can be seen in the plot below are:

  - accuracy: the proportion of observations the model correctly
    predicted
  - sensitivity (true positive rate): the proportion of actual positives
    that are correctly identified
  - specificity (true negative rate): the proportion of actual negatives
    that are correctly identified
  - auroc: Measures the entire 2d area under the ROC curve, which
    measures the power of the model under all possible classification
    thresholds

The plot below shows:

  - Consistently across the metrics the random forest model is the best
    performing
  - Despite minimal efforts to develop a robust GLM model, it holds it’s
    own against the glmnet model and breaks away when sensitivity is
    considered
  - All models have a higher specificity than sensitivity, most likely
    relating to the class imbalance

<!-- end list -->

``` r
# Extract the cross validated predictions
mod_summary_all <-
  bind_rows(mod_glm[["df"]] %>% mutate(model = "glm"),
            mod_glmnet[["df"]] %>% mutate(model = "glmnet"),
            mod_rf[["df"]] %>% mutate(model = "randomforest"))

# Summarise evaluation stats  
lapply(1:5, function(x)
  {
  bind_rows(
            # -- Accuracy
    
            # GLM
            mod_summary_all %>% 
              filter(fold == paste("Fold", x, sep = "") & model == "glm") %>%
              conf_mat(truth = truth, estimate = .pred_Class) %>%
              summary() %>%
              filter(.metric %in% c("accuracy", "sens", "spec")) %>%
              mutate(fold = x,
                     model = "glm"),
            
            # GLMNET
            mod_summary_all %>% 
              filter(fold == paste("Fold", x, sep = "") & model == "glmnet") %>%
              conf_mat(truth = truth, estimate = .pred_Class) %>%
              summary() %>%
              filter(.metric %in% c("accuracy", "sens", "spec")) %>%
              mutate(fold = x,
                     model = "glmnet"),  
            
            # Random Forest
            mod_summary_all %>% 
              filter(fold == paste("Fold", x, sep = "") & model == "randomforest") %>%
              conf_mat(truth = truth, estimate = .pred_Class) %>%
              summary() %>%
              filter(.metric %in% c("accuracy", "sens", "spec")) %>%
              mutate(fold = x,
                     model = "randomforest"),  

            # -- ROC  
            
            # GLM
            mod_summary_all %>% 
              filter(fold == paste("Fold", x, sep = "") & model == "glm") %>%
              roc_auc(truth, .pred_Churn) %>%
              mutate(fold = x,
                     model = "glm"),
            
            # GLMNET
            mod_summary_all %>% 
              filter(fold == paste("Fold", x, sep = "") & model == "glmnet") %>%
              roc_auc(truth, .pred_Churn) %>%
              mutate(fold = x,
                     model = "glmnet"),            
             
            # GLMNET
            mod_summary_all %>% 
              filter(fold == paste("Fold", x, sep = "") & model == "randomforest") %>%
              roc_auc(truth, .pred_Churn) %>%
              mutate(fold = x,
                     model = "randomforest")
            )}) %>%
  rbindlist() %>%
  arrange(fold, model, .metric) %>% 
  ggplot(aes(x = fold, y = .estimate, col = model)) +
    geom_point() +
    facet_grid(. ~ .metric) +
    scale_y_continuous(labels = percent, breaks = seq(0, 1, 0.1)) +
    theme(legend.position = "top",
          plot.title = element_text(hjust = 0.5)) +
    ggtitle("Prediction Evaluation Statistics ")
```

![](tidymodels_exploration_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

## Probability thresholds

One method of addressing class imbalance (ie. churn and remain having
different proportions) is to adjust the threshold that determines
whether the probability means churn on remain. Typically the threshold
is set at 50%, however it can be changed to a threshold which provides a
better balance between sensitivity and specificity. In this case, the
model is required to estimate churn or remain equally so a new threshold
is desirable.

The code below utilises the prediction data from the cross validation
exercise, using the threshold\_perf function from the probably package,
evaluation metrics can be calculated for each 1% increment between 0 to
1 where the threshold is change. The table below shows the values of
sensitivity and specificity for the random forest model for each of the
5 folds and the cut off points required to achieve these. The analysis
shows that a threshold of around 20% would provide more even accuracy
between sensitivity and specificity which both seem to vary around
77-78%.

``` r
# probability thresholds by fold
df_thresholds <-
  mod_summary_all %>%
    filter(model == "randomforest") %>%
    select(fold, truth, .pred_Churn) %>%
    group_by(fold) %>%
    probably::threshold_perf(truth, .pred_Churn, thresholds = seq(0, 1, 0.01)) %>% 
    filter(.metric %in% c("spec", "sens", "distance")) %>%
    spread(.metric, .estimate) %>%
    mutate(distance_abs = abs(sens - spec)) %>%
    group_by(fold) %>%
    filter(distance_abs == min(distance_abs)) %>%
    ungroup() 

# Show table
df_thresholds %>% kable(align = c('c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'))
```

| fold  | .threshold | .estimator | distance  |   sens    |   spec    | distance\_abs |
| :---: | :--------: | :--------: | :-------: | :-------: | :-------: | :-----------: |
| Fold1 |    0.22    |   binary   | 0.0963778 | 0.7810458 | 0.7799163 |   0.0011294   |
| Fold2 |    0.22    |   binary   | 0.0952755 | 0.7777778 | 0.7857741 |   0.0079963   |
| Fold3 |    0.21    |   binary   | 0.1149949 | 0.7614379 | 0.7589958 |   0.0024421   |
| Fold4 |    0.20    |   binary   | 0.1194588 | 0.7540984 | 0.7571189 |   0.0030206   |
| Fold5 |    0.19    |   binary   | 0.1476899 | 0.7278689 | 0.7286432 |   0.0007744   |

Averageing the results from the cross validation above shows that the
average threshold which should be applied is 20.8% In essence is we
apply this threshold rather 50% we will reduce how accurate we are at
predicting customers who remain, but significantly improve how well we
predict customers that churn.

``` r
# Summary threshold
df_thresholds <-
  df_thresholds %>%
      summarise(.threshold = mean(.threshold),
                distance = mean(distance),
                sens = mean(sens),
                spec = mean(spec),
                distance_abs = mean(distance_abs))

# Show table
df_thresholds %>% kable(align = c('c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'))
```

| .threshold | distance  |   sens    |   spec    | distance\_abs |
| :--------: | :-------: | :-------: | :-------: | :-----------: |
|   0.208    | 0.1147594 | 0.7604457 | 0.7620897 |   0.0030725   |

# Prediction Evaluation

The final stage of this analysis is to use the models built and tuned on
the training data and see how well they perform on the unseen test data.
All three models will be used to predict whether customers churn or
remain in the testing dataset.

``` r
# Final glm
mod_final_glm <-
  logistic_reg(mode = "classification") %>%
    set_engine("glm") %>%
    fit(exited ~ ., df_baked_test)

# Final glmnet
mod_final_glmnet <-
  logistic_reg(mode = "classification",
               penalty = mod_glmnet[["parameters"]][["penalty"]], 
               mixture = mod_glmnet[["parameters"]][["mixture"]]
               ) %>%
    set_engine("glmnet") %>%
    fit(exited ~ ., df_baked_test)


# Final random forest
mod_final_rf <-
 rand_forest(mode = "classification",
             mtry = mod_rf[["parameters"]][["mtry"]],
             trees = mod_rf[["parameters"]][["trees"]],
             min_n = mod_rf[["parameters"]][["min_n"]]
             ) %>%
  set_engine("ranger") %>%
  fit(exited ~ ., df_baked_test)
```

The table below shows the performance of the models on the test data:

  - The random forest model is the clear winner with respect to both
    auroc and accuracy
  - Glm marginally outperforms the glmnet model across both metrics

Given the superior performance achieved by the random forest model, it
will be labelled as our champion.

``` r
# Summary of prediction performance
bind_rows(
          predict(mod_final_glm, new_data = df_baked_test, type = "prob") %>%
            bind_cols(predict(mod_final_glm, new_data = df_baked_test)) %>%
            bind_cols(df_baked_test) %>%
            metrics(exited, .pred_Remain, estimate = .pred_class) %>%
            mutate(model = "glm"),
          
          predict(mod_final_glmnet, new_data = df_baked_test, type = "prob") %>%
            bind_cols(predict(mod_final_glmnet, new_data = df_baked_test)) %>%
            bind_cols(df_baked_test) %>%
            metrics(exited, .pred_Remain, estimate = .pred_class) %>%
            mutate(model = "glmnet"),
          
          predict(mod_final_rf, new_data = df_baked_test, type = "prob") %>%
            bind_cols(predict(mod_final_rf, new_data = df_baked_test)) %>%
            bind_cols(df_baked_test) %>%
            metrics(exited, .pred_Remain, estimate = .pred_class) %>%
            mutate(model = "randomforest")
          ) %>%
  filter(.metric %in% c("accuracy", "roc_auc")) %>%
  spread(.metric, .estimate) %>%
  select(-.estimator) %>%
  kable(align = c('c', 'c', 'c'))
```

|    model     | accuracy  | roc\_auc  |
| :----------: | :-------: | :-------: |
|     glm      | 0.8155262 | 0.7788283 |
|    glmnet    | 0.8083233 | 0.7756390 |
| randomforest | 0.9155662 | 0.9789261 |

The last stage of this analysis is to compare the predictions from the
random forest model with the standard threshold of 50% and the
calculated threshold of 21%. These results can be seen in the table
below and as expected the sensitivity has increased dramatically and the
specificity has reduced marginally. The expecation was that the two
metrics would have been similar, the fact that they are not may indicate
some bias between our testing and training data.

The final step which will not be completed here is that the random
forest model is re-built on the entire dataset.

``` r
# Comparison of evaluation at 50% and calculate thresholds
bind_rows(
          # 50% cut off
          predict(mod_final_rf, new_data = df_baked_test) %>%
            bind_cols(df_baked_test) %>%
            conf_mat(truth = exited, estimate = .pred_class) %>%
            summary() %>%
            filter(.metric %in% c("accuracy", "sens", "spec")) %>%
            mutate(.cutoff = "50_perc"),
          
          # Balanced cut off
          predict(mod_final_rf, new_data = df_baked_test, type = "prob") %>%
            mutate(.pred_class = factor(ifelse(.pred_Churn >= df_thresholds$.threshold, "Churn", "Remain"))) %>%
            bind_cols(df_baked_test) %>%
            conf_mat(truth = exited, estimate = .pred_class) %>%
            summary() %>%
            filter(.metric %in% c("accuracy", "sens", "spec")) %>%
            mutate(.cutoff = paste(round(df_thresholds$.threshold * 100),"_perc", sep = ""))
          ) %>%
  spread(.metric, .estimate) %>%
  select(-.estimator) %>%
  mutate(accuracy = round(accuracy, 3),
         sens = round(sens, 3),
         spec = round(spec, 3)) %>%
  kable(align = c('c', 'c', 'c', 'c'))
```

| .cutoff  | accuracy | sens  | spec  |
| :------: | :------: | :---: | :---: |
| 21\_perc |  0.866   | 0.990 | 0.835 |
| 50\_perc |  0.916   | 0.623 | 0.990 |

# Conclusion

In summary the tidymodels meta package is a really good wrapper for
predictive modelling and sits within the familiar ethos of the
tidyverse. The packages are undergoing continued development, so I am
sure additional functionality will be added overtime. The functionality
provided makes the model development and assessment process realitively
simple by breaking it down into smaller more managiable chunks.
