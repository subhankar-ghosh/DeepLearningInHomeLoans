# Deep Learning In Home Loans

### Data

We use Freddie Mac loan level dataset for our experiments. You can find more information about the dataset here: [Loan-level Dataset Overview](http://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf).
You can download the [dataset](http://www.freddiemac.com/research/datasets/sf_loanlevel_dataset.html). We used quarterly dataset, where every file contains loan data every quarter.

Every quarterly dataset has two files:
  - Loan data file: Information about every loan. Every row is about a unique loan, it has all the information about that particular loan.
  - Monthly data file: Monthly data about the loans. It contains all the variables that might change every month.
  
##### Converting data to hdfs format

Run `python3 Preprocess_main.py` to start preprocessing the data. We have to provide `dataDirectory` in the `Preprocess_main.py` script which is the path to the directory where the raw data is downloaded.

`Preprocess_main.py` internally calls `PreprocessData` function of `Preprocess` class from `Preprocess.py` which converts the data into hdfs format and splits it into train-test set. We make sure that there is no overlap in the training and testing set, to ensure this we use loans originating in certain years as training set and loans originating in other years as testing set.

### Formulation of Problem

We consider `monthly status` of the loans as label of the data. It has values like `Current` means payment is made on time, `30-days delinquency` means not paid for 30 days, `60-days delinquency` means not paid for 60 days, ..., `Prepayment` loan is pre paid before end of loan date and more, check the *Loan-level Dataset Overview*.

### Algorithms

We compare 3 algorithms:
  - Logistic Regression
  - Feed forward Neural Network
  - LSTM
