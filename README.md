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

We consider binary classification problems:
  - **30-days delinquency VS all**: Predict if next month will be 30-days delinquent or not 
  - **Prepayment VS all**: Predict if next month will be prepayment or not

### Algorithms

We compare 3 algorithms:
  - Logistic Regression
  - Feed forward Neural Network
  - LSTM

The models are defined in `Models.py`

### Training

Files:
  - `DataGen.py`: This file has code to structure and preprocess data from hdfs files into a format that can be directly fed to the models. More information is provided in the definition of the methods about what they do.
  - `ModelTrainer.py`: Defines training loop, testing loop and metric definition.
  - `Train_main.py`: Defines parameters and hyperparameters of the model and calling `ModelTrainer.py` functions to train and test.
  - `config.py`: Defines all the configurations in this file.
  
 ### Results
 
 Detailed multiple reports are present in `Reports` folder.
 
Table to show Max AUC and Mean AUC for 30 days delinquency.
 
| | Max AUC | Mean of AUC |
| ------------- |:-------------:| -----:|
Logistic Regression |0.710236 | 0.667079 |
Feed forward network | 0.716513 | 0.700304 |
LSTM | 0.742221 | 0.653029 |

Table to show Max AUC and Mean AUC for Prepayment.

| | Max AUC | Mean of AUC |
| ------------- |:-------------:| -----:|
| Logistic Regression | 0.558171 | 0.535332 |
| Feed forward network | 0.631124 | 0.593673 |
| LSTM | 0.769554 | 0.721963 |
