# Optimal MLP Layer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#------------------------- Feature Engineering -------------------------#

compas_scores_two_year= pd.read_csv("compas_scores_two_years.csv",  lineterminator='\n')

# Select features from dataset
df= compas_scores_two_year[[ 'juv_fel_count', 'juv_misd_count', 'juv_other_count' ,'age', 'c_charge_degree','race', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',  'c_jail_in', 'c_jail_out',  'v_decile_score','two_year_recid\r']]
# Process the data
df = df.loc[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30) & (df['is_recid'] != -1) & (df['c_charge_degree'] != 'O') & (df['score_text'] != 'N/A')]

#length of stay in jail 
df['length_of_stay'] = pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])
df['length_of_stay'] = df['length_of_stay'].astype('timedelta64[D]')
df['length_of_stay'] = df['length_of_stay'].astype(int)
length_factor, u_length_degree = pd.factorize(df['length_of_stay'])

quick_stay = []
short_stay=[]
medium_stay=[]
long_stay=[]

for length in length_factor:
    if length<5:
        quick_stay.append(1)
        short_stay.append(0)
        medium_stay.append(0)
        long_stay.append(0)
    elif (length<15):
        quick_stay.append(0)
        short_stay.append(1)
        medium_stay.append(0)
        long_stay.append(0)
    elif length<30:
        quick_stay.append(0)
        short_stay.append(0)
        medium_stay.append(1)
        long_stay.append(0)
    else:
        quick_stay.append(0)
        short_stay.append(0)
        medium_stay.append(0)
        long_stay.append(1)
        
#Age Category Feature
df_age = df['age'].astype(int)

twenties_and_less = []
thirties=[]
fourties=[]
fifties_and_more=[]

for age in df_age:
    if age<30:
        twenties_and_less.append(1)
        thirties.append(0)
        fourties.append(0)
        fifties_and_more.append(0)
    elif age<40:
        twenties_and_less.append(0)
        thirties.append(1)
        fourties.append(0)
        fifties_and_more.append(0)
    elif age<50:
        twenties_and_less.append(0)
        thirties.append(0)
        fourties.append(1)
        fifties_and_more.append(0)
    else:
        twenties_and_less.append(0)
        thirties.append(0)
        fourties.append(0)
        fifties_and_more.append(1)


# Degree of charge feature
df_c_charge_degree = df[['c_charge_degree']] 
crime_factor, u_charge_degree = pd.factorize(df_c_charge_degree['c_charge_degree'])

# Gender
df_sex = df[['sex']]
f_gender, uniques_gender  = pd.factorize(df_sex['sex'])

# Prior convictions
juvinile_felonies  = df[['juv_fel_count']].astype(int)
juvinile_misconduct  = df[['juv_misd_count']].astype(int)
juvinile_other  = df[['juv_other_count']].astype(int)
priors_count  = df[['priors_count']].astype(int)

#Race
AfAmerican =[]
Other = []
Caucasian = []
Hispanic = []

for race in df['race']:
    if race == 'African-American':
        AfAmerican.append(1)
        Caucasian.append(0)
        Hispanic.append(0)
        Other.append(0)
    elif race == 'Caucasian':
        AfAmerican.append(0)
        Caucasian.append(1)
        Hispanic.append(0)
        Other.append(0)
    elif race == 'Hispanic':
        AfAmerican.append(0)
        Caucasian.append(0)
        Hispanic.append(1)
        Other.append(0)
    else:
        AfAmerican.append(0)
        Caucasian.append(0)
        Hispanic.append(0)
        Other.append(1)

X = np.column_stack(( crime_factor, f_gender, priors_count, juvinile_felonies, juvinile_misconduct, juvinile_other, quick_stay, short_stay, medium_stay, long_stay, twenties_and_less, thirties, fourties, fifties_and_more))

#For target class
f_score_text, u_score_text = pd.factorize(df['score_text'] != 'Low')
#decile_score = df['decile_score']
two_year_recid = df[['two_year_recid\r']]


x_lables = [ 'Crime factor', 'Gender factor', 'Priors count', 'Juvinile felonies', 'Juvinile misconduct', 'Juvinile other', 'Quick stay', 'Short stay', 'Medium stay', 'Long stay', 'Twenties and less', 'Thirties', 'Fourties', 'Fifties and more']


# cross-validation for MLP takes a while so set them to false if you want to run only the last section
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
mean_error=[]; std_error=[]

cross_val_no_layers = False
if cross_val_no_layers:
    hidden_layer_range = [5,10,25,50,75,100]
    for n in hidden_layer_range:
        print("Hidden Layer Size: %d\n"%n)
        model = MLPClassifier(hidden_layer_sizes=(n), max_iter=1000)
        # f_score_text - the predicted chances of recidivism
        # two_year_recid - the actual recidivism occurences
        scores = cross_val_score(model, X, np.ravel(f_score_text), cv=5, scoring='f1')
        #scores = cross_val_score(model, X, np.ravel(two_year_recid), cv=5, scoring='f1')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(hidden_layer_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("#hidden layer nodes"); plt.ylabel('F1')
    plt.title("Cross validation for # Nodes")
    plt.show()


cross_val_C = False
if cross_val_C:
    mean_error=[]; std_error=[]
    C_range = [1,5,10,50, 100]
    for Ci in C_range:
        print("C: %d\n"%Ci)
        # from prev crossval, best to use no. of nodes = 10 for both pred and actual
        model = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000)
        # f_score_text - the predicted chances of recidivism
        # two_year_recid - the actual recidivism occurences
        scores = cross_val_score(model, X, np.ravel(f_score_text), cv=5, scoring='f1')
        #scores = cross_val_score(model, X, np.ravel(two_year_recid), cv=5, scoring='f1')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(C_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("C"); plt.ylabel('F1')
    plt.title("Cross validation for C")
    plt.show()


# Comparing the Predicted Recidivsm and the Actual Recidivism Models by Confusion Matrices
model = MLPClassifier(hidden_layer_sizes=(10), alpha=1/5).fit(X, np.ravel(two_year_recid))
preds = model.predict(X)
from sklearn.metrics import confusion_matrix
print("Confusion Matrix for Recdivism Rates")
print(confusion_matrix(np.ravel(two_year_recid),preds))

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent").fit(X,np.ravel(two_year_recid))
ydummy = dummy.predict(X)
print("Confusion Matrix for Baseline Classifier")
print(confusion_matrix(np.ravel(two_year_recid),ydummy))

model = MLPClassifier(hidden_layer_sizes=(10), alpha=1/10).fit(X, np.ravel(f_score_text))
preds = model.predict(X)
from sklearn.metrics import confusion_matrix
print("Confusion Matrix for  COMPAS Score")
print(confusion_matrix(np.ravel(f_score_text),preds))

# Not comparing performances of the model themselves, so no need for ROC Curves
# from sklearn.metrics import roc_curve
# preds = model.predict_proba(X)
# print(model.classes_)
# fpr,tpr,_=roc_curve(np.ravel(two_year_recid), preds[:,1])
# plt.plot(fpr,tpr)
# plt.title("ROC Curve")
# plt.show()