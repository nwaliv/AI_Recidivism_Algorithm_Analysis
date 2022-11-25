import numpy as np
import pandas as pd

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

days = []
weeks = []
months = []
years = []

for length in df['length_of_stay']:
    if length<7:
        days.append(1)
        weeks.append(0)
        months.append(0)
        years.append(0)
    elif (length<30):
        days.append(0)
        weeks.append(1)
        months.append(0)
        years.append(0)
    elif length<365:
        days.append(0)
        weeks.append(0)
        months.append(1)
        years.append(0)
    else:
        days.append(0)
        weeks.append(0)
        months.append(0)
        years.append(1)

jail_feature = []
jail_feature_squared = []
for jail in df['length_of_stay']:
    if(jail/365) > 1:
        jail_feature.append(1)
        jail_feature_squared.append((1)**2)
    else:
        jail_feature.append(jail/365)
        jail_feature_squared.append((jail/365)**2)

        
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

mean_age = df_age.mean()
age_feature_squared = []
age_feature = []
for age in df_age:
    if(age/mean_age) > 2:
        age_feature.append(2)
        age_feature_squared.append((2)**2)
    else:
        age_feature.append(age/mean_age)
        age_feature_squared.append((age/mean_age)**2)


# Degree of charge feature
df_c_charge_degree = df['c_charge_degree']
crime_factor, u_charge_degree = pd.factorize(df_c_charge_degree)

# Gender
male = []
female = []
for gender in df['sex']:
    if gender == "Male":
        male.append(1)
        female.append(0)
    else:
        male.append(0)
        female.append(1)


# Prior juvinile convictions
juvinile_felonies  = df['juv_fel_count'].astype(int)
juvinile_misconduct  = df['juv_misd_count'].astype(int)
juvinile_other  = df['juv_other_count'].astype(int)
priors_count  = df['priors_count'].astype(int)

no_prior_convictions = []
one_prior =[]
multiple_prior = []
many_prior = []

# Prior Convictions Feature
for prior in priors_count:
    if prior==0:
        one_prior.append(0)
        multiple_prior.append(0)
        many_prior.append(0)
        no_prior_convictions.append(1)
    elif prior<2:
        one_prior.append(1)
        multiple_prior.append(0)
        many_prior.append(0)
        no_prior_convictions.append(0)
    elif prior<5:
        one_prior.append(0)
        multiple_prior.append(1)
        many_prior.append(0)
        no_prior_convictions.append(0)
    else:
        one_prior.append(0)
        multiple_prior.append(0)
        many_prior.append(1)
        no_prior_convictions.append(0)

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



X = np.column_stack((crime_factor, male, female, no_prior_convictions, one_prior, multiple_prior, many_prior, juvinile_felonies, juvinile_misconduct, juvinile_other, days, weeks, months, years, jail_feature, jail_feature_squared, twenties_and_less, thirties, fourties, fifties_and_more, age_feature, age_feature_squared))

compas_score = []

for score in df['decile_score']:
    if score<5:
        compas_score.append(0)
    else:
        compas_score.append(1)


two_year_recid = df['two_year_recid\r']


x_lables = [ 'Crime factor', 'Male', 'Female', 'One Prior', 'No Priors', '1<Priors<5','Priors>5', 'Juvinile felonies', 'Juvinile misconduct', 'Juvinile other', 'Jail Days', 'Jail Weeks', 'Jail Months', 'Jail Years', 'Jail Feature', 'Jail Feature^2','Twenties and less', 'Thirties', 'Fourties', 'Fifties and more', 'Age Feature', 'Age feature^2']

#------------------------- Model Training -------------------------#

# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, two_year_recid, test_size=0.2, random_state=0)

# print size of train and test
print('-----------------size of train and test-----------------')
print('X_train', len(X_train))
print('X_test', len(X_test))
print('y_train', len(y_train))
print('y_test', len(y_test))

# logistic regression with true class
from sklearn.linear_model import LogisticRegression
model  = LogisticRegression(penalty='l2', C=0.1, max_iter=100)
model.fit(X_train, np.ravel(y_train))
y_pred = model.predict(X_test)

print('-----------------coefficients with corresponding labels -----------------')

coeff_true = model.coef_[0]
for i in range(len(x_lables)):
    print(x_lables[i], '   ', round( model.coef_[0][i], 4))


print('-----------------intercept-----------------')
print(model.intercept_)
print('-----------------score-----------------')
score_true = round(model.score(X_test, y_test),4)
print(score_true)

# logistic regression with predicted class
X_train, X_test, y_train, y_test = train_test_split(X, compas_score, test_size=0.2, random_state=0)
model  = LogisticRegression(penalty='l2', C=0.1, max_iter=100)
model.fit(X_train, np.ravel(y_train))
y_pred = model.predict(X_test)

print('-----------------coefficients with corresponding labels -----------------')
coeff_pred = model.coef_[0]
for i in range(len(x_lables)):
    print(x_lables[i], '   ', round( model.coef_[0][i], 4))
print('-----------------intercept (bias)-----------------')
print(model.intercept_)
print('-----------------score-----------------')
score_pre = round(model.score(X_test, y_test),4)
print(score_pre)


print('coeff_true', np.shape(coeff_true))


C_range =[0.0001, 0.001, 0.005, 0.01, 0.1, 1, 10]
print(C_range)
# knn  with true class
mean_score = []
mean_std = []
f1_score = []
f1_std = []

from sklearn.linear_model import LogisticRegression
for i in C_range:
    log = LogisticRegression(penalty='l2', C=i, solver='lbfgs', max_iter = 500)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(log, X, np.ravel(two_year_recid), cv=5)
    f1 = cross_val_score(log, X, np.ravel(two_year_recid), cv=5, scoring='f1')
    # print('-----------------knn with true class-----------------')
    # print('cross validation scores', scores)
    # print('cross validation mean score', scores.mean())
    mean_score.append(scores.mean())
    mean_std.append(scores.std())
    f1_score.append(f1.mean())
    f1_std.append(f1.std())


C_range_text = ["0.0001", "0.001", "0.005", "0.01", "0.1", "1", "10"]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(C_range_text, mean_score, label='Accuracy')
ax.plot(C_range_text, f1_score, label='F1 score')
ax.errorbar(C_range_text, mean_score, yerr=mean_std, fmt='o', label='Accuracy std')
ax.errorbar(C_range_text, f1_score, yerr=f1_std, fmt='o', label='F1 score std')
ax.set_xlabel('L2 Penalty')
ax.set_ylabel('Score')
ax.set_title('Altering L2 Penalty for Recidivism Rates')
plt.legend()
plt.show()

C_range =[0.0001, 0.001, 0.005, 0.01, 0.1, 1, 10]
print(C_range)
# knn  with true class
mean_score = []
mean_std = []
f1_score = []
f1_std = []

from sklearn.linear_model import LogisticRegression
for i in C_range:
    log = LogisticRegression(penalty='l2', C=i, solver='lbfgs', max_iter = 500)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(log, X, np.ravel(compas_score), cv=5)
    f1 = cross_val_score(log, X, np.ravel(compas_score), cv=5, scoring='f1')
    # print('-----------------knn with true class-----------------')
    # print('cross validation scores', scores)
    # print('cross validation mean score', scores.mean())
    mean_score.append(scores.mean())
    mean_std.append(scores.std())
    f1_score.append(f1.mean())
    f1_std.append(f1.std())


C_range_text = ["0.0001", "0.001", "0.005", "0.01", "0.1", "1", "10"]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(C_range_text, mean_score, label='Accuracy')
ax.plot(C_range_text, f1_score, label='F1 score')
ax.errorbar(C_range_text, mean_score, yerr=mean_std, fmt='o', label='Accuracy std')
ax.errorbar(C_range_text, f1_score, yerr=f1_std, fmt='o', label='F1 score std')
ax.set_xlabel('L2 Penalty')
ax.set_ylabel('Score')
ax.set_title('Altering L2 Penalty for Compas Scores')
plt.legend()
plt.show()

# SVM

mean_score = []
mean_std = []
f1_score = []
f1_std = []

from sklearn.svm import LinearSVC
for i in C_range:
    svm = LinearSVC(C=i, max_iter = 100)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(svm, X, np.ravel(compas_score), cv=5)
    f1 = cross_val_score(svm, X, np.ravel(compas_score), cv=5, scoring='f1')
    # print('-----------------knn with true class-----------------')
    # print('cross validation scores', scores)
    # print('cross validation mean score', scores.mean())
    mean_score.append(scores.mean())
    mean_std.append(scores.std())
    f1_score.append(f1.mean())
    f1_std.append(f1.std())


C_range_text = ["0.0001", "0.001", "0.005", "0.01", "0.1", "1", "10"]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(C_range_text, mean_score, label='Accuracy')
ax.plot(C_range_text, f1_score, label='F1 score')
ax.errorbar(C_range_text, mean_score, yerr=mean_std, fmt='o', label='Accuracy std')
ax.errorbar(C_range_text, f1_score, yerr=f1_std, fmt='o', label='F1 score std')
ax.set_xlabel('L2 Penalty')
ax.set_ylabel('Score')
ax.set_title('Altering L2 Penalty for Compas Scores')
plt.legend()
plt.show()

mean_score = []
mean_std = []
f1_score = []
f1_std = []

from sklearn.svm import LinearSVC
for i in C_range:
    svm = LinearSVC(C=i, max_iter = 100)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(svm, X, np.ravel(two_year_recid), cv=5)
    f1 = cross_val_score(svm, X, np.ravel(two_year_recid), cv=5, scoring='f1')
    # print('-----------------knn with true class-----------------')
    # print('cross validation scores', scores)
    # print('cross validation mean score', scores.mean())
    mean_score.append(scores.mean())
    mean_std.append(scores.std())
    f1_score.append(f1.mean())
    f1_std.append(f1.std())


C_range_text = ["0.0001", "0.001", "0.005", "0.01", "0.1", "1", "10"]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(C_range_text, mean_score, label='Accuracy')
ax.plot(C_range_text, f1_score, label='F1 score')
ax.errorbar(C_range_text, mean_score, yerr=mean_std, fmt='o', label='Accuracy std')
ax.errorbar(C_range_text, f1_score, yerr=f1_std, fmt='o', label='F1 score std')
ax.set_xlabel('L2 Penalty')
ax.set_ylabel('Score')
ax.set_title('Altering L2 Penalty for Recidivism Rates')
plt.legend()
plt.show()





