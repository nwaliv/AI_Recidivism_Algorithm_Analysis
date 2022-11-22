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
length_factor, u_length_degree = pd.factorize(df['length_of_stay'])

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

one_prior =[]
multiple_prior = []
many_prior = []
# Prior Convictions Feature
for prior in priors_count:
    if prior < 2:
        one_prior.append(1)
        multiple_prior.append(0)
        many_prior.append(0)
    elif prior < 5:
        one_prior.append(0)
        multiple_prior.append(1)
        many_prior.append(0)
    else:
        one_prior.append(0)
        multiple_prior.append(0)
        many_prior.append(1)


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

X = np.column_stack((  AfAmerican, Caucasian, Hispanic, Other, crime_factor, f_gender, one_prior, multiple_prior, many_prior, juvinile_felonies, juvinile_misconduct, juvinile_other, days, weeks, months, years, twenties_and_less, thirties, fourties, fifties_and_more))


#Change this!!!
#For target class
f_score_text, u_score_text = pd.factorize(df['score_text'] != 'Low')
#decile_score = df['decile_score']
two_year_recid = df[['two_year_recid\r']]


x_lables = [ 'Crime factor', 'Gender factor', 'One Prior', '1<Priors<5','Priors>5', 'Juvinile felonies', 'Juvinile misconduct', 'Juvinile other', 'Jail Days', 'Jail Weeks', 'Jail Months', 'Jail Years', 'Twenties and less', 'Thirties', 'Fourties', 'Fifties and more', 'African American', 'Caucasian', 'Hispanic', 'Other Race']





#------ cross validation ------


# N_range = np.arange(1, 50)
# print(N_range)
# # knn  with true class
# mean_score = []
# from sklearn.neighbors import KNeighborsClassifier
# for i in N_range:
#     knn = KNeighborsClassifier(n_neighbors=i)
    
#     from sklearn.model_selection import cross_val_score
#     scores = cross_val_score(knn, X, np.ravel(two_year_recid), cv=5)
#     # print('-----------------knn with true class-----------------')
#     # print('cross validation scores', scores)
#     # print('cross validation mean score', scores.mean())

#     mean_score.append(scores.mean())



# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.subplots(1,1)
# ax.plot(N_range, mean_score)
# ax.set_xlabel('N')
# ax.set_ylabel('Mean score')
# ax.set_title('Mean score for different N')
# plt.show()

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
ax.set_xlabel('L1 Penalty')
ax.set_ylabel('Score')
ax.set_title('Altering L1 Penalty for Recidivism Rates')
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
    scores = cross_val_score(log, X, np.ravel(f_score_text), cv=5)
    f1 = cross_val_score(log, X, np.ravel(f_score_text), cv=5, scoring='f1')
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
ax.set_xlabel('L1 Penalty')
ax.set_ylabel('Score')
ax.set_title('Altering L1 Penalty for Compas Scores')
plt.legend()
plt.show()

# from sklearn.svm import LinearSVC
# for i in C_range:
#     svm = LinearSVC(C=i, max_iter = 500)
#     from sklearn.model_selection import cross_val_score
#     scores = cross_val_score(svm, X, np.ravel(f_score_text), cv=5)
#     f1 = cross_val_score(svm, X, np.ravel(f_score_text), cv=5, scoring='f1')
#     # print('-----------------knn with true class-----------------')
#     # print('cross validation scores', scores)
#     # print('cross validation mean score', scores.mean())
#     mean_score.append(scores.mean())
#     mean_std.append(scores.std())
#     f1_score.append(f1.mean())
#     f1_std.append(f1.std())


# C_range_text = ["0.0001", "0.001", "0.005", "0.01", "0.1", "1", "10"]
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.subplots(1,1)
# ax.plot(C_range_text, mean_score, label='Accuracy')
# ax.plot(C_range_text, f1_score, label='F1 score')
# ax.errorbar(C_range_text, mean_score, yerr=mean_std, fmt='o', label='Accuracy std')
# ax.errorbar(C_range_text, f1_score, yerr=f1_std, fmt='o', label='F1 score std')
# ax.set_xlabel('L1 Penalty')
# ax.set_ylabel('Score')
# ax.set_title('Altering L1 Penalty for Compas Scores')
# plt.legend()
# plt.show()





