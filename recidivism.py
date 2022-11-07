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

X = np.column_stack((  AfAmerican, Caucasian, Hispanic, Other, crime_factor, f_gender, priors_count, juvinile_felonies, juvinile_misconduct, juvinile_other, quick_stay, short_stay, medium_stay, long_stay, twenties_and_less, thirties, fourties, fifties_and_more))

#For target class
f_score_text, u_score_text = pd.factorize(df['score_text'] != 'Low')
#decile_score = df['decile_score']
two_year_recid = df[['two_year_recid\r']]


x_lables = [ 'Crime factor', 'Gender factor', 'Priors count', 'Juvinile felonies', 'Juvinile misconduct', 'Juvinile other', 'Quick stay', 'Short stay', 'Medium stay', 'Long stay', 'Twenties and less', 'Thirties', 'Fourties', 'Fifties and more', 'African American', 'Caucasian', 'Hispanic', 'Other Race']



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

coeff_true = abs(model.coef_[0])
for i in range(len(x_lables)):
    print(x_lables[i], '   ', round( model.coef_[0][i], 4))


print('-----------------intercept-----------------')
print(model.intercept_)
print('-----------------score-----------------')
score_true = round(model.score(X_test, y_test),4)
print(score_true)

# logistic regression with predicted class
X_train, X_test, y_train, y_test = train_test_split(X, f_score_text, test_size=0.2, random_state=0)
model  = LogisticRegression(penalty='l2', C=0.1, max_iter=100)
model.fit(X_train, np.ravel(y_train))
y_pred = model.predict(X_test)

print('-----------------coefficients with corresponding labels -----------------')
coeff_pred = abs(model.coef_[0])
for i in range(len(x_lables)):
    print(x_lables[i], '   ', round( model.coef_[0][i], 4))
print('-----------------intercept (bias)-----------------')
print(model.intercept_)
print('-----------------score-----------------')
score_pre = round(model.score(X_test, y_test),4)
print(score_pre)



#-----------------View weigthing of coefficients ----------------


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)

X_axis = np.arange(len(x_lables))

plt.xticks(X_axis, x_lables, rotation=90)
ax.bar(X_axis - 0.2, coeff_true, 0.4, label='True class')
ax.bar(X_axis + 0.2, coeff_pred, 0.4, label='Predicted class')
ax.set_ylabel('Weighting')
ax.set_title('Weighting of factors')
ax.legend(['Trained with true re-offences (Accurcy:%f)' %score_true, 'Trained with compas predictions (Accurcy:%f)' %score_pre])
plt.tight_layout()

plt.show()





# ----------------- To see interesting factors in data ----------------



print('-----------------race split-----------------')
race  = ['African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native American', 'Other']
for i in race :
    print( i,len(df[df['race']== i])/len(df['race']))

# To see spread in re-offence rates
print ('-----------------Likelihood to re-offend----------------')
print('low ', len(df[df['score_text'] == 'Low']))
print('medium ', len(df[df['score_text'] == 'Medium']))
print('high ', len(df[df['score_text'] == 'High']))

print('-----------------Sex spread-----------------')
f = pd.crosstab(df['sex'], df['race'])
print(f)

# find decide score for african american
print('-----------------decile score for african american-----------------')
print(df[(df['race']) == 'African-American']['decile_score'].describe())

decile = [1,2,3,4,5,6,7,8,9,10]
# plot decide score for african american
# import matplotlib.pyplot as plt

# # bar plot for decile score for african american and caucasian
# df_race_decile_score = df[['race', 'decile_score']]
# df_african = df_race_decile_score[ df_race_decile_score['race'] == 'African-American']
# df_caucasian = df_race_decile_score[ df_race_decile_score['race'] == 'Caucasian']
# counts_decile_AA = []
# counts_decile_C = []
# temp = []
# for i in decile:
#     temp = len(df_african[df_african['decile_score'] == i])
#     counts_decile_AA.append(temp)
#     temp = len(df_caucasian[df_caucasian['decile_score'] == i])
#     counts_decile_C.append(temp)

# fig = plt.figure()
# ax = fig.subplots(1,2)
# ax[0].bar(decile, counts_decile_AA)
# ax[0].set_title('African American')
# ax[1].bar(decile, counts_decile_C)
# ax[1].set_title('Caucasian')

# ax[0].set_ylabel('Count')
# ax[0].set_xlabel('Decile score')
# ax[0].set_ylim(0, 650)
# ax[1].set_ylabel('Count')
# ax[1].set_xlabel('Decile score')
# ax[1].set_ylim(0, 650)
# plt.show()



