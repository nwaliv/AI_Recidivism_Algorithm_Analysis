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
df_c_charge_degree = df['c_charge_degree']
crime_factor, u_charge_degree = pd.factorize(df_c_charge_degree)

male = []
female = []
# Gender
for gender in df['sex']:
    if gender == "Male":
        male.append(1)
        female.append(0)
    else:
        male.append(0)
        female.append(1)


# Prior convictions
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
    

jail_feature = []
jail_feature_squared = []
for jail in df['length_of_stay']:
    if(jail/365) > 1:
        jail_feature.append(1)
        jail_feature_squared.append((1)**2)
    else:
        jail_feature.append(jail/365)
        jail_feature_squared.append((jail/365)**2)


X = np.column_stack((crime_factor, male, female, no_prior_convictions, one_prior, multiple_prior, many_prior, juvinile_felonies, juvinile_misconduct, juvinile_other, days, weeks, months, years, jail_feature, jail_feature_squared, twenties_and_less, thirties, fourties, fifties_and_more, age_feature, age_feature_squared, AfAmerican, Caucasian, Hispanic, Other))

compas_score = []

for score in df['decile_score']:
    if score<5:
        compas_score.append(0)
    else:
        compas_score.append(1)


two_year_recid = df['two_year_recid\r']


x_lables = [ 'Crime factor', 'Male', 'Female', 'One Prior', 'No Priors', '1<Priors<5','Priors>5', 'Juvinile felonies', 'Juvinile misconduct', 'Juvinile other', 'Jail Days', 'Jail Weeks', 'Jail Months', 'Jail Years', 'Jail Feature', 'Jail Feature^2','Twenties and less', 'Thirties', 'Fourties', 'Fifties and more', 'Age Feature', 'Age feature^2', 'African American', 'Caucasian', 'Hispanic', 'Other Race']

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

#-----------------View weigthing of coefficients ----------------


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)

X_axis = np.arange(len(x_lables))

plt.xticks(X_axis, x_lables, rotation=90)
ax.bar(X_axis - 0.2, coeff_true, 0.4, label='True class')
ax.bar(X_axis + 0.2, coeff_pred, 0.4, label='Predicted class')
ax.set_ylabel('Weighting')
# ax.set_title('Weighting of factors')
ax.legend(['Trained with True Recid', 'Trained with COMPAS Scores'])
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




# logistic regression with predicted class
X_train, X_test, y_train, y_test = train_test_split(X, compas_score, test_size=0.2, random_state=0)
model  = LogisticRegression(penalty='l2', C=0.1, max_iter=100)
model.fit(X, np.ravel(two_year_recid))
y_pred_train = model.predict(X)

age_and_jail = np.column_stack((age_feature, jail_feature))
y = df.iloc[:,11]
print(y)

# Plot the predicted vs actual results
plt.rc('font', size=10)
print('Accuracy : {:.2f}'.format(model.score(X, two_year_recid)))
plt.scatter(age_and_jail[y_pred_train == 1, 0], age_and_jail[y_pred_train == 1, 1], marker='o', s=75, color = 'k', label='Predicted Recid')
plt.scatter(age_and_jail[y_pred_train == 0, 0], age_and_jail[y_pred_train == 0, 1], marker='o', s=75, color = 'r', label='Predicted No Recid')
plt.scatter(age_and_jail[y == 1, 0], age_and_jail[y == 1, 1], marker='o', s=25, color = 'm', label='True Recid')
plt.scatter(age_and_jail[y == 0, 0], age_and_jail[y == 0, 1], marker='o', s=25, color = 'c', label='True No Recid')
plt.legend()
plt.xlabel("Age Feature")
plt.ylabel("Jail Feature")
plt.title("Predicted Values VS True Values, Accuracy : {:.2f}".format(model.score(X, two_year_recid)))
plt.show()

