import numpy as np
import pandas as pd

compas_scores_two_year= pd.read_csv("compas_scores_two_years.csv",  lineterminator='\n')

# fitlering the data with the following conditions
# 1. if charge data was not within 30 days of arrest
# 2. c_charge_degree is not missing
# 3. score_text is not missing
# 4. is_recid is not missing -1 means missingd

df= compas_scores_two_year[[ 'juv_fel_count', 'juv_misd_count', 'juv_other_count' ,'age', 'c_charge_degree','race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',  'c_jail_in', 'c_jail_out',  'v_decile_score','two_year_recid\r']]
df = df.loc[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30) & (df['is_recid'] != -1) & (df['c_charge_degree'] != 'O') & (df['score_text'] != 'N/A')]

#length of stay in jail
df['length_of_stay'] = pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])
df['length_of_stay'] = df['length_of_stay'].astype('timedelta64[D]')
df['length_of_stay'] = df['length_of_stay'].astype(int)
print(df['length_of_stay'])
#correlation between length of stay and decile score
print('correlation between length of stay and decile score',df['length_of_stay'].corr(df['decile_score']))

print('-----------------race split-----------------')
race  = ['African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native American', 'Other']
for i in race :
    print( i,len(df[df['race']== i])/len(df['race']))

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
import matplotlib.pyplot as plt

# bar plot for decile score for african american and caucasian
df_race_decile_score = df[['race', 'decile_score']]
df_african = df_race_decile_score[ df_race_decile_score['race'] == 'African-American']
df_caucasian = df_race_decile_score[ df_race_decile_score['race'] == 'Caucasian']
counts_decile_AA = []
counts_decile_C = []
temp = []
for i in decile:
    temp = len(df_african[df_african['decile_score'] == i])
    counts_decile_AA.append(temp)
    temp = len(df_caucasian[df_caucasian['decile_score'] == i])
    counts_decile_C.append(temp)

fig = plt.figure()
ax = fig.subplots(1,2)
ax[0].bar(decile, counts_decile_AA)
ax[0].set_title('African American')
ax[1].bar(decile, counts_decile_C)
ax[1].set_title('Caucasian')

ax[0].set_ylabel('Count')
ax[0].set_xlabel('Decile score')
ax[0].set_ylim(0, 650)
ax[1].set_ylabel('Count')
ax[1].set_xlabel('Decile score')
ax[1].set_ylim(0, 650)
plt.show()

# create some factors for logistic regression
df_c_charge_degree = df[['c_charge_degree']]
df_age_cat = df[['age_cat']]
df_race = df[['race']]
df_sex = df[['sex']]
df_age_race = df[['race']]
df_score = df[['score_text']]


length_factor, u_length_degree = pd.factorize(df['length_of_stay'])
#Change charge degree into int values
crime_factor, u_charge_degree = pd.factorize(df_c_charge_degree['c_charge_degree'])
#Change cage categories into int values
f_age_cat, u_age_cat= pd.factorize(df_age_cat['age_cat'])
#relevel age cat with reference = 1
f_age_cat = f_age_cat - 1
#Change gender to numerical values
f_gender, uniques_gender  = pd.factorize(df_sex['sex'])
# create a new maxtrix with the factors
juvinile_felonies  = df[['juv_fel_count']]
juvinile_misconduct  = df[['juv_misd_count']]
juvinile_other  = df[['juv_other_count']]
priors_count  = df[['priors_count']]
two_year_recid = df[['two_year_recid\r']]
decile_score = df['decile_score']

#factorize race TO BE DISCUSSED
# f_race_AA, u_race_AA= pd.factorize(df_age_race['race'] == 'African-American')
# f_race_C, u_race = pd.factorize(df_age_race['race'] == 'Caucasian')
#relevel race with reference = 3
# print('----------------race----------------')
# print("Numeric Representation : \n", f_race_AA)
# print("Unique Values : \n", u_race_AA)


X = np.column_stack(( f_age_cat, crime_factor, f_gender, priors_count, juvinile_felonies, juvinile_misconduct, juvinile_other, length_factor))
# build a binmal logistic regression model to explain the score text given the factors
from sklearn.linear_model import LogisticRegression
model  = LogisticRegression(penalty='l2', C=0.1, max_iter=100)
model.fit(X, two_year_recid)
ypred = model.predict(X)
#print summary
print('intercept', model.intercept_)
#print coefficients with corresponding factors
print('coefficients', model.coef_)
print('Accuracy', model.score(X, two_year_recid))
