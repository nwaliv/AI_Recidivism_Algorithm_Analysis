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





#------ cross validation ------


N_range = np.arange(1, 20)
print(N_range)
# knn  with true class
mean_score_recid_accurcy= []
mean_score_recid_f1= []

std_score_recid_accurcy = []
std_score_recid_f1 = []

from sklearn.neighbors import KNeighborsClassifier
for i in N_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    
    from sklearn.model_selection import cross_val_score
    scores_recid_accurcy = cross_val_score(knn, X, np.ravel(two_year_recid), cv=5, scoring='accuracy')
    scores_recid_f1 = cross_val_score(knn, X, np.ravel(f_score_text), cv=5, scoring='f1')

    # print('-----------------knn with true class-----------------')
    # print('cross validation scores', scores)
    # print('cross validation mean score', scores.mean())
    
    mean_score_recid_accurcy.append(scores_recid_accurcy.mean())
    std_score_recid_accurcy.append(scores_recid_accurcy.std())
    mean_score_recid_f1.append(scores_recid_f1.mean())
    std_score_recid_f1.append(scores_recid_f1.std())

   

    



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots(1,1)
ax.errorbar(N_range, mean_score_recid_accurcy, yerr=std_score_recid_accurcy, label='Accuracy')
ax.errorbar(N_range, mean_score_recid_f1, yerr=std_score_recid_f1, label='F1')
ax.set_xlabel('K')
ax.set_ylabel('Score')
ax.set_title('KNN trained on recidivism score')
ax.legend(['Accuracy', 'F1'], loc='upper right')
plt.show()








