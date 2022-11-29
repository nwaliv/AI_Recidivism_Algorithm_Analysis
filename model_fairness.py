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
#print first 10 lines of df_c_charge_degree
print('charge_degree',df_c_charge_degree.head(10))
df_c_charge_degree_male = df_c_charge_degree[(df_c_charge_degree['sex']== 'Female')]
#print first 10 lines of df_c_charge_degree_male
print('charge_degree male',df_c_charge_degree_male.head(10))
# crime_factor, u_charge_degree = pd.factorize(df_c_charge_degree['c_charge_degree'])

# # Gender
# df_sex = df[['sex']]
# f_gender, uniques_gender  = pd.factorize(df_sex['sex'])

# # Prior convictions
# juvinile_felonies  = df[['juv_fel_count']].astype(int)
# juvinile_misconduct  = df[['juv_misd_count']].astype(int)
# juvinile_other  = df[['juv_other_count']].astype(int)
# priors_count  = df[['priors_count']].astype(int)

# #Race
# AfAmerican =[]
# Other = []
# Caucasian = []
# Hispanic = []

# for race in df['race']:
#     if race == 'African-American':
#         AfAmerican.append(1)
#         Caucasian.append(0)
#         Hispanic.append(0)
#         Other.append(0)
#     elif race == 'Caucasian':
#         AfAmerican.append(0)
#         Caucasian.append(1)
#         Hispanic.append(0)
#         Other.append(0)
#     elif race == 'Hispanic':
#         AfAmerican.append(0)
#         Caucasian.append(0)
#         Hispanic.append(1)
#         Other.append(0)
#     else:
#         AfAmerican.append(0)
#         Caucasian.append(0)
#         Hispanic.append(0)
#         Other.append(1)




# X = np.column_stack(( race, crime_factor, f_gender, priors_count, juvinile_felonies, juvinile_misconduct, juvinile_other, quick_stay, short_stay, medium_stay, long_stay, twenties_and_less, thirties, fourties, fifties_and_more))

# #For target class
# f_score_text, u_score_text = pd.factorize(df['score_text'] != 'Low')
# #decile_score = df['decile_score']
# two_year_recid = df[['two_year_recid\r']]


# x_lables = [ 'Crime factor', 'Gender factor', 'Priors count', 'Juvinile felonies', 'Juvinile misconduct', 'Juvinile other', 'Quick stay', 'Short stay', 'Medium stay', 'Long stay', 'Twenties and less', 'Thirties', 'Fourties', 'Fifties and more', 'African American', 'Caucasian', 'Hispanic', 'Other Race']


# def equilizing_odds(C):
#     # #convert to probability
#     # C = (1/len(F_true_score))*C
#     # print(C)

#     # False postive rates :Pr[ ˆY = 1/S = 1, Y = 0] − Pr[ ˆY = 0/S = 0, Y = 0]
#     # False negative rates :Pr[ ˆY = 1/S = 1, Y = 1] − Pr[ ˆY = 0/S = 0, Y = 1]
    
#     FNR = abs(C[0][0,1]/(C[0][0,1]+C[0][0,0]) - C[1][0,1]/(C[1][0,1]+C[1][0,0]))
#     FPR = abs(C[0][1,0]/(C[0][1,0]+C[0][1,1]) - C[1][1,0]/(C[1][1,0]+C[1][1,1]))

#     result = [FPR, FNR]

#     return result



# from sklearn.metrics import confusion_matrix
# #C = confusion_matrix(F_true_score, F_Pred_score), confusion_matrix(M_true_score, M_Pred_score)

# #------ split into train and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, f_score_text, test_size=0.2, random_state=42)


# #for different classifiers find the equilizing odds
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

# #Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)


# /#C = confusion_matrix(F_true_score, F_Pred_score), confusion_matrix(M_true_score, M_Pred_score)
# C = confusion_matrix(y_test, y_pred)
# print('Logistic Regression')
# print(equilizing_odds(C))


    



# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.subplots(1,1)
# ax.errorbar(N_range, mean_score_recid_accurcy, yerr=std_score_recid_accurcy, label='Accuracy')
# ax.errorbar(N_range, mean_score_recid_f1, yerr=std_score_recid_f1, label='F1')
# ax.set_xlabel('K')
# ax.set_ylabel('Score')
# ax.set_title('KNN trained on recidivism score')
# ax.legend(['Accuracy', 'F1'], loc='upper right')
# plt.show()








