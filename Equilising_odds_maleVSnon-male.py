from calendar import c
from pyexpat import model
from re import L
from turtle import color
import numpy as np
import pandas as pd



compas_scores_raw= pd.read_csv("compas_score_raw.csv",  lineterminator='\n')
compas_scores_two_year= pd.read_csv("compas_scores_two_years.csv",  lineterminator='\n')
print('-----------------Compas Scores Raw-----------------')
print('type',type(compas_scores_raw))
print('-----------------Compas Scores Two Year-----------------')
print('type',type(compas_scores_two_year))

#number of rows and columns
print('-----------------Compas Scores Raw-----------------')
print('shape',compas_scores_raw.shape)
print('-----------------Compas Scores Two Year-----------------')
print('shape',compas_scores_two_year.shape)




# fitlering the data with the following conditions
# 1. if charge data was not within 30 days of arrest
# 2. c_charge_degree is not missing
# 3. score_text is not missingthe 
# 4. is_recid is not missing -1 means missing

print('-----------------Compas Scores two year-----------------')
df= compas_scores_two_year[[ 'age', 'c_charge_degree','race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',  'c_jail_in', 'c_jail_out',  'v_decile_score','two_year_recid\r']]
print(np.shape(df))

df = df.loc[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30) & (df['is_recid'] != -1) & (df['c_charge_degree'] != 'O') & (df['score_text'] != 'N/A')]
print('shape of filtered data',df.shape)

#length of stay in jail
df['length_of_stay'] = pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])
df['length_of_stay'] = df['length_of_stay'].astype('timedelta64[D]')
df['length_of_stay'] = df['length_of_stay'].astype(int)
print(df['length_of_stay'])
print('shape of filtered data',df.shape)
print('length of stay',df['length_of_stay'].describe())
#correlation between length of stay and decile score
print('correlation between length of stay and decile score',df['length_of_stay'].corr(df['decile_score']))

print('-----------------describe age-----------------')
print(df['age'].describe())
print('-----------------describe race----------------')
print(df['race'].describe())

print('-----------------race split-----------------')
race  = ['African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native American', 'Other']
num_race = []
temp = []
for i in race :
    print( i,len(df[df['race']== i])/len(df['race']))
    temp =len(df[df['race']== i])
    num_race.append(temp)


print('----------------age category split-----------------')
age_cat = ['25 - 45', 'Greater than 45', 'Less than 25']
num_age_cat = []
temp = []
for i in age_cat :
    print( i,len(df[df['age_cat']== i])/len(df['age_cat']))
    temp =len(df[df['age_cat']== i])
    num_age_cat.append(temp)

print(age_cat[0], num_age_cat[0])
print(age_cat[1], num_age_cat[1])
print(age_cat[2], num_age_cat[2])




#------bar plot for women recidivism vs prediction------
#---------------------------------------------------------

#extract the women
female_df = df[(df['sex']== 'Female') ]
male_df = df[(df['sex']== 'Male') ]
#create an array for women predicted score
female_pred_score_df = female_df[['score_text']]
male_pred_score_df = male_df[['score_text']]

#factorize the score_text with low as 0 and high or mediam as 1
F_Pred_score, u_score_text = pd.factorize(female_pred_score_df['score_text'] != 'Low')
M_Pred_score, u_score_text = pd.factorize(male_pred_score_df['score_text'] != 'Low')


# --- find true
#extract the women from two year recid
female_recid_df = female_df[['two_year_recid\r']]
male_recid_df = male_df[['two_year_recid\r']]

#factorize the two year recid with 0 as no and 1 as yes
F_true_score, u_true_score = pd.factorize(female_recid_df['two_year_recid\r'])
M_true_score, u_true_score = pd.factorize(male_recid_df['two_year_recid\r'])


from sklearn.metrics import confusion_matrix
C = confusion_matrix(F_true_score, F_Pred_score), confusion_matrix(M_true_score, M_Pred_score)
print('----------confusion matrix female-------')
print(confusion_matrix(F_true_score, F_Pred_score))
print('----------confusion matrix male-------')
print(confusion_matrix(M_true_score, M_Pred_score))


def equilizing_odds(C):
    # #convert to probability
    # C = (1/len(F_true_score))*C
    # print(C)

    # False postive rates :Pr[ ˆY = 1/S = 1, Y = 0] − Pr[ ˆY = 0/S = 1, Y = 0]
    # False negative rates :Pr[ ˆY = 1/S = 0, Y = 1] − Pr[ ˆY = 0/S = 0, Y = 1]
    
    FNR = abs(C[0][0,1]/(C[0][0,1]+C[0][0,0]) - C[1][0,1]/(C[1][0,1]+C[1][0,0]))
    FPR = abs(C[0][1,0]/(C[0][1,0]+C[0][1,1]) - C[1][1,0]/(C[1][1,0]+C[1][1,1]))

    result = [FPR, FNR]

    return result



def KLDivergence(P, Q):
    return np.sum(np.where(P != 0, P * np.log(P / Q), 0))


print('-----------------Equilizing_odds -----------------')
print('False postive rates |Pr[ Y^ = 1/S = 1, Y = 0] − Pr[ Y^ = 0/S = 0, Y = 0]|  = ',   equilizing_odds(C)[0])
print('False negative rates  |Pr[ Y^ = 1/S = 1, Y = 1] − Pr[ Y^ = 0/S = 0, Y = 1]| = ', equilizing_odds(C)[1])







#--------------- hist of male vs female for prediction of recidivism and non recidivism


# convert to counts and normalize

#true
female_score_true= np.array([np.count_nonzero(F_true_score == 1), np.count_nonzero(F_true_score == 0)])
female_score_true = (1/len(F_true_score))*female_score_true
male_score_true= np.array([np.count_nonzero(M_true_score == 1), np.count_nonzero(M_true_score == 0)])
male_score_true = (1/len(M_true_score))*male_score_true

#predicted
female_score_pred = np.array([np.count_nonzero(F_Pred_score == 1), np.count_nonzero(F_Pred_score == 0)])
female_score_pred = (1/len(F_Pred_score))*female_score_pred
male_score_pred= np.array([np.count_nonzero(M_Pred_score == 1), np.count_nonzero(M_Pred_score == 0)])
male_score_pred = (1/len(M_Pred_score))*male_score_pred


# female score low and high
female_score_low= np.array([np.count_nonzero(F_Pred_score == 0), np.count_nonzero(F_true_score == 0)])*(1/len(F_Pred_score))
female_score_high= np.array([np.count_nonzero(F_Pred_score == 1), np.count_nonzero(F_true_score == 1)])*(1/len(F_Pred_score))
#male score low and high
male_score_low= np.array([np.count_nonzero(M_Pred_score == 0), np.count_nonzero(M_true_score == 0)])*(1/len(M_Pred_score))
male_score_high= np.array([np.count_nonzero(M_Pred_score == 1), np.count_nonzero(M_true_score == 1)])*(1/len(M_Pred_score))

#kld 
print('-----------------KLDivergence women-----------------')
print('KLD', KLDivergence(female_score_true, female_score_pred))
print('female score true', female_score_true)
print('famele score pred',female_score_pred )
print('-----------------KLDivergence men-----------------')
print('KLD', KLDivergence(male_score_true, male_score_pred))
print('male score true', male_score_true)
print('mele score pred',male_score_pred )





import matplotlib.pyplot as plt


x_labels = ['predicted' ,'actual' ]
X_axis = np.arange(len(x_labels))

fig, ax= plt.subplots(nrows=1, ncols=2)

# Set the ticks and ticklabels for all axes
plt.setp(ax, xticks=X_axis, xticklabels=x_labels)


ax[0].bar(X_axis - 0.2, female_score_low, 0.4, label='Female score low')
ax[0].bar(X_axis + 0.2, female_score_high, 0.4, label='female score high')
lengend = [KLDivergence(female_score_true, female_score_pred), equilizing_odds(C)[0], equilizing_odds(C)[1]]
#ax[0].legend(['KLD = %f, FPR = %f, FNR = %f' % (lengend[0], lengend[1], lengend[2])])
ax[0].set_title('Non-male KLD = %f' % (KLDivergence(female_score_true, female_score_pred)))
ax[0].set_ylabel('Probability')
ax[0].set_ylim( 0, 1)

plt.xticks(X_axis, x_labels, rotation=0)
ax[1].set_title('Male: KLD = %f' %(KLDivergence(male_score_true, male_score_pred)))
ax[1].bar(X_axis - 0.2, male_score_low, 0.4, label='male score low')
ax[1].bar(X_axis + 0.2, male_score_high, 0.4, label='male score high')
ax[1].set_ylim( 0, 1)
plt.tight_layout()
plt.show()



