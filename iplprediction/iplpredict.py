import pandas as pd
import pickle

data = pd.read_csv('dataset/matches.csv').sort_values(by="season")


######clean the data 
columns_remove = ["umpire3","dl_applied","toss_winner","id","date",
                  "umpire1","umpire2",'player_of_match','result','win_by_runs','win_by_wickets']
data.drop(labels=columns_remove,axis =1,inplace=True)

#removeing all nan data
for i in list(data):
    data = data[data[i].notna()]


#teams filter
allteams = data['team1'].unique()
teams_filter= ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Capitals', 'Sunrisers Hyderabad']

data = data[(data['team1'].isin(teams_filter))&(data['team2'].isin(teams_filter))]


######data preprocesing 

#converting results or team names into numbs 
data['winner'] = data['winner'].apply(lambda x:teams_filter.index(x))

#comvert numbers to results or team names
retrivedata = data['winner'].apply(lambda x:teams_filter[x])

#convert to one hot encoding 
#give colums to do one hot encoding
encoded_data = pd.get_dummies(data=data,columns=['team1','team2','city','venue',
                                                 'toss_decision'])

#test train split 

#for time series analysis we take previous years and predict futer
tempdata = encoded_data[encoded_data['season']<2018]
xtrain = tempdata.drop(labels=['winner'],axis =1)
ytrain =tempdata[['winner']]

tempdata1 = encoded_data[encoded_data['season']>=2018]
xtest = tempdata1.drop(labels=['winner'],axis =1)
ytest =tempdata1[['winner']]


##removing the season because we are using it only to split test train
xtest.drop(labels='season',axis=1,inplace=True)
xtrain.drop(labels='season',axis=1,inplace=True)

'''

##standradise
from sklearn.preprocessing import StandardScaler

x = StandardScaler()
x.fit(xtrain)

xtrain = x.transform(xtrain)
xtest = x.transform(xtest)
'''

##############support vector machine 
from sklearn import svm

clf = svm.SVC()

clf.fit(xtrain,ytrain)

preddata = clf.predict(xtest)

from sklearn.metrics import accuracy_score

print("accuracy ==",accuracy_score(ytest,preddata))


predprocesseddata = [teams_filter[i] for i in preddata]
realprocesseddata = [teams_filter[i] for i in ytest['winner']]

#custom data predict

#Dubai International Cricket Stadium, Dubai 

#taking single row
customdata = xtest.iloc[[1]]
#making everythng to zero
for i in list(customdata):
    customdata[i] =0

#selecting parameters
customdata['team1_Sunrisers Hyderabad']=1
customdata['team2_Royal Challengers Bangalore'] = 1
customdata['toss_decision_field'] =1

#result
newpred = teams_filter[clf.predict(customdata)[0]]

import pymsgbox

pymsgbox.alert(newpred+'\n\naccuracy: '+str(accuracy_score(ytest,preddata)*100)+"%","next ipl match winner")
