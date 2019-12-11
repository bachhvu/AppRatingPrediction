import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

df = pd.read_csv('C:/Users/Admin/Downloads/google.csv')

df.info()

#Pre-process by removing all NaN values
df.dropna(inplace = True)

#Using label-encoder
le = preprocessing.LabelEncoder()
#Converting the category into binary
df["Category_c"] = le.fit_transform(df['Category'])

#Converting Type classification into binary6
df["Type_c"] = le.fit_transform(df['Type'])

#Converting of the content rating section into integers. 
#In this specific instance, given that the concent rating is somewhat relatable and has an order to it,
#we do not use one-hot encoding.

#Cleaning of content rating classification
df["Content Rating_c"] = le.fit_transform(df['Content Rating'])

#Cleaning of genres

df['Genres_c'] = le.fit_transform(df['Genres'])



#Cleaning of sizes of the apps and also filling up the missing values using ffill
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)
        return(x)
    else:
        return None

df["Size"] = df["Size"].map(change_size)

#filling Size which had NA
df.Size.fillna(method = 'ffill', inplace = True)

# Cleaning number of installs classification
df['Installs'] = [int(i[:-1].replace(',','')) for i in df['Installs']]


#I dropped these portions of information as i deemed it unecessary for our machine learning algorithm
df.drop(labels = ['Last Updated','Current Ver','Android Ver','App'], axis = 1, inplace = True)


#Cleaning of the prices of the apps to floats
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price
df['Price'] = df['Price'].map(price_clean).astype(float)

#drop price larger than 100
df = df[df.Price <100]

# Finally converting the number reviews column into integers
df['Reviews'] = df['Reviews'].astype(int)

#In this instance, I created another dataframe that specifically created dummy values 
#for each categorical instance in the dataframe, defined as df2
# for dummy variable encoding for Categories
df2 = pd.get_dummies(df, columns=['Category'])

fig, ax = plt.subplots()
d1 = df.loc[df['Type'] == 'Free']
d2 = df.loc[df['Type'] == 'Paid']
ax.boxplot([d1['Rating'],d2['Rating']])
plt.show()

var = df.groupby('Category').Rating.count()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('#App')
ax.set_ylabel('Category')
ax.set_title("Market distribution")
var.plot(kind='barh')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set(xlabel = 'Reviews', ylabel = 'Installs', xscale="log", yscale="log")
sns.regplot(x=df["Reviews"], y=df["Installs"])
plt.show()


fig = plt.figure()
plt.scatter(d2['Rating'], d2['Price'], c=d2['Size'], cmap='jet')
plt.xlabel('Rating')
plt.ylabel('Price ($)')
plt.colorbar()
plt.show()

var = d2.groupby('Category').Price.mean()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Price ($)')
ax.set_ylabel('Category')
ax.set_title("Average price for category")
var.plot(kind='barh')
plt.show()

var = df.groupby('Category').Size.mean()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Size (KB)')
ax.set_ylabel('Category')
ax.set_title("Average size of category")
var.plot(kind='barh')
plt.show()


fig = plt.figure()
plt.scatter(df['Size'], df['Category'], c=df['Rating'], cmap='jet')
plt.colorbar()
plt.show()

fig = plt.figure()
var=df.groupby(['Type']).sum().stack()
temp=var.unstack()
type(temp)
x_list = temp['Rating']
label_list = temp.index
plt.axis("equal") 
plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 
plt.show()

fig = plt.figure()
plt.scatter(df['Rating'],df['Size'])
plt.xlabel('Rating')
plt.ylabel('Size (KB)')
plt.show()

var = df.groupby('Rating').Rating.count()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Rating')
ax1.set_ylabel('#App')
ax1.set_title("Rating wise #App")
var.plot(kind='bar')

    
var = df.groupby('Rating').Size.mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Rating')
ax1.set_ylabel('Average Size (in KB)')
ax1.set_title("Rating wise Average Size")
var.plot(kind='line')

var = df.groupby('Rating').Installs.mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Rating')
ax1.set_ylabel('Sum of Installs')
ax1.set_title("Rating wise Average number of Installs")
var.plot(kind='bar')


var = df.groupby('Rating').Reviews.mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Rating')
ax1.set_ylabel('Average number of Reviews')
ax1.set_title("Rating wise Average number of Reviews")
var.plot(kind='bar')

var = df.groupby('Content Rating').Rating.count()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('#App')
ax.set_ylabel('Category')
ax.set_title("Market distribution")
var.plot(kind='barh')


#for evaluation of error term and 
def Evaluationmatrix(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))
#to add into results_index for evaluation of error term 
def Evaluationmatrix_dict(y_true, y_predict, name = 'RFR - Integer'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
    return dict_matrix


#Predicting correlation - these can prove to be important features to predict app rating
corr=df.corr()["Rating"]
corr[np.argsort(corr, axis=0)[::-1]]

num_feat=df.columns[df.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(df[col].values, df.Rating.values)[0,1])
fig=plt.figure() 
ax = fig.add_subplot(1,1,1)
rects = ax.barh(labels, np.array(values), color='red')

corrMatrix = df.corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');



from sklearn.ensemble import RandomForestRegressor

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres','Genres_c','Content Rating', 'Type' ],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = RandomForestRegressor()
model.fit(X_train,y_train)
Results = model.predict(X_test)

#evaluation
resultsdf = pd.DataFrame()
resultsdf = resultsdf.from_dict(Evaluationmatrix_dict(y_test,Results),orient = 'index')
resultsdf = resultsdf.transpose()

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Genres_c','Content Rating', 'Type', 'Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = RandomForestRegressor()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results_d, name = 'RFR - Dummy'),ignore_index = True)

plt.figure(figsize=(12,7))
sns.regplot(Results,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('RFR model - excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

print ('Integer encoding(mean) :' + str(Results.mean()))
print ('Dummy encoding(mean) :'+ str(Results_d.mean()))
print ('Integer encoding(std) :' + str(Results.std()))
print ('Dummy encoding(std) :'+ str(Results_d.std()))

#for integer
feats = {} # a dictionary to hold feature_name: feature_importance
for feature,importance in zip(X.columns,model.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importances.sort_values(by='Importance').plot(kind='bar', rot=45)

#for dummy
feats_d = {} # a dictionary to hold feature_name: feature_importance
for feature,importance in zip(X_d.columns,model_d.feature_importances_):
    feats_d[feature] = importance

importances = pd.DataFrame.from_dict(feats_d, orient='index').rename(columns={0: 'Importances'})
importances.sort_values(by='Importances').plot(kind='barh')

#Including Genres_C

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres','Content Rating', 'Type'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
modela = RandomForestRegressor()
modela.fit(X_train,y_train)
Resultsa = modela.predict(X_test)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Resultsa, name = 'RFR(inc Genres) - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Content Rating', 'Type', 'Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
modela_d = RandomForestRegressor()
modela_d.fit(X_train_d,y_train_d)
Resultsa_d = modela_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Resultsa_d, name = 'RFR(inc Genres) - Dummy'),ignore_index = True)

plt.figure(figsize=(12,7))
sns.regplot(Resultsa,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Resultsa_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('RFR model - including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

print ('Integer encoding(mean) :' + str(Resultsa.mean()))
print ('Dummy encoding(mean) :'+ str(Resultsa_d.mean()))
print ('Integer encoding(std) :' + str(Resultsa.std()))
print ('Dummy encoding(std) :'+ str(Resultsa_d.std()))

#for integer
feats = {} # a dictionary to hold feature_name: feature_importance
for feature,importance in zip(X.columns,modela.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importances.sort_values(by='Importance').plot(kind='barh',)

#for dummy
feats_d = {} # a dictionary to hold feature_name: feature_importance
for feature,importance in zip(X_d.columns,modela_d.feature_importances_):
    feats_d[feature] = importance

importances = pd.DataFrame.from_dict(feats_d, orient='index').rename(columns={0: 'Importances'})
importances.sort_values(by='Importances').plot(kind='barh')






#excluding Genre label
from sklearn.linear_model import LinearRegression 

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres','Genres_c', 'Type', 'Content Rating'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
modell = LinearRegression()
modell.fit(X_train,y_train)
Resultsl = modell.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Resultsl, name = 'Linear - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c','Type', 'Content Rating'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
modell_d = LinearRegression()
modell_d.fit(X_train_d,y_train_d)
Resultsl_d = modell_d.predict(X_test_d)

#adding results into results dataframe
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Resultsl_d, name = 'Linear - Dummy'),ignore_index = True)

print ('Actual mean of population:' + str(y.mean()))
print ('Integer encoding(mean) :' + str(Results.mean()))
print ('Dummy encoding(mean) :'+ str(Results_d.mean()))
print ('Integer encoding(std) :' + str(Results.std()))
print ('Dummy encoding(std) :'+ str(Results_d.std()))

#Including genre label

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres', 'Type', 'Content Rating'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
modell2 = LinearRegression()
modell2.fit(X_train,y_train)
Resultsl2 = modell2.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Resultsl2, name = 'Linear(inc Genre) - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c', 'Type', 'Content Rating'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
modell2_d = LinearRegression()
modell2_d.fit(X_train_d,y_train_d)
Resultsl2_d = modell2_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Resultsl2_d, name = 'Linear(inc Genre) - Dummy'),ignore_index = True)

print ('Integer encoding(mean) :' + str(Results.mean()))
print ('Dummy encoding(mean) :'+ str(Results_d.mean()))
print ('Integer encoding(std) :' + str(Results.std()))
print ('Dummy encoding(std) :'+ str(Results_d.std()))

resultsdf.set_index('Series Name', inplace = True)
plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')
plt.subplot(3,1,2)
resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')
plt.subplot(3,1,3)
resultsdf['Mean Squared Log Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'Mean Squared Log Error')
plt.show()


