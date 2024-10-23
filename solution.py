# This is Dogukan Arasli's solution to FitMinds Part Time Problem Analyst programming case #
# Throughout the script, the lines that print or visualize are taken into comment for clarity in the output

# IMPORTS #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split#,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import SMOTE,RandomOverSampler
#from imblearn.combine import SMOTEENN,SMOTETomek
#from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
#import time as t

# STORING DATA #

input_data_df = pd.read_excel('FM2024 - Dataset.xlsx', sheet_name = 'Input Data', skiprows=1,usecols='B:L')
#Workspace needs to be set into a folder that includes the dataset and the script

# PREPROCESSING AND FEATURE ENGINEERING #

# Missing value check #
#print(input_data_df.isna().any())
# No action is needed since no missing value

# Sample Size #
ss = len(input_data_df)
#print(ss) 11897

# Distribution over HCP's #
visits_to_hcps = input_data_df['HCP Name'].value_counts()
#print(visits_to_hcps)
#print('Number of HCPs reached only 1 time:', (visits_to_hcps == 1).sum())
# There are 7589 HCP's while 5678 of them has been reached only 1 time and max number is 47

# Adding a new column of Num of Visits to HCPs #
input_data_df['Number of Visits to HCPs'] = input_data_df['HCP Name'].map(visits_to_hcps)

# Distribution over Representatives #
reps = input_data_df['Representative Name'].unique()
rep_visits = input_data_df['Representative Name'].value_counts()
#print('Number of Representatives:', len(reps))
#plt.bar(x= reps, height = rep_visits[reps].values)
#plt.title('Number of Visits a representative made')
#plt.xlabel('Representative Names')
#plt.ylabel('Number of Visits')
#plt.show()
# Among 15 representative other than the representative 'TS', each representaive made between 700-1000 visits

# Adding 'WeekDay' column (Friday, Monday etc.) as it makes more sense to work with #
for i in range(ss):
    # In the date column there were 20 dates addressing 29th of February 2023 which is not a real date
    # They are mapped back to 28th of February 2023 (They could be mapped to 28th of Feb or 1st of March randomly)
    if type(input_data_df.loc[i, 'Date']) == str :
        d = str(input_data_df.loc[i, 'Date']).split('.')
        input_data_df.loc[i, 'Date'] = d[-1] + '-' + d[-2] + '-' + '28' + ' 00:00:00'
input_data_df['Date'] = pd.to_datetime(input_data_df['Date'])
input_data_df['Week Day'] = input_data_df['Date'].dt.day_name()

# Distribution over the week days #
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
week_day_visits = input_data_df['Week Day'].value_counts()
#plt.bar(x = week_days, height = week_day_visits[week_days])
#plt.title('Number of Visits made each Week Day')
#plt.xlabel('Week Days')
#plt.ylabel('Number of Visits')
#plt.show()

# Adding column Rates of Successful Visits by HCPs #
succ_visits = input_data_df[input_data_df['Reach'] == 'Ulasildi']['HCP Name'].value_counts()
input_data_df['Rate of Successful Visits to HCPs'] = input_data_df['HCP Name'].map(succ_visits)
input_data_df.replace(to_replace=np.nan,value=0,inplace=True)
input_data_df['Rate of Successful Visits to HCPs'] = input_data_df['Rate of Successful Visits to HCPs'] / input_data_df['Number of Visits to HCPs']

# Distribution of Success over Week Days #
week_day_succ = input_data_df[input_data_df['Reach'] == 'Ulasildi']['Week Day'].value_counts()
#plt.bar(x = week_days, height = week_day_succ[week_days])
#plt.title('Number of Visits made each Week Day')
#plt.xlabel('Week Days')
#plt.ylabel('Number of Visits')
#plt.show()
# This shows a balanced distribution

# ENCODINGS and SCALINGS #

# Turning Activity time to minutes
input_data_df['Activity Time'] = input_data_df['Activity Time'].apply(lambda x: x.hour * 60 + x.minute)

# Creating Numpy nd.arrays that will be used for training in x-y pairs #
X = input_data_df[['Representative Name','Activity Time','Number of Visits to HCPs','Rate of Successful Visits to HCPs',
                   'Specialty Name','Product Name','Week Day']].values

# One-Hot Encodings (of Weeks and Representative Name)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,-3,-2,-1])], remainder='passthrough')
X = ct.fit_transform(X)
#print(X)

# Label Encodings (of Reach) #
Y = input_data_df['Reach'].values
le = LabelEncoder()
Y = le.fit_transform(Y)
#print(Y)

# Splitting the dataset into training and testing #
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1, random_state = 1,stratify=Y) #CHANGE TEST_SIZE IF NECESSARY
# test_size defined as the smallest amount that does not interfene with training of nn / logistic regression models while also
# providing large enough numbers for the test set evaluation.

# Scaling continous variables in the standart sense #
sc = StandardScaler(with_mean=False)
X_train[:,28:] = sc.fit_transform(X_train[:,28:])
X_test[:,28:] = sc.transform(X_test[:,28:])

# Basically what I have done up so far is: After determining the data is sparse regarding HCPs (5678s of them only have 1 visit),
# I could not encode them one-hot way. Instead, I encoded them with 2 variables, number-of-visits and successful visit rate to a HCP.
# By doing so, I implicitly grouped them by these variables. I also added a weekday column as it made more sense regarding weekly schedules. 
# If classification models turn out to be insufficient then adding new features (like month of the year, representative success rate) 
# can be added. After encoding and scalings data is ready to be trained on for now.

# MACHINE LEARNING MODEL # See the comments under IMPORT section for the whole lists of methods that was tried here.

# Training the model #
classifier = LogisticRegression(random_state=1)
classifier.fit(X_train,Y_train)

# Testing the model #
Y_pred_train = classifier.predict(X_train)
Y_pred_test = classifier.predict(X_test)

# Printing the Results #
print('Logistic Regression')
cm = confusion_matrix(Y_test,Y_pred_test)
print(cm)
print('Training Accuracy: ', accuracy_score(Y_train,Y_pred_train))
print('Test Accuracy: ', accuracy_score(Y_test,Y_pred_test))
print('-------------------------------')

# These results both show that although overall accuracy may be sufficient, the model does predict 'Ulasilamadi'
# a lot when the reach was 'Ulasildi'. This is expected since the dataset is very imbalanced between Positives
# and Negatives.

# HANDLING THE IMBALANCE #

# Training the model #
sampler = RandomUnderSampler(random_state=1) # RandomUnderSampler was slightly better than other resampling methods
X_train_resampled, Y_train_resampled = sampler.fit_resample(X_train,Y_train)
classifier_resampled = LogisticRegression(random_state=1)
classifier_resampled.fit(X_train_resampled,Y_train_resampled)

# Testing the model #
Y_pred_train = classifier_resampled.predict(X_train)
Y_pred_test = classifier_resampled.predict(X_test)

# Printing the Results #
print('Logistic Regression with Resampling')
cm = confusion_matrix(Y_test,Y_pred_test)
print(cm)
print('Training Accuracy: ', accuracy_score(Y_train,Y_pred_train))
print('Test Accuracy: ', accuracy_score(Y_test,Y_pred_test))

# As it can be seen, when known imbalance solutions are used, although the number of wrongly guessed 'Ulasilmadi'
# decreases, wrongly guessed 'Ulasildi's increases so overall accuracy stays same. For further testing, 
# I will try neural networks as dataset is already large

# NEURAL NETWORK #

#print('-------------------------------')
#print('Neural Network')
batch_size = 1024 # TUNE THIS
train_size = len(Y_train)
test_size = len(Y_test)

# Changing the data to torch.tensors #
X_train_tensor = torch.tensor(X_train.toarray(),dtype=torch.float32)
Y_train_tensor = torch.tensor(np.float32(Y_train.reshape(train_size,1)), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(),dtype=torch.float32)
Y_test_tensor = torch.tensor(np.float32(Y_test.reshape(test_size,1)), dtype=torch.float32)

# Creating the test set objects #
train_dataset = torch.utils.data.TensorDataset(X_train_tensor,Y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor,Y_test_tensor)

# Creating data loaders #
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# Creating Neural Network #
class my_nn(nn.Module):
    def __init__(self,D,H1,H2,H3,H4):
        super(my_nn, self).__init__()
        torch.manual_seed(1)
        self.fc1 = nn.Linear(D,H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.fc2 = nn.Linear(H1,H2)
        #self.bn2 = nn.BatchNorm1d(H2)
        #self.fc3 = nn.Linear(H2,H3)
        #self.bn3 = nn.BatchNorm1d(H3)
        #self.fc4 = nn.Linear(H3,H4)
        #self.bn4 = nn.BatchNorm1d(H4)
        #self.fc5 = nn.Linear(H4,1)        

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = self.fc3(x)
        #x = self.bn3(x)
        #x = F.relu(x)
        #x = self.fc4(x)
        #x = self.bn4(x)
        #x = self.fc5(x)
        x = F.sigmoid(x)
        return x
    
# Hyperparameters # TUNE THESE
D = X.shape[1] # Input Dimension
H1 = 10 # Hidden Layer Size
H2 = 1 # Hidden Layer Size
H3 = 5 # Hidden Layer Size
H4 = 2  # Hidden Layer Size
num_of_epoch = 5
learning_rate = 0.5
my_loss = nn.BCELoss()
my_model = my_nn(D,H1,H2,H3,H4) # Creating the model instance
my_opt = optim.SGD(my_model.parameters(),lr=learning_rate,weight_decay=0.05)

# Training Loop #
def train(model, criterion, optimizer, epochs, dataloader, verbose=False): #verbose=True for tuning

  loss_history = [] 
  for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):    
      
      # Current batch:
      inputs, labels = data

      # Zero the gradients as PyTorch accumulates them
      optimizer.zero_grad()

      # Obtain the scores
      outputs = model(inputs)

      # Calculate loss
      loss = criterion(outputs, labels)

      # Backpropagate
      loss.backward()

      # Update the weights
      optimizer.step()

      loss_history.append(loss.item())

    # Print the loss    
    if verbose: print(f'Epoch {epoch+1} / {epochs}: avg. loss of last 5 iterations {np.sum(loss_history[:-6:-1])/5}')

  return loss_history

# Training the model # THIS PART IS TAKEN INTO COMMENT AS IT TAKES SLIGHTLY SOME TIME
#loss_history = train(my_model,my_loss,my_opt,num_of_epoch,train_loader)
#plt.plot(loss_history) #For tuning
#plt.xlabel('Iteration number')
#plt.ylabel('Loss value')
#plt.show()

# Test set accuracy #
#correct = 0
#total = 0
#all_labels = []
#all_preds = []

#with torch.no_grad():
  for data in train_loader:
    samples, labels = data
    predicted = my_model(samples) > 0.5 # Set every probability that is bigger than 0.5 as 'Ulasildi'
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  train_acc = correct / total

  correct = 0
  total = 0    
  for data in test_loader:
    samples, labels = data
    predicted = my_model(samples) > 0.5 # Set every probability that is bigger than 0.5 as 'Ulasildi'
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    all_labels.extend(labels.numpy())   # Store every prediction from the sample to a general var 
    all_preds.extend(predicted.numpy()) # Store every label from the sample to a general var

  test_acc = correct / total

# Calculate confusion matrix #
#cm = confusion_matrix(all_labels, all_preds)
#print(cm)
#print('Training Accuracy: ', train_acc)
#print('Test Accuracy: ', test_acc)

# After a careful search on the hyperparameter plane: neural network does not perform better than logistic regression, 
# usually overfits to either 'Ulasilmadi' or, when resampling is used, to 'Ulasildi'; causing overall accuracy to 
# decrease more significantly compared to Logistic Regression with Resampling. Theoretically, this overfit may seem 
# promising however, at this point, it is not worth to pursue any further. Therefore, I will continue solution with
# Logistic Regression with Resampling. Main reason behind this choice is although this model's overall accuracy is
# the worst between 3 models presented in this report, It is better at determining 'Ulasildi' (i.e., class accuracy). 
# Therefore, it provides reliable predictions both in its 'Ulasilmadi' decisions (although lower class accuracy in 
# 'Ulasilmadi' compared to other models, still sufficient) and 'Ulasildi' decisions (which is more important in the 
# following part of the problem).

# SCHEDULE #

# First we need to split a week into discrete timeslots for optimization purposes. #
#print('-------------------------------')
#print('Max. Dialing time: ', input_data_df['Dialing Time (sec)'].max(), 'seconds') # 6.9
#print('Max. Talking time: ', input_data_df['Talk Time (sec)'].max(), 'seconds') # 49.99
# Total is approximately 60 seconds or 1 minute, So we can leave 1 minute for each talk,
# i.e., we can set our timeslots 1 minutes apart. However, this would yield a very large dataset.

# !Remember: Activity Time column was transformed to minutes.
# Checking to see if there is a patern between representatives regarding their work hours
#print('Earliest Visit time for Representatives: ', input_data_df.groupby('Representative Name')['Activity Time'].min().min()) #480
#print('Latest Visit time for Representatives: ', input_data_df.groupby('Representative Name')['Activity Time'].max().max()) #1163
# We can safely assume that everyday starts at min 480 and ends at 1163

#print('Maximum visits made by a representative: ', input_data_df['Representative Name'].value_counts().max()) #1043
# Even if we have divided a day into 200 slots, that would yield 7 * 200 = 1400 slots which is more than enough to schedule
# all calls. Further optimization can also be done in a sequential manner within these slots after reaching to a optimal solution.
# As this problem has 16M decision variables (11897 alignment * 1400 slots), I will use a greedy heuristic.

# Creating possible time slots based on the findings #
timeslots = np.linspace(480,1163,200) # linearly spaced time slots
timeslot_df = pd.MultiIndex.from_product([week_days,timeslots], names=["Week Day (new)", "Activity Time (new)"]) # creating time slots for every day of the week

# Duplicating every sample for every time slot on every day #
input_data_df = input_data_df.assign(key=1).merge(
    pd.DataFrame({'key': 1, 'Week Day (new)': timeslot_df.get_level_values(0), 'Activity Time (new)': timeslot_df.get_level_values(1)}), 
    on='key'
).drop(columns='key')

# Getting the data
X = input_data_df[['Representative Name','Activity Time (new)','Number of Visits to HCPs','Rate of Successful Visits to HCPs',
                   'Specialty Name','Product Name','Week Day (new)']].values

# Applying the transformations #
X = ct.transform(X)
X[:,28:] = sc.transform(X[:,28:])

# Predict the probabilities #
input_data_df['Success Probability'] = classifier_resampled.predict_proba(X)[:,1]

# Sort #
input_data_df.sort_values(by='Success Probability', ascending=False, inplace=True, axis=0)

# Iteratively drop values to prevent relapsing and making 1 sell only 1 time #
final_schedule = pd.DataFrame() # Accepted candidates will accumulate in final_schedule

selected_mask = np.zeros(len(input_data_df), dtype=bool)

# This takes so much time, must be optimized.
for i in range(len(input_data_df)):
  if selected_mask[i]:
    continue

  rep = input_data_df.iloc[i,0]   #column: Representative Name
  time = input_data_df.iloc[i,-2] #column: Activity time (new)
  day = input_data_df.iloc[i,-3]  #column: Week Day (new)
  id = input_data_df.iloc[i,6]    #column: HCP Integration ID

  final_schedule = pd.concat([final_schedule, input_data_df.iloc[[i]]])

  # Create boolean masks to mark rows to be excluded in the next iterations
  rep_time_day_mask = (input_data_df['Representative Name'] == rep) & \
    (input_data_df['Activity Time (new)'] == time) & (input_data_df['Week Day (new)'] == day)

  hcp_id_mask = (input_data_df['HCP Integration ID'] == id)

  # Update the selected mask to mark these rows as assigned
  selected_mask |= (rep_time_day_mask | hcp_id_mask)

print(final_schedule)
