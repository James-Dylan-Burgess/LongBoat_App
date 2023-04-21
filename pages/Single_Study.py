import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import csv
import json


#model = pickle.load(open('LONGBOAT_Model.pkl','rb'))

xbg_reg = xgb.Booster()

model = xbg_reg.load_model("LONGBOAT_Model.json")

run_model = False #Boolean that switches to true when run model button has been pressed


Title = st.title("Single Study Forecast")

#display_text = st.text("Once study parameters are entered and confirmed you can run the model")



st.sidebar.header("Study Parameters")

Study_Ref = st.sidebar.text_input('Study Reference')
Ped_Status = st.sidebar.selectbox("Paediatric Status", ["Yes","No"])
Study_Start = st.sidebar.number_input("Year of study start",min_value=2010,max_value=2030,value=2023,step=1)
FSO = st.sidebar.selectbox("FSO", ["Yes","No","Unknown"])
FTIH = st.sidebar.selectbox("First time in humans?", ["Yes","No"])
HV = st.sidebar.selectbox("Healthy Volunteer?", ["Yes","No"])
Centres = st.sidebar.number_input('Number of Centres',min_value=1,value=1)
Subjects = st.sidebar.number_input('Number of Subjects',min_value=1,value=1)
Duration = st.sidebar.number_input('Duration of study in months',min_value=1,value=1)  ## Possible to replace this with Study start date and study end date, using calendar selection (Also negates the FSFV)
Phase = st.sidebar.selectbox("Study Phase", [1,2,3,4])
Countries = st.sidebar.number_input("Number of Countries",min_value=1,value=1)
TA = st.sidebar.selectbox("Therapy Area",['Respiratory', 'Metabolic', 'Neurosciences', 'Infectious Diseases', 'Inflammation', 'Cardiovascular Diseases', 'Oncology', 'Urology', 'Gastrointestinal', 'Ophthalmology','Dermatology', 'Other'])

params = [Study_Ref,Ped_Status,Study_Start,FSO,FTIH,HV,Centres,Subjects,Duration,Phase,Countries,TA]



###########

#Need a function here for data validation

#########

def store_params(parameters):
    
    Data = []
    
    Cols = ['ID','Pediatric Study','FSFV Year','FSO_No','FSO_Unknown','FSO_Yes','FTIH','Healthy Volunteer','Number of Centres','Number of Subjects','Duration (Months)',"Dur_Subjects","Dur_Centres","SpC",'Phase_PHASE I', 'Phase_PHASE II','Phase_PHASE III', 'Phase_PHASE IV','Number_of_Countries','Therapy Area_Cardiovascular Diseases', 'Therapy Area_Dermatology','Therapy Area_Gastrointestinal', 'Therapy Area_Infectious Diseases','Therapy Area_Inflammation', 'Therapy Area_Metabolic','Therapy Area_Neurosciences','Therapy Area_Oncology','Therapy Area_Ophthalmology', 'Therapy Area_Other','Therapy Area_Respiratory', 'Therapy Area_Urology']
    df = pd.DataFrame(columns = Cols)
    
    
    Data.append(parameters[0]) #Study Reference

    Ped_Status = parameters[1]
    if Ped_Status == "Yes (Paediatrics involved)":
        Data.append(1)
    else:
        Data.append(0)
    
    Data.append(parameters[2]) #FSFV
    
    FSO = parameters[3]
    if FSO == 'Yes':
        Data.append(0)
        Data.append(0)
        Data.append(1)
    elif FSO == "No":
        Data.append(1)
        Data.append(0)
        Data.append(0)
    elif FSO == 'Unknown':
        Data.append(0)
        Data.append(1)
        Data.append(0)
    
    FTIH = parameters[4]
    if FTIH == 'Yes':
        Data.append(1)
    else:
        Data.append(0)
        
    HV = parameters[5]
    if HV == 'Yes':
        Data.append(1)
    else:
        Data.append(0)
    
    Data.append(parameters[6]) #Number of Centres

    Data.append(parameters[7]) #Number of subjects

    Data.append(parameters[8]) #Duration

    Data.append((parameters[8] * parameters[7])) #Subjects * Duration
    
    Data.append((parameters[8] * parameters[6])) #Centres * Duration

    Data.append((parameters[7] / parameters[6])) #Subjects per Centre

    Phase = parameters[9]
    if Phase == 1:
        Data.append(1)
        Data.append(0)
        Data.append(0)
        Data.append(0)
    elif Phase == 2:
        Data.append(0)
        Data.append(1)
        Data.append(0)
        Data.append(0)
    elif Phase == 3:
        Data.append(0)
        Data.append(0)
        Data.append(1)
        Data.append(0)
    elif Phase == 4:
        Data.append(0)
        Data.append(0)
        Data.append(0)
        Data.append(1)
    
    Data.append((parameters[10]))
    
    select_TAs = ['Cardiovascular Diseases','Dermatology','Gastrointestinal','Infectious Diseases','Inflammation','Metabolic','Neurosciences','Oncology','Ophthalmology','Other','Respiratory','Urology']
    TA_data = np.zeros((len(select_TAs))) 
    ind = select_TAs.index(parameters[11]) #TA
    TA_data[ind] = 1
    for ta in TA_data:    
        Data.append(ta)
    
    #print(Data)
    
    df.loc[len(df)] = Data
    
    return df

def get_params():
    params = [Study_Ref,Ped_Status,Study_Start,FSO,FTIH,HV,Centres,Subjects,Duration,Phase,Countries,TA]
    df = store_params(params)
    
    return df
    
def show_df(df):
    Cols = ['ID','Pediatric Study','FSFV Year','FSO_No','FSO_Unknown','FSO_Yes','FTIH','Healthy Volunteer','Number of Centres','Number of Subjects','Duration (Months)',"Dur_Subjects","Dur_Centres","SpC",'Phase_PHASE I', 'Phase_PHASE II','Phase_PHASE III', 'Phase_PHASE IV','Number_of_Countries','Therapy Area_Cardiovascular Diseases', 'Therapy Area_Dermatology','Therapy Area_Gastrointestinal', 'Therapy Area_Infectious Diseases','Therapy Area_Inflammation', 'Therapy Area_Metabolic','Therapy Area_Neurosciences','Therapy Area_Oncology','Therapy Area_Ophthalmology', 'Therapy Area_Other','Therapy Area_Respiratory', 'Therapy Area_Urology']
    Data = df.loc[df.index == 0].tolist()
    Drop_List = []
    for i in range(len(Cols)):
        if 'Phase_' in Cols[i] or 'Therapy Area' in Cols[i]:
            if Data[i] == 0:
                Drop_List.append(Cols[i])
    
    df = df.drop(Drop_List,axis=1)
    return df

def float_to_money_format(number):
    money_format = "£{:,.1f}"
    return money_format.format(number)    

def run(model,data):
    IDs = data['ID'].tolist()
    data = data.drop(['ID'],axis= 1)
    
    for i in range(len(data)):
        X1 = data.iloc[i].to_numpy().reshape(1,-1)
        prediction = model.predict(X1)[0]
      
    cost = float_to_money_format(prediction)
    
    Statement = "## Predicted cost for study {} is **{}**".format(IDs[0],cost)
    st.markdown(Statement)
    
    return cost,IDs    
    
    
    
def save_cost(cost,IDs):
    file = open('Results.csv', 'w+', newline ='')
    with file:   
        write = csv.writer(file)
        header = ['ID','Predicted Cost (£)']
        write.writerow(header)
        write.writerow([IDs,cost])
    
    
    
confirmed = False
if not confirmed:
    if st.sidebar.button('Confirm Parameters'):
        df = get_params()
        confirmed = True
        
if confirmed:
    st.button('Run Model',on_click=lambda:run(model,df))
    display_text = st.text('')





    
        



        

