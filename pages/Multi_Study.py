import streamlit as st
import pandas as pd
import numpy as np
import pickle
import csv
import time
import json
import xgboost as xgb

#model = pickle.load(open('LONGBOAT_Model.pkl','rb'))
with open('LONGBOAT_Model.json','r') as f:
    model_dict = json.load(f)
model = xgb.Booster(model_file=None,model_buf=model_dict)
run_model = False #Boolean that switches to true when run model button has been pressed


Title = st.title("Multiple Study Forecasts")
display_text = st.text("Upload a file below in order to run the model")
file = st.file_uploader("Upload data", ['csv','xlsx'], False)

def Tidy_Data(df):
    
    Model_Cols = ['Clinical Study ID',"Phase","Therapy Area","Population <18 yrs Studied","FTIH","Healthy Volunteer",
                  'FSFV Year','Total Enrol + Treat (mths)','Country List','FSO (based on logic)',"Number of Centres",
                  "Number of Subjects",'Full Title']
    
    df_cols = df.columns.tolist()
    
    if all(col in df_cols for col in Model_Cols ) == False:
        return 'Model Incompatible'
        
    else:
        df = df[Model_Cols] #This drops any columns not in Model Cols list
    
            
        #We only need to know if a trial is in phase 1-4, A/B isn't as necessary and will only increase the feature space. So we replace the phase names in the column prior to one hot encoding
        df = df.replace(["PHASE IIA","PHASE IIB","PHASE IIIA","PHASE IIIB"],["PHASE II","PHASE II","PHASE III","PHASE III"])
        df = pd.get_dummies(df,columns=["Phase"]) #One hot encoding
        
        #Ensure that all Phase values are in the uploaded dataframe
        Phases = ['Phase_PHASE I', 'Phase_PHASE II','Phase_PHASE III', 'Phase_PHASE IV']
        df_cols = df.columns.tolist()
        for col in df_cols:
            if 'PHASE' in col:
                Phase_index = Phases.index(col)
                Phases.pop(Phase_index) #Remove phase from phase list if it is present in uploaded dataframe
        Len = len(df)
        Zero_List = [0] * Len
        for phase in Phases:
            df[phase] = Zero_List
        
        
        df = df.rename(columns = {'FSO (based on logic)':'FSO'}) #Collates both FSO and FSO (based on logic) into the same category before one hot encoding
        df = pd.get_dummies(df,columns=["FSO"])
        
        #Ensure all FSO categories are covered
        FSO_Options = ['FSO_No', 'FSO_Unknown','FSO_Yes']
        df_cols = df.columns.tolist()
        for col in df_cols:
            if 'FSO' in col:
                FSO_Index = FSO_Options.index(col)
                FSO_Options.pop(FSO_Index)
        for fso in FSO_Options:
            df[fso] = Zero_List
            
        #Similarly, healthy patients vs ill patients can be binary encoded
        #Currently, there is a weird encoding of 0 for non-healthy patients and Yes/Yes(by title) for yes for healthy
        Healthy = df["Healthy Volunteer"].tolist()
        Binary_Healthy = []
        for h in Healthy:
            if type(h) == int:
                Binary_Healthy.append(0)
            else:
                Binary_Healthy.append(1) #If datatype is a string, from looking at value counts we are sure that this is yes
        df["Healthy Volunteer"] = Binary_Healthy
        
        
            
        #Binary encode FITH (First time in humans)
        #Same silly encoding system as used for healthy volunteer
        ftih = df["FTIH"].tolist()
        Binary_ftih = []
        for f in ftih:
            if type(f) == int:
                Binary_ftih.append(0)
            else:
                Binary_ftih.append(1) #Same as above
        df["FTIH"] = Binary_ftih
        
        
        #Loop through every row of the dataframe to find out which countries are in each trial
        No_of_Countries = []
        for i in range(len(df)):
    
            row = df.loc[df.index == i]
    
            Countries = row["Country List"].iloc[0] #All countries the trial took place in
            Countries_list = []
            if ';' in Countries: #i.e. if multiple countries for one study
                split_countries = Countries.split(';')
                for c in split_countries:
                    Countries_list.append(c.strip())
            else:
                Countries_list.append(Countries)
            No_of_Countries.append(len(Countries_list))
        
        df["No_of_Countries"] = No_of_Countries

        
        #Years are represented as the years since 2010, the earliest trial in our training set
        #Simple standardisation
        Years = df['FSFV Year'].tolist()
        Years = [year-2010 for year in Years]
        df['FSFV Year'] = Years
       
        
        #We don't need to one hot encode subject population, as this is a binary label anyway, either pediatric or adult
        Ped = df["Population <18 yrs Studied"]
        Ped = Ped.to_list()
    
        Pediatric = []
        for i in range(len(Ped)):
            if "No" in Ped[i]:
                Pediatric.append(0)
            else:
                Pediatric.append(1) #As looking at value counts there are some situations with yes - in care
        df["Population <18 yrs Studied"] = Pediatric
        df = df.rename(columns={"Population <18 yrs Studied": "Pediatric Study"})
        
        TA_List = ['Respiratory', 'Metabolic', 'Neurosciences', 'Infectious Diseases', 'Inflammation', 
                   'Cardiovascular Diseases', 'Oncology', 'Urology', 'Gastrointestinal', 'Ophthalmology', 
                   'Dermatology', 'Other']
        #Label any TA not in the established training set of therapy areas as 'Other', which is already coded for and thus can be one hot encoded
        
        TAs = df["Therapy Area"].tolist()
        Valid_TAs = []
        for ta in TAs:
            if ta not in TA_List:
                Valid_TAs.append('Other')
            else:
                Valid_TAs.append(ta)
        
        #This creates dummies of the therapy areas but only the ones present in the uploaded data
        df = pd.get_dummies(df,columns=["Therapy Area"])
    
        
        #We need to ensure there are also dummies of the other therapy areas present in the dataset. Which will all be filled with 0 as none of uploaded trials are in that category
        #These columns need to be present for the ML model to intake. These are all TA present in training set
        TA_List = ['Therapy Area_Cardiovascular Diseases', 
                 'Therapy Area_Dermatology',
                 'Therapy Area_Gastrointestinal', 
                 'Therapy Area_Infectious Diseases',
                 'Therapy Area_Inflammation', 
                 'Therapy Area_Metabolic',
                 'Therapy Area_Neurosciences',
                 'Therapy Area_Oncology',
                 'Therapy Area_Ophthalmology', 
                 'Therapy Area_Other',
                 'Therapy Area_Respiratory', 
                 'Therapy Area_Urology']
        
        df_cols = df.columns.tolist() #List of all columns, now including therapy area dummies
        for col in df_cols:
            if 'Therapy Area' in col:
                TA_List_index = TA_List.index(col)
                TA_List.pop(TA_List_index) #Remove TAs from TA list if they are already present in our dataframe
        
        Len = len(df)
        Zero_List = [0] * Len

        for TA in TA_List:
            df[TA] = Zero_List #Add new column to dataframe, with the TA name as column header, then fill with 0s same length as the df
            
        
        
        
        #Feature engineering:
    
        df["Dur_Subjects"] = df['Total Enrol + Treat (mths)'] * df["Number of Subjects"]
        df["Dur_Centres"]  = df['Total Enrol + Treat (mths)'] * df['Number of Centres']
        
        df["SpC"] = df["Number of Subjects"] / df["Number of Centres"] #Subjects per centre
        
        df.rename(columns = {'Total Enrol + Treat (mths)':"Duration (Months)"},inplace=True)
    
        
        #All columns for dataframe
        Cols = ['Clinical Study ID',
         'Pediatric Study',
         'FSFV Year',
         'FSO_No', 
         'FSO_Unknown',
         'FSO_Yes',
         'FTIH',
         "Healthy Volunteer",
         'Number of Centres',
         'Number of Subjects',
         "Duration (Months)",
         "Dur_Subjects",
         "Dur_Centres",
         "SpC",
         'Phase_PHASE I', 
         'Phase_PHASE II',
         'Phase_PHASE III', 
         'Phase_PHASE IV',
         'No_of_Countries',
         'Therapy Area_Cardiovascular Diseases', 
         'Therapy Area_Dermatology',
         'Therapy Area_Gastrointestinal', 
         'Therapy Area_Infectious Diseases',
         'Therapy Area_Inflammation', 
         'Therapy Area_Metabolic',
         'Therapy Area_Neurosciences',
         'Therapy Area_Oncology',
         'Therapy Area_Ophthalmology', 
         'Therapy Area_Other',
         'Therapy Area_Respiratory', 
         'Therapy Area_Urology']
        
        labelled_df = df[Cols]
        
        ML_df = labelled_df.drop(["Clinical Study ID"],axis=1)
        
        return (labelled_df, ML_df)


def float_to_money_format(number):
    money_format = "£{:,.1f}"
    return money_format.format(number)    

def run(model,df):
    IDs = df['Clinical Study ID'].tolist()
    df = df.drop(['Clinical Study ID'],axis= 1)
    
    Results = []
    
    for i in range(len(df)):
        X1 = df.iloc[i].to_numpy().reshape(1,-1)
        prediction = model.predict(X1)[0]
      
        cost = float_to_money_format(prediction)
        
        Results.append([IDs[i],cost])
    
    results_df = pd.DataFrame(columns = ['ID','Predicted Cost (£)'])
    IDs = []
    Costs = []
    for result in Results:
        IDs.append(result[0])
        Costs.append(result[1])
    results_df['ID'] = IDs
    results_df['Predicted Cost (£)'] = Costs
    
    
    st.dataframe(results_df)
    
    
    file = open('Multiple Study Results.csv', 'w+', newline ='')
    with file:   
        write = csv.writer(file)
        header = ['ID','Predicted Cost (£)']
        write.writerow(header)
        for result in Results:
           write.writerow(result)
        
    return Results


def make_readable(df):
    
    Shown_df = pd.DataFrame(columns = ['Clinical Study ID','Paediatric study','Year of Study','First time in humans','Healthy volunteers','Number of Centers','Number of Subjects','Duration of Study (months)','Number of Countries','Phase','Therapy Area'])                                                               
    
    IDs = df['Clinical Study ID'].tolist()
    Ped = df['Pediatric Study'].tolist()
    Year = df['FSFV Year'].tolist()
    FTIH = df['FTIH'].tolist()
    Healthy = df['Healthy Volunteer'].tolist()
    Centers = df['Number of Centres'].tolist()
    Subjects = df['Number of Subjects'].tolist()
    Duration = df['Duration (Months)'].tolist()
    Countries = df['No_of_Countries'].tolist()
    
    
    #fso_df = df.filter(regex='^FSO')
    phase_df = df.filter(regex='^Phase')
    ta_df = df.filter(regex='^Therapy Area')
    
    phase = phase_df.idxmax(axis=1).str.replace('Phase_', '')
    therapy_area = ta_df.idxmax(axis=1).str.replace('Therapy Area_', '')
    #fso = fso_df.idmax(axis=1).str.replace('FSO_', '')

    #FSO = fso['FSO'].tolist()
    
    Shown_df['Clinical Study ID'] = IDs
    Shown_df['Paediatric study'] = Ped
    Shown_df['Year of Study'] = Year
    Shown_df['First time in humans'] = FTIH
    Shown_df['Healthy volunteers'] = Healthy
    Shown_df['Number of Centers'] = Centers
    Shown_df['Number of Subjects'] = Subjects
    Shown_df['Duration of Study (months)'] = Duration
    Shown_df['Number of Countries'] = Countries
    Shown_df['Phase'] = phase
    Shown_df['Therapy Area'] = therapy_area
    
    #Shown_df= pd.concat([Shown_df,phase,therapy_area],axis=1)
    
    return Shown_df
    
     
    
    
    
def download_results(Results):
    file = open('Multiple Study Results.csv', 'w+', newline ='')
    with file:   
        write = csv.writer(file)
        header = ['ID','Predicted Cost (£)']
        write.writerow(header)
        for result in Results:
           write.writerow(result)

def show_results(Results):
    
    results_df = pd.DataFrame(columns = ['ID','Predicted Cost (£)'])
    IDs = []
    Costs = []
    for result in Results:
        IDs.append(result[0])
        Costs.append(result[1])
    results_df['ID'] = IDs
    results_df['Predicted Cost (£)'] = Costs
    
    st.dataframe(results_df)
    


if file and run_model == False:
    
    df = pd.read_csv(file)
    output = Tidy_Data(df)
    if type(output)==str:
        st.markdown('Uploaded data is incompatible with the model')
    else:
        labelled_df,ML_df = output
        
        shown_df = make_readable(labelled_df)
        st.dataframe(shown_df)
        if st.button('Run Model'):
            st.button('Show Results')
            st.button('Download Results')
            Results = run(model,labelled_df)
            run_model = True
    
        
        
        

    
        
    
    


