import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from collections import OrderedDict

# Path to your service account key file
service_account_path = 'C:\\Users\\marko\\OneDrive\\Documents\\GitHub\\cannon-ball_test\\serviceAccountKey.json'

# Initialize the Firebase Admin SDK
cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

def fetch_subject_data(study_id):
    all_participant_df = []
    
    # Reference to the subjects sub-collection
    subjects_ref = db.collection('Cannonball_MF_pilot').document(study_id).collection('subjects')
    
    try:
        
        # Get all documents in the subjects sub-collection
        subjects = subjects_ref.stream()
        for subject in subjects:
           
                subject_data = subject.to_dict()
                
                
                
                subject_df = pd.DataFrame(subject_data['trial_data'].values())
                print(subject_df)
                subject_df['subjectID'] = subject.id  # Add the participant ID to the data
                subject_df['trial_info_file'] = subject.trialinfo
                #subject_data['study_id'] = study_id  # Add the study ID to the data
                subject_df = subject_df.sort_values(by="trial") 
                subject_df = subject_df.replace(-999, float("nan"))
                
                #print(subject_df)
                        
                
                all_participant_df.append(subject_df)
        all_participant_data = pd.concat(all_participant_df)
        return all_participant_data
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

# Fetch data for all participants
all_participant_data = fetch_subject_data('NONE')
#print(all_participant_data)   

if all_participant_data.any:
    # Do something with the fetched data, e.g., print or analyze

    # Convert the fetched data to a pandas DataFrame
    #df = pd.DataFrame(all_participant_data['trial_data'].values())    
     # Save the DataFrame as a CSV file
    all_participant_data.to_csv('all_subject_df.csv', index=False)
    #df_trial_data = df.iloc[:, [3]]
    #df_trial_data = df_trial_data.sort_values(by="trial")
    #df_trial_data.to_csv('pilot_trial_data.csv', index=False)
    #trial_data_dict = df_trial_data.to_dict(orient='records')
    #print(df_trial_data)
    # Order the dictionary by key
    #ordered_trial_data_dict = [OrderedDict(sorted(item.items())) for item in trial_data_dict]
    #print(ordered_trial_data_dict[1])
    
     
    
    #print("Fetched Data for All Participants:")
    #for data in all_participant_data:
    #    print(data)

    # You can now run your analysis on `all_participant_data`
    # Example: print the number of participants
    print(f"Number of participants: {len(all_participant_data)}")
else:
    print("No data found.")
