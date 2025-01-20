import os
import json
root_folder = '../eye_tracking_participant'
participant = []
for root, dirs, files in os.walk(root_folder):
    print(f"Current Directory: {root}")
    
    for dir in dirs:
        # Build the correct path to meta/participant
        participant_path = os.path.join(root, dir, "meta", "participant")
        
        if os.path.isfile(participant_path):
            print(f"Found Participant File: {participant_path}")
            with open(participant_path, 'r') as f:
                data = json.load(f)    
                number = data['name']
                participant.append((dir,number))

print(participant)
            

        