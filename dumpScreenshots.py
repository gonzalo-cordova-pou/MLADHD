'''
A script to dump all the screenshots from all the sessions into a single folder for manual inspection.
'''
import os
import shutil


DEPLOYMENT_PATH = os.path.join(os.getcwd(), '..', 'deployment')
DESTINATION_PATH = os.path.join(os.getcwd(), '..', 'newScreenshots')

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)

# make a destination dir for focused and distracted screenshots
FOCUSED_PATH = os.path.join(DESTINATION_PATH, "focused")
if not os.path.exists(FOCUSED_PATH):
    os.makedirs(FOCUSED_PATH)
DISTRACTED_PATH = os.path.join(DESTINATION_PATH, "distracted")
if not os.path.exists(DISTRACTED_PATH):
    os.makedirs(DISTRACTED_PATH)

for session in os.listdir(DEPLOYMENT_PATH):
    # iterate over focused and distracted folders
    for folder in ['focused', 'distracted']:
        # iterate over screenshots
        for file in os.listdir(os.path.join(DEPLOYMENT_PATH, session, 'screenshots', folder)):
            shutil.copyfile(os.path.join(DEPLOYMENT_PATH, session, 'screenshots', folder, file),
                            os.path.join(DESTINATION_PATH, folder, file))