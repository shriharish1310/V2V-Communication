# Beamforming for V2V Communication Utilizing DeepLearning
Baseline for the [DeepSense6G Beam Prediction Challenge 2023](https://deepsense6g.net/beam_prediction_challenge_2023/) based on V2V datasets. 

## Requirements to run the code

The Python environment used is exported in ```deepsense-env.txt```, but most of them are not 
necessary. All that is necessary to run the scripts is a basic Python installation and the
modules: 
- NumPy
- SciPy
- Pandas
- tqdm
- Pickle
- Matplotlib

Here are the recommended steps to setup a Python environment that can run this code:

1. Install Python -> https://github.com/PackeTsar/Install-Python/blob/master/README.md#windows-
2. Download and Install VSCode for windows -> https://code.visualstudio.com/
2. Launch VS Code and open a new terminal Window within VS Code
3. Install the required packages using pip
```pip install numpy scipy pandas tqdm matplotlib```

## Setting up the environment
Follow these steps to set up the environment for your project:

1. Download all the necessary files and place them in the directory where you have installed the Python packages.
Maintain Directory Structure:
2. Ensure that the directory structure of the files remains unchanged. This is crucial to avoid errors when running the program.

**To Reproduce the Results, Follow These Steps:**

1. Models Used:
    - The project has tested with a total of 5 models:
        - SVM (Support Vector Machine)
        - KNN (K-Nearest Neighbors)
        - Random Forest
        - XGBoost
        - Decision Tree
2. Running the Models:
    - To run a specific model, open the corresponding Python file present in the code directory:
        - SVM: svm_model.py DONE
        - KNN: knn_model.py DONE
        - Random Forest: random_forest.py DONE
        - XGBoost: xgboost.py 
        - Decision Tree: decision_tree.py DONE
3. Instructions:
    - Navigate to the Python file of the desired model and execute it to reproduce the results.
4. Different scenarios have been used to test these models to get more information refer to [Deepsense 6G Scenarios](https://www.deepsense6g.net/scenarios36-39/)
5. After specifying the model and running the file you'll be prompted to choose the scenarios on which you want to test the code.
6. Specify the scenario names as mentioned and run the code to get the desired output