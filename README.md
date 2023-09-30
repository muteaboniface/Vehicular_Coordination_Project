# VEHICULAR COORDINATION IN LANE CHANGE PREDICTION FOR COLLISION AVOIDANCE

 AIM - Develop a driver assist system to aid in lane change and overtaking decision making.
 
 OBJECTIVE - Design and implement a vehicular coordinated framework using state of the art recurrent 
neural networks in collision avoidance during overtaking and lane changes to reduce 
accident fatalities.

 
Directory Structure
-------------------
```
Vehicular Coordination Project/ .... Top src dir
|-- data ............. Folder containig the HighD dataset
|-- environ ............. folder containing the azureml_environment and conda_dependencies configuration files
|-- inference_data ............. Folder containing the sample data to use for testing
|-- LICENSE ..................... Full license text
|-- lstmpicklemodel.pkl ................ LSTM saved model as specified in the scoring file
|-- output ............. folder containing serialized data
|-- Vehicular_Coordination_Visualization.ipynb ..................... data visualisations
|-- Vehicular_Coordination_Prepare_Data.ipynb ..................... data preparation
|-- Vehicular_Coordination_rnn_model.ipynb ..................... model building and testing
|-- Vehicular_coordination_deploy_azure.ipynb ..................... model deployment on azure
|-- score1.py ............. scoring file
|-- rnnclass.py ................ implemented rnn class
```

Usage
-----
1. Put the HighD dataset in the ``data/`` directory.
2. You must run the prepare_data notebook before the rnn_model notebook

3. The scoring file has been customised to fit this usecase. It is not the standard format
3. Use python3
4. Spin a local server in the same directory as the scoring file > azmlinfsrv --entry_script score1.py
4. Use ! curl -p 127.0.0.1:5001/score to perform inference


Notice
------
All rights reserved.









