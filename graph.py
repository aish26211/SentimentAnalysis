import graphviz

# Create a new directed graph
workflow = graphviz.Digraph(comment='Experiment Workflow', format='png')

# Define the nodes for each step
workflow.node('A', 'Data Collection\nGather Twitter and Amazon review data')
workflow.node('B', 'Data Preprocessing\nClean text data (tokenization, stopword removal, etc.)')
workflow.node('C', 'Feature Engineering\nConvert text to numerical representations using TF-IDF')
workflow.node('D', 'Model Training\nTrain the SVM model on the processed data')
workflow.node('E', 'Sentiment Prediction\nClassify new reviews using the trained model')
workflow.node('F', 'Visualization\nCreate graphs and charts to present findings and model performance')

# Define the edges between nodes (the workflow steps)
workflow.edge('A', 'B')
workflow.edge('B', 'C')
workflow.edge('C', 'D')
workflow.edge('D', 'E')
workflow.edge('E', 'F')

# Render and display the graph
workflow.render('experiment_workflow', view=True)
