import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

network = BayesianNetwork([
    ('Idade', 'Nível de Açúcar no Sangue'),
    ('IMC', 'Nível de Açúcar no Sangue'),
    ('Pressão Arterial', 'Nível de Açúcar no Sangue'),
    ('Histórico Familiar de Diabetes', 'Nível de Açúcar no Sangue'),
    ('Atividade Física', 'Nível de Açúcar no Sangue'),
    ('Consumo de Álcool', 'Nível de Açúcar no Sangue'),
    ('Nível de Açúcar no Sangue', 'Diabetes'),
    ('IMC', 'Diabetes'),
    ('Histórico Familiar de Diabetes', 'Diabetes'),
    ('Atividade Física', 'Diabetes'),
    ('Consumo de Álcool', 'Diabetes')
])

data = pd.DataFrame({
    'Idade': [45, 50, 34, 60, 40, 55, 42, 37, 48, 53],
    'IMC': [25, 30, 22, 28, 24, 32, 26, 23, 29, 27],
    'Pressão Arterial': [80, 85, 75, 90, 80, 88, 77, 84, 82, 89],
    'Histórico Familiar de Diabetes': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'Atividade Física': [3, 2, 5, 1, 4, 2, 5, 3, 4, 2],  
    'Consumo de Álcool': [2, 1, 0, 2, 1, 2, 0, 1, 1, 0],  
    'Nível de Açúcar no Sangue': [100, 120, 90, 130, 110, 140, 95, 115, 105, 125],
    'Diabetes': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

network.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(network)

result = inference.map_query(variables=['Diabetes'], evidence={
    'Idade': 50,
    'IMC': 30,
    'Pressão Arterial': 85,
    'Histórico Familiar de Diabetes': 1,
    'Atividade Física': 2,
    'Consumo de Álcool': 1,
    'Nível de Açúcar no Sangue': 120
})

print("Probabilidade de Diabetes:", result)

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

network = BayesianNetwork([
    ('Idade', 'Nível de Açúcar no Sangue'),
    ('IMC', 'Nível de Açúcar no Sangue'),
    ('Pressão Arterial', 'Nível de Açúcar no Sangue'),
    ('Histórico Familiar de Diabetes', 'Nível de Açúcar no Sangue'),
    ('Atividade Física', 'Nível de Açúcar no Sangue'),
    ('Consumo de Álcool', 'Nível de Açúcar no Sangue'),
    ('Nível de Açúcar no Sangue', 'Diabetes'),
    ('IMC', 'Diabetes'),
    ('Histórico Familiar de Diabetes', 'Diabetes'),
    ('Atividade Física', 'Diabetes'),
    ('Consumo de Álcool', 'Diabetes')
])
network.fit(train_data, estimator=MaximumLikelihoodEstimator, states={'Diabetes': [0, 1]})

inference = VariableElimination(network)
correct_predictions = 0
for _, row in test_data.iterrows():
    evidence = row.drop('Diabetes').to_dict()
    true_label = row['Diabetes']
    predicted = inference.map_query(
        variables=['Diabetes'], evidence=evidence
    )['Diabetes']
    if predicted == true_label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)
print("Precisão do modelo:", accuracy)
