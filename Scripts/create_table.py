import os
import re
import pandas as pd

def find_result_files(start_path):
    result_files = []
    for root_dir in os.listdir(start_path):
        root_path = os.path.join(start_path, root_dir)
        if os.path.isdir(root_path):
            for sub1_dir in os.listdir(root_path):
                sub1_path = os.path.join(root_path, sub1_dir)
                if os.path.isdir(sub1_path) and os.path.isfile(os.path.join(sub1_path,'result.txt')):
                    result_files.append(os.path.join(sub1_path,'result.txt'))

    return result_files

def extract_accuracies(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'MEAN IoU: (\d+\.\d+)% \(Model Original\) --- MEAN IoU: (\d+\.\d+)%', line)
            if match:
                return float(match.group(1)), float(match.group(2))
    return None, None

def generate_accuracy_table(result_files):
    data = {'Model Original': [], 'Model Pruned': []}
    print(str(result_files))
    for file in result_files:
        orig_acc, pruned_acc = extract_accuracies(file)
        if orig_acc is not None and pruned_acc is not None:
            data['Model Original'].append(orig_acc)
            data['Model Pruned'].append(pruned_acc)

    df = pd.DataFrame(data)
    return df


path_project = "./"
if os.path.exists('/content/AMLProjectBase'):
    path_project = '/content/AMLProjectBase/'
if os.path.basename(os.getcwd()) != "Scripts":
    os.chdir(f"{path_project}Scripts")
# Esempio di utilizzo:
start_path = '../trained_models/modelPrunedCompleted'  # Modificare con il percorso corretto
result_files = find_result_files(start_path)
accuracy_table = generate_accuracy_table(result_files)
print(accuracy_table)
