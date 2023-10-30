import subprocess

def run_annovar(input_file, output_file, annovar_dir, build_ver="hg19"):
    annotate_variation = annovar_dir + "/annotate_variation.pl"
    command = [
        annotate_variation,
        "-out", output_file,
        "-build", build_ver,
        input_file,
        annovar_dir + "/humandb/"
    ]
    subprocess.run(command)

run_annovar("your_input_file.txt", "your_output_file", "/path/to/annovar")


import requests

def get_pathways_for_genes(gene_list):
    url = "https://reactome.org/ContentService/data/query/enhanced/low/diagram/entity/LLP/"
    headers = {
        "Accept": "application/json"
    }
    pathways = []
    for gene in gene_list:
        response = requests.get(url + gene, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                pathways.append(item['displayName'])
    return set(pathways)

genes = ["LRFN5", ...]  # 你的基因列表
pathways = get_pathways_for_genes(genes)
print(pathways)


import torch.nn as nn

class CombinedNN(nn.Module):
    def __init__(self, num_genes, num_clinical_features):
        super(CombinedNN, self).__init__()
        
        input_size = num_genes + num_clinical_features
        hidden_size = ... # define as needed
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Assuming `genes_data` is the matrix of genetic data and `clinical_data` is the matrix of clinical features
weighted_genes = genes_data * 0.5
clinical_data[:, 0] *= 0.1  # Assuming 0th column is "weight gain"
clinical_data[:, 1] *= 0.1  # Assuming 1st column is "loss of interest"
clinical_data[:, 2] *= 0.15  # Assuming 2nd column is "insomnia"
clinical_data[:, 3] *= 0.15  # Assuming 3rd column is "hypersomnia"

combined_data = torch.cat((weighted_genes, clinical_data), dim=1)
