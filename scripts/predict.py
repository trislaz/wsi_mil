from wsi_mil.deepmil.predict import predict, load_model
# Imports
import os
import torch
import pandas as pd
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('--models_folder', type=str, required=True)
    parser.add_argument('--encoded_wsi_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = glob(os.path.join(args.models_folder, '*.pt.tar'))
    os.makedirs(args.output_folder, exist_ok=True)
    dfs = []
    for model_path in tqdm(models):
        df = predict_one_model(model_path, args.encoded_wsi_folder, device)
        df['model'] = os.path.basename(model_path)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(args.output_folder, 'predictions_all.csv'), index=False)
    df = df.drop('model', axis=1).groupby(['name', 'class']).mean().reset_index()
    df.to_csv(os.path.join(args.output_folder, 'predictions.csv'), index=False)
    print('Done')
    
def predict_one_model(model_path, encoded_wsi_folder, device):
    model = load_model(model_path, device)
    tc = model.label_encoder.classes_
    results = predict(model, encoded_wsi_folder)
    names = results['name']
    probas = results['proba']
    df_dict = {'name': [], 'class' : [], 'proba': []}
    for name, proba_vec in zip(names, probas):
        for class_idx, class_label in enumerate(tc):
            df_dict['name'].append(name)
            df_dict['class'].append(f'label_{class_label}')
            df_dict['proba'].append(proba_vec[0][class_idx])
    df = pd.DataFrame(df_dict)
    return df

if __name__ == '__main__':
    main()


