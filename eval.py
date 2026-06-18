# cook your dish here
import os
import torch
import numpy as np
import sklearn.preprocessing
from tqdm import tqdm
from eval.verification import load_bin
from backbones import get_model
from sklearn.metrics import roc_curve, auc
import time

def calculate_fnmr_at_fmr(embeddings1, embeddings2, issame, fmr_targets):

    scores = np.sum(embeddings1 * embeddings2, axis = 1)
    print(f"[DEBUG ROC]")
    print(f" Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    print(f" #Pairs: {len(issame)}, #Genuine: {np.sum(issame)}, #Impostor: {len(issame)-np.sum(issame)}")

    
    # Calculate ROC curve: FPR (FMR), TPR, thresholds
    fpr, tpr, thresholds = roc_curve(issame, scores, pos_label=1)
    print(f" FPR range: {fpr.min():.6f} to {fpr.max():.6f}")
    print(f" TPR range: {tpr.min():.6f} to {tpr.max():.6f}")

    
    print(f" FPR: min={fpr.min():.6f}, max={fpr.max():.6f}, unique values={len(np.unique(fpr))}")
    print(f" FPR targets: {fmr_targets}")

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    thresholds = np.flipud(thresholds)

    fnmr = 1-tpr

    eer_idx = np.argmin(np.abs(fpr - fnmr))
    eer = (fpr[eer_idx] + fnmr[eer_idx])/2

    predictions = scores > thresholds[eer_idx]
    accuracy = np.mean(predictions==issame)

    auc_score = auc(fpr, tpr)

    fnmr_at_fmr = {}
    for fmr_target in fmr_targets:
        # Find closest FPR to target FMR
        diff = np.abs(fpr - fmr_target)
        idx = np.argmin(diff)
        actual_fmr = fpr[idx]
        fnmr_at_fmr[fmr_target] = {
            'fnmr': fnmr[idx],
            'actual_fmr': actual_fmr
        }
        
    results = {
        'auc': auc_score,
        'eer': eer,
        'accuracy': accuracy,
        'fnmr@fmr=1e-1': fnmr_at_fmr[1e-1]['fnmr'],
        'fnmr@fmr=1e-2': fnmr_at_fmr[1e-2]['fnmr'],
        'fnmr@fmr=1e-3': fnmr_at_fmr[1e-3]['fnmr'],
        'actual_fmr_1e-1': fnmr_at_fmr[1e-1]['actual_fmr'],
        'actual_fmr_1e-2': fnmr_at_fmr[1e-2]['actual_fmr'],
        'actual_fmr_1e-3': fnmr_at_fmr[1e-3]['actual_fmr'],
    }
    
    return results


def extract_embeddings(model, data_list, batch_size):
    """Extract and normalize embeddings"""
    embeddings_list = []
    
    for data in data_list:
        embeddings = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            img = ((batch / 255) - 0.5) / 0.5
            
            if torch.cuda.is_available():
                img = img.cuda()
            
            with torch.no_grad():
                emb = model(img).cpu().numpy()
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        embeddings_list.append(embeddings)
    
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    return embeddings


def evaluate_model_on_dataset(model, bin_path, batch_size, fmr_targets, log_file):
    """Evaluate one model on one dataset"""
    image_size = [112, 112]
    start_time = time.time()
    print(f"Loading data from {bin_path}")
    data_list, issame_list = load_bin(bin_path, image_size)
    end_time = time.time()
    print(f"Loading data {bin_path} in {end_time-start_time}s")
    print(f"Len of data list {len(data_list)}")
    print(f"Len of issame list {len(issame_list)}")
    issame = np.array(issame_list)
    num_pairs = len(issame_list)

    print(f"Number of Face Pairs: {num_pairs}")
    embeddings = extract_embeddings(model, data_list, batch_size)
    
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    
    results = calculate_fnmr_at_fmr(embeddings1, embeddings2, issame, fmr_targets)
    
    return results


def evaluate_all_models(network, models_root, data_dir, fmr_targets=[1e-1, 1e-2, 1e-3], batch_size=64, log_path="evaluation.txt"):
    """
    Evaluate all models in folder structure
    
    Args:
        models_root: root folder containing subfolders with .pt files
        data_dir: directory containing .bin files
        fmr_targets: list of FMR values
        batch_size: batch size for inference
    """
    # Find all backbone folders
    pt_files_in_root = [f for f in os.listdir(models_root) if f.endswith('.pt')]
    if not pt_files_in_root:
        print("Evaluate all backbones")
        with open(log_path, 'w') as log_file:
            print(f"Log path: {log_path}")
            log_header = (f"="*90 + "\n"
                     f"Face Recognition Evaluation - FNMR @ FMR\n"
                     f"Models Root: {models_root}\n"
                     f"Data Directory: {data_dir}\n"
                     f"FMR Targets: {fmr_targets}\n"
                     f"Batch Size: {batch_size}\n"
            )
            log_file.write(log_header)
            backbone_folders = sorted([f for f in os.listdir(models_root) 
                                    if os.path.isdir(os.path.join(models_root, f))])
            
            log_file.write(f"Backbone Folders: {backbone_folders}\n")
            
            if not backbone_folders:
                print(f"No subfolders found in {models_root}")
                return
            
            # Find all .bin files
            bin_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
            
            if not bin_files:
                print(f"No .bin files found in {data_dir}")
                return
            
            dataset_names = [f.replace('.bin', '') for f in bin_files]
            
            # Evaluate each backbone folder
            for backbone_name in backbone_folders:
                log_file.write(f"Backbone Name: {backbone_name}\n")
                backbone_path = os.path.join(models_root, backbone_name)
                
                # Find all .pt files in this folder
                model_files = sorted([f for f in os.listdir(backbone_path) if f.endswith('.pt')], reverse=True)
                
                if not model_files:
                    continue
                
                print(f"\n{'='*80}")
                print(f"BACKBONE: {backbone_name}\n")
                print(f"{'='*80}\n")


                # Evaluate each model
                for model_file in model_files:
                    model_path = os.path.join(backbone_path, model_file)
                    
                    print(f"Model: {model_file}")
                    log_file.write(f"Model: {model_file}")
                    # Load model
                    weights = torch.load(model_path)
                    model = get_model(network, dropout=0, fp16=False).cuda()
                    model.load_state_dict(weights)
                    model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()
                    
                    # Evaluate on all datasets
                    all_results = {}
                    
                    pbar = tqdm(bin_files, desc="Evaluating", ncols=100)
                    for bin_file in pbar:
                        dataset_name = bin_file.replace('.bin', '')
                        bin_path = os.path.join(data_dir, bin_file)
                        
                        pbar.set_postfix_str(dataset_name)
                        
                        results = evaluate_model_on_dataset(model, bin_path, batch_size, fmr_targets, log_file)
                        all_results[dataset_name] = results
                    
                    # Print results table
                    header = "Dataset:"
                    for fmr in fmr_targets:
                        header += f" {'FNMR@' + f'{fmr:.0e}':<12}"
                    header += '\n'

                    seperator = '-' * 80 + '\n'

                    print(header)
                    print('-' * 80)
                    log_file.write(header)
                    log_file.write(seperator)
                    
                    for dataset_name in dataset_names:
                        print(f"Result of dataset: {dataset_name}")
                        if dataset_name in all_results:
                            row = f"{dataset_name:<15}"
                            # for fmr in fmr_targets:
                            #     fnmr = all_results[dataset_name][fmr]['fnmr']
                            #     row += f" {fnmr:<12.6f}"
                            # print(row)
                            log_file.write(f"\n{row}")
                            log_file.write(f"\n{all_results[dataset_name]}")
                            log_file.write(f"\n")
                    print()
                    log_file.write('\n')
                    log_file.flush()
                    
                    # Cleanup
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all models in folder structure')
    parser.add_argument('--network', required=True, default = "vit_l_dp005_mask_005", help='Backbone')
    parser.add_argument('--models-root', required=True, help='Root folder containing backbone subfolders')
    parser.add_argument('--data-dir', required=True, help='Directory containing .bin files')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--fmr', type=str, default='1e-1,1e-2,1e-3', help='Comma-separated FMR targets')
    parser.add_argument('--log-path', type=str)
    args = parser.parse_args()
    
    fmr_targets = [float(x) for x in args.fmr.split(',')]
    
    evaluate_all_models(
        network = args.network,
        models_root=args.models_root,
        data_dir=args.data_dir,
        fmr_targets=fmr_targets,
        batch_size=args.batch_size,
        log_path=args.log_path
    )