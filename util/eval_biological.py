import os
import csv
import sys
import torch
import argparse

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from scipy.stats import entropy, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.dab import DABExtractor

# Same set eval_quantitative.py matches on, so both scripts pair the same files.
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def compute_p90_scores(dab_maps):
    """Compute p90 DAB scores: mean of pixels >= 90th percentile.

    Args:
        dab_maps: [B, H, W] raw DAB intensity maps

    Returns:
        scores: numpy array [B]
    """
    scores = []

    for i in range(dab_maps.shape[0]):
        flat = dab_maps[i].flatten()
        p90 = torch.quantile(flat, 0.9)
        mask = flat >= p90
        scores.append(flat[mask].mean().item() if mask.sum() > 0 else flat.mean().item())

    return np.array(scores)

def compute_dab_metrics(gen_01, real_01, labels=None):
    """Compute DAB staining fidelity metrics.

    Args:
        gen_01: [N, 3, H, W] in [0, 1]
        real_01: [N, 3, H, W] in [0, 1]
        labels: [N] int class labels (optional, for per-class breakdown)

    Returns:
        dict with aggregate metrics,
        list of per-image dicts
    """
    dab_extractor = DABExtractor(device='cpu')

    dab_gen = dab_extractor.extract_dab_intensity(gen_01.float(), normalize="none")
    dab_real = dab_extractor.extract_dab_intensity(real_01.float(), normalize="none")

    gen_scores = compute_p90_scores(dab_gen)
    real_scores = compute_p90_scores(dab_real)

    agg = {}
    agg['dab_mae'] = float(np.mean(np.abs(gen_scores - real_scores)))

    if len(gen_scores) > 2:
        r, p_val = pearsonr(gen_scores, real_scores)
        agg['dab_pearson_r'] = float(r)

    # Per-pair DAB KL/JSD (ODA-GAN: 256-bin histogram)
    n_bins = 256
    eps = 1e-10
    pair_kls, pair_jsds = [], []
    per_image = []

    for i in range(dab_gen.shape[0]):
        g = dab_gen[i].flatten().numpy()
        r = dab_real[i].flatten().numpy()
        
        hist_range = (0, max(g.max(), r.max()) + 1e-6)
        hg, _ = np.histogram(g, bins=n_bins, range=hist_range, density=True)
        hr, _ = np.histogram(r, bins=n_bins, range=hist_range, density=True)
        hg = hg + eps; hr = hr + eps
        hg = hg / hg.sum(); hr = hr / hr.sum()
        
        kl = float(entropy(hg, hr))
        m = 0.5 * (hg + hr)
        jsd = float(0.5 * entropy(hg, m) + 0.5 * entropy(hr, m))
        pair_kls.append(kl)
        pair_jsds.append(jsd)

        per_image.append({'p90_gen': float(gen_scores[i]),
                          'p90_real': float(real_scores[i]),
                          'dab_mae': float(abs(gen_scores[i] - real_scores[i])),
                          'dab_kl': kl,
                          'dab_jsd': jsd,})

    agg['dab_kl'] = float(np.mean(pair_kls))
    agg['dab_kl_std'] = float(np.std(pair_kls))
    agg['dab_jsd'] = float(np.mean(pair_jsds))
    agg['dab_jsd_std'] = float(np.std(pair_jsds))

    # Per-class breakdown + ordering (BCI only)
    if labels is not None:
        class_names = {0: '0', 1: '1+', 2: '2+', 3: '3+'}
        within_rs = []
        
        for cls, name in class_names.items():
            mask = (labels == cls).numpy() if isinstance(labels, torch.Tensor) else (labels == cls)
            
            if mask.sum() > 0:
                agg[f'dab_mae_class_{name}'] = float(np.mean(np.abs(gen_scores[mask] - real_scores[mask])))
                
                if mask.sum() > 5:
                    r_cls, _ = pearsonr(gen_scores[mask], real_scores[mask])
                    agg[f'dab_pearson_r_class_{name}'] = float(r_cls)
                    within_rs.append(r_cls)

        if within_rs: agg['dab_pearson_r_within_class'] = float(np.mean(within_rs))

        # Ordering violations
        class_gen_means = {}
        for cls in range(4):
            mask = (labels == cls).numpy() if isinstance(labels, torch.Tensor) else (labels == cls)
            if mask.sum() > 0: class_gen_means[cls] = float(np.mean(gen_scores[mask]))
        
        ordered_pairs = [(3, 2), (3, 1), (3, 0), (2, 1), (2, 0), (1, 0)]
        violations, total_pairs = 0, 0
        
        for high_cls, low_cls in ordered_pairs:
            if high_cls in class_gen_means and low_cls in class_gen_means:
                total_pairs += 1
                if class_gen_means[high_cls] < class_gen_means[low_cls]: violations += 1

        agg['ordering_violations'] = violations
        agg['ordering_total_pairs'] = total_pairs

    return agg, per_image

def compute_iod_metrics(gen_01, real_01, labels=None):
    """Compute Integrated Optical Density metrics (Beer-Lambert).

    Args:
        gen_01: [N, 3, H, W] in [0, 1]
        real_01: [N, 3, H, W] in [0, 1]
        labels: [N] optional class labels

    Returns:
        dict with aggregate metrics,
        list of per-image dicts
    """
    gen_255 = (gen_01.clamp(0, 1) * 255.0).clamp(min=1.0)
    real_255 = (real_01.clamp(0, 1) * 255.0).clamp(min=1.0)

    od_gen = -torch.log10(gen_255 / 255.0)
    od_real = -torch.log10(real_255 / 255.0)

    miod_gen = od_gen.mean(dim=(1, 2, 3)).numpy()
    miod_real = od_real.mean(dim=(1, 2, 3)).numpy()
    iod_gen = od_gen.sum(dim=(1, 2, 3)).numpy()
    iod_real = od_real.sum(dim=(1, 2, 3)).numpy()

    alpha = 1.8
    fod_gen = od_gen.pow(alpha).mean(dim=(1, 2, 3)).numpy()
    fod_real = od_real.pow(alpha).mean(dim=(1, 2, 3)).numpy()

    agg = {}
    agg['miod_diff'] = float(np.mean(miod_gen) - np.mean(miod_real))
    agg['miod_abs_diff'] = float(np.mean(np.abs(miod_gen - miod_real)))
    agg['iod_diff_1e7'] = float((np.mean(iod_gen) - np.mean(iod_real)) / 1e7)
    agg['mfod_abs_diff'] = float(np.mean(np.abs(fod_gen - fod_real)))

    if len(miod_gen) > 2:
        r, _ = pearsonr(miod_gen, miod_real)
        agg['iod_pearson_r'] = float(r)

    per_image = []
    for i in range(len(miod_gen)):
        per_image.append({'miod_gen': float(miod_gen[i]),
                          'miod_real': float(miod_real[i]),
                          'miod_diff': float(miod_gen[i] - miod_real[i]),
                          'fod_gen': float(fod_gen[i]),
                          'fod_real': float(fod_real[i]),})

    # Per-class mIOD
    if labels is not None:
        class_names = {0: '0', 1: '1+', 2: '2+', 3: '3+'}

        for cls, name in class_names.items():
            mask = (labels == cls).numpy() if isinstance(labels, torch.Tensor) else (labels == cls)
            if mask.sum() > 0: agg[f'miod_diff_class_{name}'] = float(np.mean(miod_gen[mask]) - np.mean(miod_real[mask]))

    return agg, per_image


def main():
    parser = argparse.ArgumentParser(description='Biological Information Consistency Evaluation')
    
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory of generated IHC images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory of real IHC images')
    
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV path')
    parser.add_argument('--labels_csv', type=str, default=None, help='CSV with image_id and label columns (for per-class breakdown)')
    
    parser.add_argument('--skip_dab', action='store_true', help='Skip DAB metrics')
    parser.add_argument('--skip_iod', action='store_true', help='Skip IOD metrics')

    args = parser.parse_args()

    to_tensor = transforms.ToTensor()

    # Find matched image pairs. Matching is on the filename stem, so a .png
    # prediction pairs with a .jpg ground truth: inference writes PNG, while
    # MIST ships JPEG and BCI ships PNG. Globbing '*.png' on both sides found
    # nothing on MIST. eval_quantitative.find_matched_pairs does the same.
    def _index(dir_path):
        out = {}
        for f in os.listdir(dir_path):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS: out[stem] = os.path.join(dir_path, f)

        return out

    pred_map = _index(args.pred_dir)
    gt_map = _index(args.gt_dir)
    common = sorted(set(pred_map) & set(gt_map))

    pred_only = len(pred_map) - len(common)
    gt_only = len(gt_map) - len(common)

    if pred_only > 0: print(f"[Warning] {pred_only} pred images without matching GT")
    if gt_only > 0: print(f"[Warning] {gt_only} GT images without matching pred")

    pairs = [(pred_map[stem], gt_map[stem], stem) for stem in common]

    if not pairs:
        print("ERROR: No matching pairs found.")
        sys.exit(1)

    print(f"Found {len(pairs)} matched image pairs")

    # Load all images as [0, 1] tensors
    print("Loading images...")
    pred_list, gt_list, names = [], [], []

    for pf, gf, stem in tqdm(pairs, desc="Loading"):
        pred_list.append(to_tensor(Image.open(pf).convert('RGB')))
        gt_list.append(to_tensor(Image.open(gf).convert('RGB')))
        names.append(stem)

    pred_tensor = torch.stack(pred_list)  # [N, 3, H, W] in [0, 1]
    gt_tensor = torch.stack(gt_list)

    print(f"Loaded: {pred_tensor.shape}")

    # Load labels if provided
    labels = None
    if args.labels_csv:
        df = pd.read_csv(args.labels_csv)
        label_col = None

        for col in ['label', 'class', 'grade', 'her2_grade', 'score']:
            if col in df.columns:
                label_col = col; break

        # Handle string labels like '0', '1+', '2+', '3+'
        if label_col:
            score_map = {'0': 0, '1+': 1, '2+': 2, '3+': 3}
            df = df.set_index('image_id')
            label_list = []

            for name in names:
                if name in df.index:
                    raw = df.loc[name, label_col]
                    
                    if isinstance(raw, str) and raw in score_map: label_list.append(score_map[raw])
                    else:
                        try: label_list.append(int(raw))
                        except (ValueError, TypeError): label_list.append(-1)
                
                else: label_list.append(-1)
            
            labels = torch.tensor(label_list)
            print(f"Labels loaded: {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}")

    # Per-image results
    per_image = [{'img_name': n} for n in names]

    # DAB metrics
    dab_agg = {}
    if not args.skip_dab:
        print("\n--- DAB Metrics ---")
        dab_agg, dab_per = compute_dab_metrics(pred_tensor, gt_tensor, labels)
        
        for i, d in enumerate(dab_per): per_image[i].update(d)
        
        print(f"  DAB MAE:       {dab_agg['dab_mae']:.4f}")
        print(f"  DAB Pearson-r: {dab_agg.get('dab_pearson_r', float('nan')):.4f}")
        print(f"  DAB KL:        {dab_agg['dab_kl']:.4f} +/- {dab_agg['dab_kl_std']:.4f}")
        print(f"  DAB JSD:       {dab_agg['dab_jsd']:.4f} +/- {dab_agg['dab_jsd_std']:.4f}")
        
        if 'ordering_violations' in dab_agg: print(f"  Ordering:      {dab_agg['ordering_violations']}/{dab_agg['ordering_total_pairs']} violations")

    # IOD metrics
    iod_agg = {}
    if not args.skip_iod:
        print("\n--- IOD Metrics ---")
        iod_agg, iod_per = compute_iod_metrics(pred_tensor, gt_tensor, labels)
        
        for i, d in enumerate(iod_per): per_image[i].update(d)
        
        print(f"  mIOD diff:     {iod_agg['miod_diff']:.4f}")
        print(f"  mIOD abs diff: {iod_agg['miod_abs_diff']:.4f}")
        print(f"  IOD diff 1e7:  {iod_agg['iod_diff_1e7']:.3f}")
        print(f"  IOD Pearson-r: {iod_agg.get('iod_pearson_r', float('nan')):.4f}")

    # Save CSV. abspath first: a bare filename such as `bio_run.csv` has an
    # empty dirname, and os.makedirs('') raises FileNotFoundError. run_eval.sh
    # passes exactly that form. eval_quantitative.write_csv does the same.
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # Per-image columns
    per_cols = ['img_name']

    if not args.skip_dab: per_cols += ['p90_gen', 'p90_real', 'dab_mae', 'dab_kl', 'dab_jsd']
    if not args.skip_iod: per_cols += ['miod_gen', 'miod_real', 'miod_diff', 'fod_gen', 'fod_real']

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_cols, extrasaction='ignore')
        writer.writeheader()

        for row in per_image: writer.writerow({k: f"{row[k]:.6f}" if isinstance(row.get(k), float) else row.get(k, '') for k in per_cols})

        # Summary row
        summary = {'img_name': 'SUMMARY'}
        summary.update(dab_agg)
        summary.update(iod_agg)

        f.write('\n')
        f.write(f"# Aggregate Metrics\n")

        for k, v in {**dab_agg, **iod_agg}.items(): f.write(f"# {k}: {v}\n")

    print(f"\nSaved: {args.output_csv}")


if __name__ == '__main__':
    main()
