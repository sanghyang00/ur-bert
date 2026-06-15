import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans as SklearnKMeans
import wandb  

from dataset import KMeansDataset 

class StreamingKMeans(nn.Module):
    def __init__(self, num_clusters, embedding_size=1024, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_clusters = num_clusters
        self.embedding_size = embedding_size
        self.decay = decay
        self.eps = eps
        
        self.register_buffer('cluster_size', torch.zeros(num_clusters))
        self.register_buffer('cluster_sum', torch.zeros(num_clusters, embedding_size))
        self.register_buffer('centroids', torch.randn(num_clusters, embedding_size))
        self.register_buffer('is_initialized', torch.tensor(0, dtype=torch.uint8))

    def set_initial_centroids(self, centroids_data):
        if centroids_data.shape != (self.num_clusters, self.embedding_size):
            raise ValueError("Shape mismatch")
        self.centroids.data.copy_(centroids_data.to(self.centroids.device))
        self.cluster_sum.data.copy_(self.centroids.data)
        self.cluster_size.fill_(1)
        self.is_initialized.fill_(1)
        print(">> StreamingKMeans initialized with external data.")

    def _preprocess(self, x, lengths):
        batch_size, max_len, _ = x.shape
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        flat_x = x[mask]
        return flat_x

    def _compute_distance_and_assign(self, flat_x):
        with torch.cuda.amp.autocast(enabled=False):
            flat_x_f32 = flat_x.float()
            centroids_f32 = self.centroids.float()
            dists = torch.cdist(flat_x_f32, centroids_f32, p=2).pow(2)
            min_dists, assignments = torch.min(dists, dim=1)
        return min_dists, assignments

    def _update_ema(self, flat_x, assignments):
        one_hot = F.one_hot(assignments, self.num_clusters).type_as(flat_x)
        curr_counts = one_hot.sum(0)
        curr_sums = torch.matmul(one_hot.t(), flat_x)
        
        self.cluster_size.mul_(self.decay).add_(curr_counts, alpha=1 - self.decay)
        self.cluster_sum.mul_(self.decay).add_(curr_sums, alpha=1 - self.decay)
        
        n = self.cluster_size.sum()
        smoothed_counts = (self.cluster_size + self.eps) / (n + self.num_clusters * self.eps) * n
        self.centroids.data.copy_(self.cluster_sum / smoothed_counts.unsqueeze(1))

    def forward(self, x, lengths):
        if self.is_initialized.item() == 0:
             flat_x = self._preprocess(x, lengths)
             self._random_init(flat_x)
        
        flat_x = self._preprocess(x, lengths)
        min_dists, assignments = self._compute_distance_and_assign(flat_x)
        
        if self.training:
            self._update_ema(flat_x, assignments)
            
        return min_dists, assignments
        
    def _random_init(self, flat_x):
        n_data = flat_x.size(0)
        if n_data < self.num_clusters:
            indices = torch.randint(0, n_data, (self.num_clusters,)).to(flat_x.device)
        else:
            indices = torch.randperm(n_data)[:self.num_clusters].to(flat_x.device)
        self.centroids.data.copy_(flat_x[indices])
        self.cluster_size.fill_(1)
        self.cluster_sum.data.copy_(self.centroids.data)
        self.is_initialized.fill_(1)
        
    def get_perplexity(self):
        probs = self.cluster_size / self.cluster_size.sum()
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
        return perplexity.item()

    def get_dead_codebook_ratio(self, threshold=1.0):
        dead_count = (self.cluster_size < threshold).float().sum()
        return (dead_count / self.num_clusters).item()

def collate_fn(batch):
    inputs, input_lengths = [], []
    for audio in batch:
        if audio.dim() == 2: audio = audio.squeeze(0)
        inputs.append(audio)
        input_lengths.append(audio.shape[-1])

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    return inputs, input_lengths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--num_clusters', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    
    # 1. WandB Init
    wandb.init(
        project='URBERT (Preprocessing)',
        name=f'Streaming KMeans (n_clusters={args.num_clusters})',
    )

    # 2. Load Model
    print(f">> Loading Wav2Vec2 from {args.ckpt_path}...")
    model = torchaudio.models.wav2vec2_xlsr_300m().to(args.device)
    if os.path.exists(args.ckpt_path):
        model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    model.eval() 

    # 3. Datasets
    dataset = KMeansDataset(args.data_dir, split='train', max_samples_per_lang=1000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    
    dataset_for_init = KMeansDataset(args.data_dir, split='train', max_samples_per_lang=30)
    dataloader_for_init = DataLoader(dataset_for_init, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    
    kmeans = StreamingKMeans(args.num_clusters, embedding_size=args.embedding_size).to(args.device)
    
    # --- Phase 1: Init ---
    print(">> Phase 1: Initialization...")
    init_feats_list = []
    with torch.no_grad():
        for i, (audio, audio_lengths) in enumerate(tqdm(dataloader_for_init, desc="Init Collection")):
            audio, audio_lengths = audio.to(args.device), audio_lengths.to(args.device)
            features, feat_lengths = model(audio, audio_lengths)
            
            B, T, C = features.shape
            mask = torch.arange(T, device=features.device).expand(B, T) < feat_lengths.unsqueeze(1)
            init_feats_list.append(features[mask].cpu())

    if len(init_feats_list) > 0:
        all_init_feats = torch.cat(init_feats_list, dim=0).numpy()
        print(f"   Collected {all_init_feats.shape[0]} vectors. Running K-Means++...")
        sklearn_kmeans = SklearnKMeans(n_clusters=args.num_clusters, n_init=1, max_iter=10)
        sklearn_kmeans.fit(all_init_feats)
        kmeans.set_initial_centroids(torch.from_numpy(sklearn_kmeans.cluster_centers_).float())
    else:
        print("!! Warning: Random Init will be used.")

    # --- Phase 2: Training ---
    print(">> Phase 2: Streaming Training...")
    kmeans.train()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global_step = 0
    
    for epoch in range(args.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        total_dist = 0.0
        total_tokens = 0
        
        epoch_assignments = []

        for batch_idx, (audio, audio_lengths) in enumerate(pbar):
            audio, audio_lengths = audio.to(args.device), audio_lengths.to(args.device)
            
            with torch.no_grad():
                features, feat_lengths = model(audio, audio_lengths)
            
            min_dists, assignments = kmeans(features, feat_lengths)
            
            curr_dist = min_dists.sum().item()
            curr_tokens = min_dists.numel()
            total_dist += curr_dist
            total_tokens += curr_tokens
            avg_dist = curr_dist / (curr_tokens + 1e-9)

            if batch_idx % 10 == 0: 
                epoch_assignments.append(assignments.detach().cpu())

            if global_step % args.log_interval == 0:
                perplexity = kmeans.get_perplexity()
                dead_ratio = kmeans.get_dead_codebook_ratio()
                
                wandb.log({
                    "train/loss": avg_dist,
                    "train/perplexity": perplexity,
                    "train/dead_codebook_ratio": dead_ratio,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step
                })
                
            pbar.set_postfix({'AvgDist': total_dist / (total_tokens + 1e-5), 'PPL': kmeans.get_perplexity()})
            global_step += 1
        
        # --- Epoch End Logging ---
        if len(epoch_assignments) > 0:
            all_epoch_assigns = torch.cat(epoch_assignments).numpy()
            wandb.log({
                "viz/assignment_histogram": wandb.Histogram(all_epoch_assigns, num_bins=args.num_clusters),
                "epoch": epoch + 1
            })
            
        torch.save(kmeans.state_dict(), os.path.join(args.output_dir, f'kmeans_ep{epoch+1}.pt'))

    wandb.finish()
    print(">> Training Complete.")

if __name__ == "__main__":
    main()
