"""Full orchestrator pipeline that runs normalize -> train -> embed -> cluster in sequence."""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("full_pipeline")


def run_module(module_name: str, args_list: list = None) -> str:
    cmd = [sys.executable, "-m", module_name]
    if args_list:
        cmd.extend(args_list)
        
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")
    
    # Run process and capture stdout/stderr
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        logger.error(f"Process {module_name} failed with exit code {res.returncode}")
        logger.error(f"STDOUT:\n{res.stdout}")
        logger.error(f"STDERR:\n{res.stderr}")
        raise RuntimeError(f"Subprocess failed: {cmd_str}")
        
    return res.stdout


def main():
    parser = argparse.ArgumentParser(description="GhostSig Full Processing Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training and use checkpoints")
    parser.add_argument("--once", action="store_true", help="Run normalization once (rather than continuous)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for training")
    args = parser.parse_args()

    logger.info("==============================================")
    logger.info("         STARTING FULL GHOSTSIG PIPELINE      ")
    logger.info("==============================================")

    # 1. Normalize
    logger.info("[Step 1/5] Normalizing events & updating fingerprints...")
    norm_args = ["--once"] if args.once else []
    run_module("pipeline.normalize", norm_args)

    # 2. Train Models if not skipped
    if not args.skip_training:
        logger.info("[Step 2a/5] Training Temporal Encoder...")
        run_module("ml.train_temporal", ["--epochs", "5", "--device", args.device, "--batch-size", "8"])
        
        logger.info("[Step 2b/5] Training Entropy Encoder...")
        run_module("ml.train_entropy", ["--epochs", "5", "--device", args.device, "--batch-size", "8"])
        
        logger.info("[Step 2c/5] Training Fusion Encoder...")
        run_module("ml.train_fusion", ["--epochs", "5", "--device", args.device, "--batch-size", "8"])
    else:
        logger.info("[Step 2/5] Skipping model training. Using existing checkpoints.")

    # 3. Embed Modal Features
    logger.info("[Step 3a/5] Generating Temporal Embeddings...")
    run_module("ml.embed_temporal")

    logger.info("[Step 3b/5] Generating Entropy Embeddings...")
    run_module("ml.embed_entropy")

    # 4. Embed Fusion Representation
    logger.info("[Step 4/5] Running Fusion Encoder...")
    run_module("ml.embed_fusion")

    # 5. Run UMAP + HDBSCAN Clustering
    logger.info("[Step 5/5] Running Clustering & Campaign Attribution...")
    clustering_out = run_module("ml.run_clustering")

    # Parse metrics from run_clustering stdout
    accounts_processed = "N/A"
    n_clusters = "N/A"
    noise_fraction = "N/A"
    silhouette = "N/A"

    for line in clustering_out.splitlines():
        if "Total accounts clustered:" in line:
            accounts_processed = line.split(":")[-1].strip()
        elif "Number of clusters found:" in line:
            n_clusters = line.split(":")[-1].strip()
        elif "Noise fraction:" in line:
            noise_fraction = line.split(":")[-1].strip()
        elif "Silhouette score (non-noise):" in line:
            silhouette = line.split(":")[-1].strip()

    # Print Summary Table
    print("\n" + "=" * 50)
    print("             GHOSTSIG RUN SUMMARY             ")
    print("=" * 50)
    print(f"Accounts Processed:      {accounts_processed}")
    print(f"Clusters Found:          {n_clusters}")
    print(f"Noise Fraction:          {noise_fraction}")
    print(f"Silhouette Score:        {silhouette}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
