import argparse
from preprocess import load_and_vectorize
from kmeans_gpu import kmeans_gpu

def main(args):
    with open(args.input_file, 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    X = load_and_vectorize(docs)
    labels, _ = kmeans_gpu(X, args.k, max_iter=args.iter)
    for i, label in enumerate(labels):
        print(f"Doc {i+1}: Cluster {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to .txt file with one document per line')
    parser.add_argument('--k', type=int, default=3, help='Number of clusters')
    parser.add_argument('--iter', type=int, default=100, help='Maximum number of iterations')
    args = parser.parse_args()
    main(args)
