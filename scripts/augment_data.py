import os

from src.augmentation.ssmba import gen_neighborhood

def main():
    # Get the augmentation path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    polarity_path = os.path.join(cur_dir, "..", "data", "rt-polaritydata")
    # Augment data
    # Strightforward for the sentiment analysis!
    for aug_file in ["neg", "pos"]:
        gen_neighborhood(
            in_file=os.path.join(polarity_path, f"{aug_file}.txt"),
            output_path=os.path.join(polarity_path, f"{aug_file}_augmented"),
            model="bert-base-uncased",
            tokenizer="bert-base-uncased",
            num_samples=2,
            noise_prob=0.15,
            topk=10,
            seed=42
        )

if __name__ == "__main__":
    main()

