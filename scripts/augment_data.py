import os

from src.aug.ssmba import gen_neighborhood

def main():
    # Get the augmentation path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    pos_text_path = os.path.join(cur_dir, "..", "..", "data", "rt-polaritydata", "pos.txt")
    neg_text_path = os.path.join(cur_dir, "..", "..", "data", "rt-polaritydata", "neg.txt")
    # Augment data
    gen_neighborhood()

if __name__ == "__main__":
    main()

