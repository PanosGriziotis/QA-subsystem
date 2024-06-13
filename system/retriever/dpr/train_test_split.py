import random
import json
import os
import argparse


def train_dev_split(filepath, dev_split):

    with open (filepath, 'r') as file:
        data = json.load(file)
        random.shuffle(data)
        dev_samples = len(data) * dev_split
        train_samples = round(len(data) - dev_samples)
        train_set = data[:train_samples]
        dev_set = data[train_samples:] 

        # count how many instances are per dataset  

        print (f"train set instances: {len(train_set)})\n dev set instances: {len(dev_set)}")
        path = os.path.dirname (filepath)
        with open(os.path.join(path, f"train_{os.path.basename(filepath)}"), 'w') as train_file:
            json.dump(train_set, train_file, ensure_ascii=False, indent=4)
        with open(os.path.join(path,f"dev_{os.path.basename(filepath)}"), 'w') as dev_file:       
            json.dump (dev_set, dev_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument ('--filepath', type = str, required=True)
    parser.add_argument('--dev_split_ratio', type = float,  default = 0.2 )
    args = parser.parse_args()

    train_dev_split (args.filepath, args.dev_split_ratio)