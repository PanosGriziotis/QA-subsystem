import argparse
import json
import random
import os
from haystack.utils import SquadData

def split_squad_dataset (filepath, split_ratio: int = 0.1):

    with open(filepath, encoding="utf-8") as f:

        # load and count total num of examples
        data = json.load(f)
        num_of_examples =  SquadData (data).count()

        # shuffle examples
        data = data["data"]
        random.shuffle(data)
        counter = 0
        test_set = []

        for article in data:
            
            for paragraph in article["paragraphs"]:
                counter += (len(paragraph["qas"]))

            if counter >= round (num_of_examples * split_ratio):
                break
            else:
                test_set.append (article)
                    
        train_set = {"data" : data [len(test_set):]}
        test_set = {"data" : test_set}

    print (f"train set instances: {(num_of_examples-counter)}\n dev set instances: {counter}")
    
    # Write datasets
    path = os.path.dirname (filepath)

    with open(os.path.join(path, "train_file.json"), 'w') as train_file:
        json.dump(train_set, train_file, ensure_ascii=False, indent=4)

    with open(os.path.join(path,"dev_file.json"), 'w') as dev_file:       
        json.dump (test_set, dev_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument ('-f', type = str, required=True)
    args = parser.parse_args()
    split_squad_dataset(args.f)