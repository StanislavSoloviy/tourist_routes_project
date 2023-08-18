import os
import main
import random

def create_db(name_db, path):
    result = [(str(direct) + "/" +  str(file), main.forward_dict[direct])
              for direct in os.listdir(path)
              for file in os.listdir(path + '/' + direct) ]
    random.shuffle(result)
    with open(name_db + "_x.txt", "w", encoding="utf-8") as x_file:
        with open(name_db + "_y.txt", "w", encoding="utf-8") as y_file:
            for raw in result[:len(result)-1]:
                x_file.write(str(raw[0]) + "\n")
                y_file.write(str(raw[1]) + "\n")
            x_file.write(str(raw[0]))
            y_file.write(str(raw[1]))
    return None

create_db("train", "Landscape_classifier/training")
create_db("test", "Landscape_classifier/testing")

