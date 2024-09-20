# Documentations

English | [Origin](./README_origin.md) 


### ABout the dataset

The dataset in use are `inspired` and `redial` datasets which adapted from the data provided by [`CRSLab`](https://github.com/RUCAIBox/CRSLab/tree/main)

| File Name | Description | Example |
| -- | -- | -- |
|  `entity2id.json`  |  The mapping from the movie names (in `DBPedia` or `IMDB` format) to item ids. |  `{"<http://dbpedia.org/resource/Hoffa_(film)>": 0}`  |
|  `item_ids.json`  | A list of all the item ids.   |  `[0, 2049, 16388, 12292, 6, 4109, ...]` |
|  `movies_ids.json`  |  A list of movies have been converted to ID which exist in database (dataset)  |  `[5, 8, 16393, 16399, 16, 16402, 28, 16420, 16425, 16426, 42, 41 ....]`  |
|  `relation2id.json`  |  A relationship of a movie itself to other existing entities  |  `{"<http://dbpedia.org/ontology/subsequentWork>": 0, "<http://dbpedia.org/ontology/genre>": 1 ...}`  |
|  `dbpedia_subkg.json`  |  A knowledge graph that connect a movie that cover which relationship to a correspond entity  |  `{"7838": [[24, 7838], [7, 17668] , ....` |
|  `train/val/test_data_processed.json`  |  Dataset for training, evaluation of sequential dialogs  |  `{"context": [], "resp": "", "rec": [] , "entity": []`  |

### Training full process
The training process stay the same with the original source code but we would make it easier as on a bash script for short and convinient

```bash
#Preprocessing
cd InferConverRec/data
!cp -r ./inspired ../src/data/
%!cd src
!python data/inspired/process.py

#Promt Training
!bash train_pre.sh

# Mask movie name for conversational training process
!python data/inspired/process_mask.py  
!cp -r ../data/inspired ./data/
cd src

#Conversational Training
!bash train_conv.sh
!bash infer_conv.sh

cd src
cp -r data/inspired/. data/inspired_gen/
!python data/inspired_gen/merge.py --gen_file_prefix lr1e3 ## adapt your own dataset save in data/inspired or data/

#Recommendation training
!bash train_rec.sh
```

### Chitchat Inferencing.
We provide two `chat_utils.py` and `chat_utils2.py` in `src` folder, whereas the first one do baseline process is generating an answer if there exists a entity movie name. The second one covers a more complex scenario when a user can mistyping entities or did not provide any item in the chat.

In the end run `chatbot_demo.py` for testing it on gradio




