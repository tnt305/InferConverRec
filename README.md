# Documentations

English | [Origin](./README_origin.md) 


### ABout the dataset

We uploaded the test data in `data/` folder, which can be used to form the prompt to query different LLMs. The `inspired` and `redial` datasets are adapted from the data provided by [`CRSLab`](https://github.com/RUCAIBox/CRSLab/tree/main), where we added some additional data fields like `is_user`.

| File Name | Description | Example |
| -- | -- | -- |
|  `entity2id.json`  |  The mapping from the movie names (in `DBPedia` or `IMDB` format) to item ids. |  `{"<http://dbpedia.org/resource/Hoffa_(film)>": 0}`  |
|  `item_ids.json`  | A list of all the item ids.   |  `[0, 2049, 16388, 12292, 6, 4109, ...]` |
|  `movies_ids.json`  |  A list of movies have been converted to ID which exist in database (dataset)  |  `[5, 8, 16393, 16399, 16, 16402, 28, 16420, 16425, 16426, 42, 41 ....]  |
|  `relation2id.json`  |  A relationship of a movie itself to other existing entities  |  `{"<http://dbpedia.org/ontology/subsequentWork>": 0, "<http://dbpedia.org/ontology/genre>": 1 ...}`  |
|  `dbpedia_subkg.json`  |  A knowledge graph that connect a movie that cover which relationship to a correspond entity  |  `{"7838": [[24, 7838], [7, 17668] , .... |
|  `train/val/test_data_processed.json`  |  Dataset for training, evaluation of sequential dialogs  |  `{"context": [], "resp": "", "rec": [] , "entity": []`  |
