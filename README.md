# Documentations

English | [Origin](./README_origin.md) 


### ABout the dataset

We uploaded the test data in `data/` folder, which can be used to form the prompt to query different LLMs. The `inspired` and `redial` datasets are adapted from the data provided by [`CRSLab`](https://github.com/RUCAIBox/CRSLab/tree/main), where we added some additional data fields like `is_user`.

| File Name | Description | Example |
| -- | -- | -- |
|  `entity2id.json`  |  The mapping from the movie names (in `DBPedia` or `IMDB` format) to item ids. |  `{"<http://dbpedia.org/resource/Hoffa_(film)>": 0}`  |
|  `item_ids.json`  | A list of all the item ids.   |  `[0, 2049, 16388, 12292, 6, 4109, ...]` |
|  `test.json`  |  Conversational context (in `input` field) and target recommendation item (in `rec` field)  |  `{"input": "System: How did you like Hustlers? It definitely has the drama aspect...\n User: I liked it ..." "rec": [9722]}`  |
|  `test_p2.json`  |  Similar to `test.json`, but we use historical mentioned movies only. P2 stands for **"Placeholder 2"**  |  `{"input": "System: Hustlers\n User:" "rec": [9722]}`  |
|  `test_p3.json`  |  Similar to `test.json`, but we use historical conversational text without mentioned movies. P3 stands for **"Placeholder 3"**  |  `{"input": "System: How did you like ? It definitely has the drama aspect...\n User: I liked it ..." "rec": [9722]}`  |
|  `test_p4.json`  |  Similar to `test.json`, but we use historical conversational text with randomly mentioned movies. P4 stands for **"Placeholder 4"**  |  `{"input": "System: How did you like Titanic? It definitely has the drama aspect...\n User: I liked it ..." "rec": [9722]}`  |
|  `test_raw.json`  |  Raw data file similar to the files provided in [`CRSLab`](https://github.com/RUCAIBox/CRSLab/tree/main)  |  `{"context": ["", ... "the last movie i saw in the theater was Hustlers . I generally like comedy, drama and documentaries"], "resp": "How did you like Hustlers? ...", "rec": [9722], "entity": [9722, 15748], "prev_entity": [15748, 17158, 8683, 8881, 16785, 9722], "dialog_id": "test_0", "turn_id": "test_4", "is_user": 0, "entity_name": ["drama", "Hustlers"]}`  |
