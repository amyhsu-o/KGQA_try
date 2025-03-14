## Install

```bash
pip install -r requirements.txt
```

## Quick Start

Run 5 selected questions from CRAG dataset.

```bash
./example_run.sh
```

<details>
<summary>5 selected questions</summary>

|line_id|query|correct answer|
|---|---|---|
|2|where did the ceo of salesforce previously work?|marc benioff spent 13 years at oracle, before launching salesforce.|
|54|what was mike epps's age at the time of next friday's release?|29|
|92|what was the 76ers' record the year allen iverson won mvp?|56-26|
|98|what age did ferdinand magelan discovered the philippines|41 years old|
|272|what was taylor swifts age when she released her debut album?|16 years old|

</details>


## Usage

### KG construction

Run the script (`construction.py`) using:

```bash
python construction/construction.py --dataset <dataset> [optional arguments]

# Ex. python construction/construction.py --dataset crag --crag_line_id 2
```

Arguments: 

|Argument|Type|Description|
|---|---|---|
|`--dataset`|`str`|Should be either `"crag"` or `"wiki"`.|
|`--crag_line_id`|`int`|Required when `--dataset=crag`, unless `--crag_top_n` is provided. Specifies the specific multi-hop question to process. `crag_line_id` can be checked at `./data/multi-hops.txt`. |
|`--crag_top_n`|`int`|Required when `--dataset=crag`, unless `--crag_line_id` is provided. Specifies the top n multi-hop questions to process.|
|`--wiki_title`|`str`|Required when `--dataset=wiki`. Specifies the Wikipedia page title to process.|

### Query

Run the script (`ToG.py`) using:

```bash
python qa/ToG/ToG.py --path <result_path> --query <query>

# Ex. python qa/ToG/ToG.py --path "./question_2" --query "where did the ceo of salesforce previously work?"
```

Arguments:
|Argument|Type|Description|
|---|---|---|
|`--path`|`str`|Specifies the directory path to be processed. Path can be checked in [Logging](#logging).|
|`--query`|`str`|Specifies the search query to apply.|

### Logging

During execution, all records and output files will be stored under a dynamically generated path:

- if `--dataset=crag`, the path will be: `./question_<crag_line_id>`
- If `--dataset=wiki`, the path will be: `./<wiki_title_without_spaces>`

Below information will be generated during execution:

- `./{path}/construction`

    |File Name|Description|
    |---|---|
    |`chunks*.txt`|Chunks of given documents.|
    |`chunks_entities*.txt`|Entities extracted from chunks.|
    |`entities*.txt`|Entities extracted from chunks with entity type.|
    |`chunks_triples*.txt`|Triples extracted from chunks.|
    |`graph.html`|Resulted knowledge graph from all documents.|

- `./{path}/reasoning_path`

    |File Name|Description|
    |---|---|
    |`graph*.html`|Extracted subgraph from each round of ToG traversal.|

- `./{path}/log`

    |File Name|Description|
    |---|---|
    |`construction.log`|Log during KG construction.|
    |`reasoning_path.log`|Log during query.|

### Graph Visualization

All of the `graph*.html` file can be shown using:

```bash
python utils/show_graph.py --path <graph_path>

# Ex. python utils/show_graph.py --path ./question_2/construction/graph.html
```