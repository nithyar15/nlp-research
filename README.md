# NLP Research: Do Adaptive Dynamic Retrieval Augment LLMs Really Work?
This repository contains code for a retrieval-augmented generation (RAG) pipeline enhanced with hallucination detection. The system uses LLaMA-3 or similar LLMs to iteratively generate answers to complex questions, while fetching supporting evidence from an ElasticSearch index based on hallucination-triggering mechanisms like token logprobs or attention weights.

## 📁 Contents

- `generate.py`: Core logic for different RAG strategies (basic, token-aware, attention-weighted, etc.)
- `main.py`: Main script to execute inference with config-based setup
- `evaluate.py`: Evaluation script for computing EM, F1, precision, recall, etc.
- `prep_elastic.py`: Indexes corpus passages into ElasticSearch
- `FLARE_100_samples.json`, `config_new.json`: Sample config and input files

## 📥 Download the IIRC Dataset

To download and prepare the IIRC dataset, run the following commands:

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```


## 🚀 How to Run (on a Unity server or SLURM-like environment)

Replace paths like `/path/to/your/...` with actual file locations.

```bash
# Step 1: Start a SLURM session
srun --partition=superpod-a100 --gres=gpu:1 --mem=40G --cpus-per-task=4 --time=04:00:00 --pty bash

# Step 2: Launch Elasticsearch
cd /path/to/elasticsearch-7.17.9
export ES_JAVA_OPTS="-Xms1g -Xmx1g"
rm -rf data/nodes
nohup bin/elasticsearch > ~/es.log 2>&1 &

# Step 3: Activate environment
conda activate dragin
cd /path/to/dragin-ciscoproj/src

# Step 4: Check cluster health
curl -s http://localhost:9200/_cluster/health

# Step 5: Index the data
python prep_elastic.py --data_path /path/to/iirc-beir/corpus.tsv --index_name iirc

# Step 6: Verify index
curl -X GET "localhost:9200/_cat/indices?v"

# Step 7: Run the main script
python -u main.py -c /path/to/ablation_config.json > ~/train.log 2>&1 &
