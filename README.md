# LLM Translation Quality Evaluation: Mandarin to English

A comprehensive research project for evaluating and comparing the translation quality of various Large Language Models (LLMs) and machine translation systems from Mandarin Chinese to English.

## Project Overview

This project analyzes the translation quality of multiple translation systems by comparing them against professional human translations using advanced NLP techniques including sentiment analysis and semantic similarity measurement.

### Translation Systems Evaluated

- **GPT-4**: OpenAI's GPT-4 model
- **GPT-4o**: OpenAI's GPT-4o (optimized) model
- **DeepSeek**: DeepSeek LLM
- **Google Translate**: Google's machine translation service
- **Expert Translation**: Professional human translations (baseline)

### Text Corpora

1. **Literary Works**
   - *A Dream of Red Mansions* (红楼梦) - Chapters 1-3
     - Expert Translation: H. Bencraft Joly
   - *Red Sorghum* (红高粱) - Chapters 1-4
     - Expert Translation: Howard Goldblatt

2. **News Articles**
   - Global Times news articles (8 articles)
   - English versions as expert baseline

## Key Features

### 1. Translation Quality Metrics (`evaluation/compute_mt_metrics.py`)
- **BERTScore** precision / recall / F1 against the expert translation (`roberta-large`)
- **COMET** (`Unbabel/wmt22-comet-da`) using source, output and expert reference
- Optional reference-free **CometKiwi** (`--kiwi`)
- **MPNet** cosine similarity (`all-mpnet-base-v2`) between each system output and the expert translation
- Paired **Wilcoxon signed-rank tests** between systems for MPNet and COMET (`evaluation/paired_comparisons.py`),
  Holm-adjusted, with rank-biserial effect sizes, pooled and per corpus
- Expert paragraphs are aligned to the source with a LaBSE-based monotonic aligner; scores are
  per aligned segment, aggregated with 95% bootstrap confidence intervals
- See [`evaluation/METRICS.md`](evaluation/METRICS.md) for exactly what every script computes

### 1b. Sentence-BERT Cosine Similarity (original analysis)
- Uses Sentence Transformers (`all-MiniLM-L6-v2` in the main script, `all-mpnet-base-v2` in the comparison tables)
- Cosine similarity of pooled sentence embeddings. This is **not** BERTScore

### 2. Sentiment Analysis
- Multi-label sentiment classification with 9 emotion categories:
  - **Positive**: Optimistic, Thankful, Humour
  - **Neutral**: Empathetic
  - **Negative**: Pessimistic, Anxious, Sad, Annoyed, Denial
- Fine-tuned on labeled tweet datasets
- Analyzes sentiment preservation across translations

### 3. Polarity Analysis
- Categorizes text into Positive, Neutral, and Negative polarity
- Compares sentiment polarity distribution across different translations

### 4. Comprehensive Scoring System
```
Overall Score = 0.5 × Semantic Similarity + 0.5 × Sentiment Similarity
```
- Semantic Similarity: Sentence-BERT (`all-MiniLM-L6-v2`) cosine similarity with expert translation, computed on mean-pooled embeddings of the whole corpus
- Sentiment Similarity: Cosine similarity of positive/negative ratio vectors

### 5. Advanced Visualizations
- Sentiment distribution charts (stacked bar charts, radar charts)
- Semantic similarity comparison (bar charts, polar charts)
- Polarity distribution analysis
- Comprehensive quality metrics heatmap
- Word frequency analysis and triple extraction
- Modern, publication-ready visualizations with high DPI (300)

## Project Structure

```
LLM-Translation-Madarin/
├── data set/                           # Translation corpora
│   ├── Human/                          # Expert human translations
│   │   ├── News_English_Version/
│   │   ├── Red_mansion_expert_merge/
│   │   └── Red_sorghum_expert_merge/
│   ├── Madarin original/               # Original Chinese texts
│   ├── gpt4/                           # GPT-4 translations
│   ├── gpt4o/                          # GPT-4o translations
│   ├── deepseek/                       # DeepSeek translations
│   └── google/                         # Google Translate translations
│
├── data_pre_processing/                # Data preprocessing scripts
│   ├── data_pre_processing.ipynb       # Convert DOCX to CSV
│   ├── gpt_translate.ipynb             # GPT translation script
│   └── deepseek_translate.ipynb        # DeepSeek translation script
│
├── semantic similarity compare table/  # Sentence-level similarity analysis
│   ├── translation_similarity_analysis.py
│   └── *.xlsx                          # Similarity comparison results
│
├── evaluation/                         # BERTScore + COMET + MPNet evaluation
│   ├── compute_mt_metrics.py           # per-segment metrics and system summaries
│   ├── paired_comparisons.py           # paired significance tests -> CSV + LaTeX tables
│   ├── test_evaluation.py              # synthetic tests (+ --smoke end-to-end run)
│   ├── requirements.txt
│   ├── results/                        # segment/system scores, correlations, alignments, paired_comparisons/
│   └── METRICS.md                      # Metric documentation / audit
│
├── plot/                               # Generated visualizations
│
├── sentiment and semantic analysis.py  # Main analysis script
│
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Required Packages

```bash
pip install pandas numpy matplotlib seaborn
pip install transformers sentence-transformers torch
pip install scikit-learn networkx spacy nltk
pip install python-docx openpyxl
```

### Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

## Usage

### 1. Data Preprocessing

Convert DOCX files to CSV format:
```python
# See data_pre_processing/data_pre_processing.ipynb
# Converts Word documents into structured CSV files
```

### 2. Run Comprehensive Analysis

Execute the main analysis script:

```python
python "sentiment and semantic analysis.py"
```

This will:
- Load all translation datasets
- Train/load sentiment analysis model
- Perform semantic similarity analysis
- Generate comprehensive visualizations
- Output detailed statistical reports

### 3. BERTScore, COMET and MPNet Evaluation

```bash
pip install -r evaluation/requirements.txt
python evaluation/compute_mt_metrics.py            # add --kiwi for reference-free CometKiwi
                                                   # --only-mpnet adds MPNet to existing results
python evaluation/paired_comparisons.py            # paired tests between systems
python evaluation/test_evaluation.py [--smoke]     # tests; --smoke runs the pipeline on one document per corpus
```

Writes `evaluation/results/{segment_scores,system_scores,metric_correlations}.csv`, per-document alignments, and
`evaluation/results/paired_comparisons/paired_comparisons_{all,news,red_mansions,red_sorghum}.{csv,tex}`.

### 4. Sentence-Level Similarity Analysis

For detailed sentence-level comparison:

```python
cd "semantic similarity compare table"
python translation_similarity_analysis.py
```

## Analysis Workflow

```
1. Data Loading
   └─> Load translations from CSV files

2. Model Initialization
   ├─> Sentiment Analysis Model (multi-label classifier)
   └─> Semantic Model (Sentence Transformers)

3. Analysis
   ├─> Sentiment Distribution Analysis
   ├─> Semantic Similarity Computation
   ├─> Polarity Classification
   └─> Comprehensive Scoring

4. Visualization
   ├─> Sentiment charts (distribution, radar, polarity)
   ├─> Semantic similarity charts
   ├─> Heatmaps and ranking tables
   └─> Word frequency and triple extraction

5. Output
   ├─> PNG visualizations (300 DPI)
   ├─> Console statistical reports
   └─> Qualitative analysis reports
```

## Output Files

The script generates the following visualizations for each text type (news, red_mansions, red_sorghum):

- `{type}_modern_comprehensive_analysis.png` - 5-in-1 comprehensive dashboard
- `{type}_modern_sentiment_distribution.png` - Sentiment distribution charts
- `{type}_polarity_distribution.png` - Polarity analysis
- `{type}_modern_semantic_similarity.png` - Semantic similarity comparison
- `{type}_modern_scoring_table.png` - Comprehensive ranking table
- `{type}_modern_triple_frequency.png` - Triple extraction analysis
- `{type}_{translation}_modern_word_freq.png` - Word frequency analysis

## Key Metrics

### Sentiment Categories
- **Optimistic**: Positive, hopeful expressions
- **Thankful**: Gratitude and appreciation
- **Empathetic**: Understanding and compassion
- **Pessimistic**: Negative, doubtful outlook
- **Anxious**: Worry and nervousness
- **Sad**: Sorrow and melancholy
- **Annoyed**: Irritation and frustration
- **Denial**: Rejection and disbelief
- **Humour**: Jokes and playful content

## Configuration

### Modify Analysis Parameters

Edit `sentiment and semantic analysis.py` to customize:

```python
# Training data folders
TRAINING_FOLDERS = [r"path/to/training/data"]

# Translation folder mappings
NEWS_TRANSLATION_FOLDERS = {
    'Expert Translation': r"path/to/expert",
    'GPT-4o': r"path/to/gpt4o",
    # Add more translations...
}
```

### Adjust Visualization Settings

```python
# Color schemes
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    # Customize colors...
}

# Figure DPI
plt.rcParams['savefig.dpi'] = 300  # High-resolution output
```

## Performance Considerations

- **Memory Usage**: Large datasets may require 8GB+ RAM
- **Processing Time**: Full analysis may take 30-60 minutes depending on dataset size
- **Model Caching**: Models are cached in `~/.cache/huggingface` for faster subsequent runs
- **GPU Acceleration**: Recommended for faster embedding generation

## Research Applications

This project can be used for:

1. **Translation Quality Assessment**: Benchmark LLM translation capabilities
2. **Comparative Studies**: Analyze differences between translation systems
3. **Sentiment Preservation**: Study emotional tone preservation in translations
4. **Literary Translation Analysis**: Evaluate translation quality for literary works
5. **Machine Translation Research**: Develop better evaluation metrics

## Limitations

- Requires expert translations as baseline for comparison
- Sentiment analysis model trained on English tweets (may not perfectly capture literary sentiment)
- Computational resources needed for large-scale analysis
- Current version focuses on Mandarin-to-English translation only

## Future Enhancements

- [ ] Support for more language pairs
- [ ] Real-time translation quality evaluation
- [ ] Integration with more LLM APIs
- [ ] Interactive web dashboard
- [ ] Fine-tuned sentiment models for literary texts
- [ ] Cross-cultural sentiment analysis

## Citation

If you use this project in your research, please cite:

```
@misc{llm-translation-evaluation,
  title={LLM Translation Quality Evaluation: Mandarin to English},
  author={Yue Zhang},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/LLM-Translation-Madarin}
}
```


## Acknowledgments

- Expert translators: H. Bencraft Joly, Howard Goldblatt
- Hugging Face Transformers library
- Sentence Transformers framework
- SenWave sentiment analysis dataset

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.


**Note**: This project is for research and educational purposes. Translation quality assessment is subjective and should be interpreted in context with domain expertise.
