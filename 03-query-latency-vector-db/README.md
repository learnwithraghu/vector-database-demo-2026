# ZVec Query Latency Performance Demonstration

A comprehensive demonstration of ZVec vector database query latency performance using NFL 2025 observation data.

**🐳 Docker-Ready**: No Python environment setup needed! Just run with Docker.

## Overview

This project showcases the speed and efficiency of ZVec vector database through detailed latency benchmarking and performance analysis. Using real-world NFL 2025 preview data, the demonstration measures query response times, throughput metrics, and latency distributions across various query types and batch sizes.

## Features

- **PDF Document Ingestion**: Automated extraction and processing of NFL 2025 observation PDFs
- **Vector Embeddings**: Semantic embeddings using sentence-transformers
- **ZVec Integration**: High-performance vector storage and retrieval
- **Latency Benchmarking**: Comprehensive query performance measurements
- **Performance Visualizations**: Charts and graphs showing:
  - Query response times
  - Throughput metrics
  - Latency distributions
  - Percentile analysis (P50, P95, P99)
  - Simple vs complex query comparisons
- **Batch Query Analysis**: Throughput benchmarks for various batch sizes
- **Real-time Metrics**: Live performance monitoring during execution

## Dataset

- **Source**: NFL 2025 Preview observations
- **Format**: PDF documents in [`pdf_doc/`](pdf_doc/) subfolder
- **Content**: Comprehensive NFL team analyses, statistics, predictions, and schedules
- **Processing**: Chunked into semantic segments for optimal vector search

## Performance Metrics

The demonstration measures and visualizes:

1. **Single Query Latency**
   - Mean, median, min, max response times
   - Standard deviation and variance
   - Percentile distributions (P50, P75, P90, P95, P99, P99.9)

2. **Batch Query Throughput**
   - Queries per second for different batch sizes
   - Optimal batch size identification
   - Scaling characteristics

3. **Query Complexity Analysis**
   - Simple queries (single keywords)
   - Complex semantic queries (full sentences)
   - Performance comparison

4. **Ingestion Performance**
   - PDF loading time
   - Text chunking time
   - Embedding generation time
   - Vector insertion rate

## Project Structure

```
query-latency-vector-db/
├── README.md                      # This file
├── setup.sh                       # Environment setup script
├── query_latency_demo.ipynb       # Main Jupyter notebook
├── pdf_doc/                       # PDF documents directory
│   └── NFL_2025.pdf              # NFL 2025 observations
└── venv/                         # Virtual environment (created by setup.sh)
```

## Requirements

### Docker (Recommended - Easiest!)
- Docker Desktop or Docker Engine
- 4GB+ RAM recommended
- ~3GB disk space for Docker image and data

**No Python installation needed!** Docker handles everything.

## Installation & Usage

### 🐳 Docker Setup (Recommended)

**Easiest way - no Python setup required!**

1. **Install Docker** (if not already installed):
   - **macOS/Windows**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - **Linux**: [Docker Engine](https://docs.docker.com/engine/install/)

2. **Run the demo**:
   ```bash
   cd 03-query-latency-vector-db
   ./run-docker.sh
   ```

3. **Access Jupyter Notebook**:
   - Open your browser to: **http://localhost:8888**
   - No token required - direct access enabled
   - The notebook will be ready to use!

4. **Stop the container**:
   - Press `Ctrl+C` in the terminal
   - Or run: `docker-compose down`

**What the Docker setup does:**
- ✓ Uses Python 3.10 (no version conflicts!)
- ✓ Installs all dependencies automatically
- ✓ Launches Jupyter Notebook on port 8888
- ✓ Mounts your notebook for live editing
- ✓ No local Python environment needed!

## Running the Demo

```bash
cd 03-query-latency-vector-db
./run-docker.sh
```

Then open **http://localhost:8888** in your browser.

### Execute the Notebook

Run all cells in sequence to:
   - Load and process NFL 2025 PDF documents
   - Generate vector embeddings
   - Initialize ZVec database
   - Insert vectors with timing
   - Run query latency benchmarks
   - Generate performance visualizations
   - View comprehensive performance summary

### Notebook Sections

The [`query_latency_demo.ipynb`](query_latency_demo.ipynb) notebook includes:

1. **Setup and Imports** - Load required libraries
2. **PDF Processing** - Extract and chunk text from NFL documents
3. **Embedding Generation** - Create vector embeddings
4. **ZVec Initialization** - Set up vector database
5. **Data Insertion** - Load vectors with performance tracking
6. **Single Query Benchmarks** - Measure individual query latency
7. **Batch Query Benchmarks** - Test throughput at scale
8. **Query Complexity Analysis** - Compare simple vs complex queries
9. **Visualizations** - Generate performance charts
10. **Percentile Analysis** - Detailed latency distribution
11. **Performance Summary** - Comprehensive metrics report
12. **Sample Queries** - Demonstrate actual search results
13. **Conclusion** - Key findings and use cases

## Performance Expectations

Based on typical hardware (modern laptop/desktop):

- **Query Latency**: 1-10 ms (mean)
- **P95 Latency**: < 20 ms
- **P99 Latency**: < 50 ms
- **Throughput**: 100-1000+ queries/second (batch)
- **Insertion Rate**: 1000-10000+ vectors/second

*Actual performance varies based on hardware, dataset size, and query complexity.*

## Key Technologies

- **[ZVec](https://github.com/zvec/zvec)**: High-performance vector database
- **[Sentence Transformers](https://www.sbert.net/)**: State-of-the-art text embeddings
- **[PyPDF2](https://pypdf2.readthedocs.io/)**: PDF text extraction
- **[Jupyter](https://jupyter.org/)**: Interactive notebook environment
- **[Matplotlib](https://matplotlib.org/)** & **[Seaborn](https://seaborn.pydata.org/)**: Data visualization

## Visualizations

The notebook generates several visualizations:

1. **`query_latency_analysis.png`** - 4-panel visualization:
   - Mean query latency with error bars
   - Latency distribution histogram
   - Batch throughput scaling
   - Simple vs complex query comparison

2. **`latency_percentiles.png`** - Percentile distribution chart

3. **`performance_summary.csv`** - Exportable metrics table

## Use Cases

This demonstration is relevant for:

- **Real-time Search Applications**: Low-latency semantic search
- **Recommendation Systems**: Fast similarity matching
- **Question Answering**: Interactive Q&A systems
- **Content Discovery**: Efficient document retrieval
- **Performance Benchmarking**: Vector database evaluation

## Customization

### Using Your Own Data

Replace the PDF in [`pdf_doc/`](pdf_doc/) with your own documents:

```python
# In the notebook, modify:
pdf_path = Path('pdf_doc/YOUR_DOCUMENT.pdf')
```

### Adjusting Parameters

Customize performance testing:

```python
# Chunk size for text processing
chunk_size = 500  # Adjust based on your content

# Number of benchmark runs
num_runs = 50  # More runs = more accurate statistics

# Batch sizes to test
batch_sizes = [10, 50, 100, 200, 500]  # Add/remove sizes

# Number of results per query
k = 5  # Top-k results
```

### Different Embedding Models

Try alternative models:

```python
# Options: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', etc.
model = SentenceTransformer('all-mpnet-base-v2')
```

## Troubleshooting

### Common Issues

#### Docker Issues

**Issue**: `Cannot connect to the Docker daemon`
- **Solution**: Ensure Docker Desktop is running
- **macOS/Windows**: Start Docker Desktop application
- **Linux**: `sudo systemctl start docker`

**Issue**: Port 8888 already in use
- **Solution**: Stop other Jupyter instances or change port in `docker-compose.yml`:
  ```yaml
  ports:
    - "8889:8888"  # Use port 8889 instead
  ```

**Issue**: Docker build fails
- **Solution**: Ensure you have internet connection and sufficient disk space (~3GB)

#### Python/Dependency Issues

**Issue**: `ModuleNotFoundError: No module named 'zvec'`
- **Solution**: Rebuild Docker image: `docker-compose build --no-cache`

**Issue**: PDF extraction fails
- **Solution**: Verify PDF is not encrypted or corrupted

**Issue**: Out of memory errors
- **Solution**:
  - **Docker**: Increase Docker memory limit in Docker Desktop settings
  - **Local**: Reduce batch size or chunk size in the notebook

**Issue**: Slow embedding generation
- **Solution**: Reduce batch size or use a smaller embedding model

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version` (should be 3.8+)
3. Ensure PDF file exists in [`pdf_doc/`](pdf_doc/) directory
4. Review error messages in notebook cells

## Performance Optimization Tips

1. **Batch Processing**: Use larger batches for better throughput
2. **Embedding Model**: Smaller models = faster but less accurate
3. **Chunk Size**: Balance between granularity and performance
4. **Hardware**: GPU acceleration available for embedding generation
5. **Index Type**: ZVec supports different index types for speed/accuracy tradeoffs

## License

This demonstration project is provided as-is for educational and benchmarking purposes.

## Acknowledgments

- NFL 2025 observation data used for demonstration purposes
- ZVec team for the high-performance vector database
- Sentence Transformers for state-of-the-art embeddings
- Open source community for supporting libraries

## Next Steps

After running this demonstration:

1. **Experiment** with different query types
2. **Compare** ZVec performance with other vector databases
3. **Scale** to larger datasets
4. **Integrate** into your own applications
5. **Optimize** parameters for your specific use case

---

**Ready to see ZVec in action?**

```bash
./run-docker.sh
```

Then open **http://localhost:8888**

No Python installation needed - Docker handles everything! 🐳
