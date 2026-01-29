# Specializing LLMs to Low-Documented Domains with RAG
This repo shares:
- the prompts used that implement the RAG pipeline
- the benchmark used to evaluate the pipeline

## 📂 Repository Contents

- `benchmark.json` – The benchmark dataset in JSON format.
- `benchmark_reader.py` – Python code for reading and validating the benchmark.
 used in the study.
- `APPENDIX D -- prompts.py` – Reference implementation of the prompt templates

---
---

## Unity XRI v2 Q&A Benchmark

This benchmark structure is designed to be extensible — you can add Q&A datasets for any XR platform and toolkit. However, this repository currently includes only one dataset: **Unity** as the platform and **XRI version 2** as the toolkit. 

It includes a Python utility script for easily loading, validating, and querying the dataset.

---


### 📘 Benchmark Structure

The benchmark is organized as a hierarchy:

- **benchmark_info** – General metadata.
- **platforms[]** – E.g., Unity, Web(Mock).
  - **toolkits[]** – E.g., XRIv2, MRTK3(Mock), A-Frame(Mock).
    - **dataset** – List of Q&A pairs, with optional metadata.

---

#### Example

```json
{
  "benchmark_info": {
    "name": "XRI-benchmark",
    "description": "Text-based, Q&A Benchmark for Virtual Reality applications...",
    "version": "0.1",
    "date": "2024-09-15",
    "author": "CG3HCI (https://cg3hci.dmi.unica.it/lab/)",
    "email": "jacopo.mereu@unica.it"
  },
  "platforms": [
    {
      "name": "Unity",
      "toolkits": [
        {
          "name": "XRIv2",
          "dataset": [
            {
              "question": "What is ... ?",
              "answer": "... is a ...",
              "metadata1": "A value",
              ...
              "metadataN": "Another value"
            }
          ]
        }
      ]
    }
  ]
}

