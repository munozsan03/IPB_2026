# IPB_2026

**Enzyme–Substrate Prediction with Protein Embeddings and Chemical Representations**

This repository contains the work I completed during my summer 2026 research internship at the [Leibniz Institute of Plant Biochemistry (IPB)](https://www.ipb-halle.de/) in Halle, Germany, as part of the Davari Mehdi research group.

The primary goal of this project was to replicate and extend **CATNIP**, a machine learning workflow developed by the Gomes Group for predicting enzyme–substrate compatibility. The original CATNIP project combines information from protein sequence space and chemical space to identify enzyme candidates that are most likely to interact with a given small molecule.

Original CATNIP repository: [github.com/gomesgroup/catnip](https://github.com/gomesgroup/catnip)

## Project Background

Identifying an enzyme capable of catalyzing a reaction with a particular substrate is an important challenge in biocatalysis. Traditionally, finding suitable enzymes can require screening large numbers of candidates experimentally, which can be expensive and time-consuming.

CATNIP addresses this problem by using computational methods to rank enzyme candidates for a given substrate, rather than experimentally testing every available enzyme. The original workflow combines protein sequence similarity and small-molecule chemical representations into a gradient-boosted decision tree model that produces a ranked list of candidate enzymes.

One important limitation, however, is that sequence similarity does not necessarily imply functional similarity — two proteins can have similar sequences while differing in their ability to interact with a particular substrate. Likewise, two chemically similar molecules may not necessarily be recognized by the same enzyme.

## Project Objective

The objective of this project was to reproduce the original CATNIP workflow and investigate whether incorporating protein language-model embeddings and additional representations of chemical and protein similarity could improve enzyme–substrate prediction.

The project was developed incrementally, with each model building on the previous version.

## Model Development

The project progressed through four main stages:

### 1. CATNIP replication
Reproduced the original CATNIP workflow using the publicly available code and dataset. This established a baseline against which all subsequent modifications could be evaluated.

### 2. CATNIP + protein embeddings
Introduced protein embeddings generated using ESMC-600M in addition to the original protein representation. Protein language-model embeddings provide a numerical representation of protein sequences that can capture information beyond direct sequence similarity, potentially allowing the model to learn additional relationships between enzymes.

### 3. CATNIP + protein embeddings + sequence/alignment information
Combined the ESMC protein embeddings with the original sequence-based similarity information, letting the model use both learned protein representations and explicit relationships between enzyme sequences.

### 4. CATNIP + enhanced protein and chemical representations
Adapted the existing **EviCYP** pipeline to the CATNIP/BioCatSet1 dataset. The original EviCYP framework was designed for enzyme–substrate prediction using protein and molecular representations; I replaced its original data with my own enzyme and substrate data while keeping the existing pipeline intact.

EviCYP repository: [github.com/yjyang-lin/EviCYP](https://github.com/yjyang-lin/EviCYP)

Each stage was evaluated using the same enzyme–substrate prediction task and ranking metrics, allowing direct comparison of the different representations.

## Results and Comparison

Each model configuration is evaluated using ranking-based metrics:

- Precision@10
- Recall@10
- NDCG@10
- Enrichment@10

These metrics measure different aspects of the model's ability to retrieve the correct enzyme candidates within the top-ranked predictions.

The results section of this repository compares the four model configurations and examines how each additional representation affects prediction performance, with the goal of determining whether protein embeddings and richer chemical representations can improve upon the original CATNIP approach.

## Reproducing the Results

### 1. Environment Setup

The project was developed using Python 3.10 and a Conda environment.

```bash
conda create -n sm_ipb_2026 python=3.10
conda activate sm_ipb_2026
pip install -r requirements.txt