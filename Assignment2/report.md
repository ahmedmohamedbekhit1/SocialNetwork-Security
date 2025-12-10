# Social Network Analysis with Bot Detection and Adversarial Attacks
## Research Report

### 1. Executive Summary
This report presents an empirical analysis of automated account detection in social networks and the efficacy of adversarial attacks against machine learning-based detection systems. Using the Facebook Egonets dataset from Stanford's Network Analysis Project, we developed a baseline classifier and systematically evaluated its robustness against two adversarial paradigms: Structural Evasion and Graph Poisoning attacks.

---

### 2. Introduction

#### 2.1 Background
Online social networks face persistent challenges from automated accounts that disseminate disinformation, manipulate public discourse, and engage in coordinated inauthentic behavior. Robust detection mechanisms are essential for maintaining platform integrity and user trust.

#### 2.2 Objectives
- Build and analyze a social network graph from real-world data
- Develop a bot detection model using graph-based features
- Evaluate the impact of adversarial attacks on detection performance
- Compare different attack strategies and their effectiveness

#### 2.3 Dataset
- **Source**: Facebook Egonets from Stanford SNAP
- **Description**: Social circles (friends lists) from Facebook users
- **Structure**: Undirected graph representing friendship connections
- **URL**: https://snap.stanford.edu/data/egonets-Facebook.html

---

### 3. Methodology

#### 3.1 Graph Construction
The social network was constructed as an undirected graph G = (V, E) where:
- V represents users (nodes)
- E represents friendship connections (edges)

#### 3.2 Feature Engineering
We extracted the following graph-based features for each node:

**Structural Features:**
- **Degree**: Number of direct connections
- **Average Neighbor Degree**: Mean degree of adjacent nodes
- **Clustering Coefficient**: Density of triangles around a node

**Centrality Measures:**
- **Betweenness Centrality**: Frequency of appearing on shortest paths
- **Closeness Centrality**: Average distance to all other nodes
- **Eigenvector Centrality**: Influence based on connections to important nodes
- **PageRank**: Importance score based on network structure

**Community Features:**
- **Community ID**: Membership in detected communities (modularity-based)

#### 3.3 Label Generation Methodology
Ground truth labels were derived using behavioral heuristics that correlate with automated account activity:
- Reduced clustering coefficients (limited community embeddedness)
- Elevated degree counts (indiscriminate following patterns)
- Diminished betweenness centrality (peripheral network positioning)

Bot score formula:
```
bot_score = -clustering + 0.5 × (normalized_degree) - 0.3 × betweenness
```

#### 3.4 Baseline Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, max depth 10
- **Features**: All graph metrics except community ID
- **Train/Test Split**: 70/30 with stratification
- **Preprocessing**: Standard scaling

#### 3.5 Attack Scenarios

**Scenario 1: Structural Evasion Attack**
Local topology manipulation to evade detection mechanisms:
1. Triangle completion among neighboring nodes (clustering enhancement)
2. Strategic edge pruning targeting high-degree connections
3. Preferential attachment to high-centrality nodes (legitimacy signaling)

**Scenario 2: Graph Poisoning Attack**
Training data corruption via adversarial injection:
1. Sybil node insertion with deceptive connectivity patterns
2. Strategic linking between sybil nodes and legitimate accounts
3. Adversarial edges between automated accounts and high-reputation users
4. Label manipulation (sybil nodes mislabeled as legitimate)

---

### 4. Results

#### 4.1 Graph Statistics

**Original Graph:**
- Nodes: 4,039
- Edges: 88,234
- Average Degree: 43.69
- Average Clustering Coefficient: 0.606
- Number of Communities: 77

**Bot Distribution:**
- Total Bots: 605
- Bot Ratio: 15.0%

#### 4.2 Baseline Performance

| Metric | Legitimate | Automated |
|--------|-------|-----|
| Precision | 1.00 | 0.99 |
| Recall | 1.00 | 0.98 |
| F1-Score | 1.00 | 0.99 |

**Overall Accuracy**: 99.59%

**Feature Importance:**
Top features contributing to classification:
1. Clustering coefficient (local network density)
2. Betweenness centrality (bridging position)
3. PageRank (network influence)
4. Average neighbor degree (connection quality)
5. Eigenvector centrality (connection to important nodes)

#### 4.3 Post-Attack Performance

**After Structural Evasion:**

| Metric | Legitimate | Automated |
|--------|-------|-----|
| Precision | 1.00 | 0.97 |
| Recall | 1.00 | 0.99 |
| F1-Score | 1.00 | 0.98 |

**Overall Accuracy**: 99.42%
**Accuracy Degradation**: 0.17%

**After Graph Poisoning:**

| Metric | Legitimate | Automated |
|--------|-------|-----|
| Precision | 1.00 | 0.94 |
| Recall | 0.99 | 0.98 |
| F1-Score | 0.99 | 0.96 |

**Overall Accuracy**: 98.85%
**Accuracy Degradation**: 0.74%

#### 4.4 Comparative Analysis

| Condition | Accuracy | F1-Score | Precision (Bot) | Recall (Bot) |
|-----------|----------|----------|-----------|--------|
| Baseline | 99.59% | 0.9862 | 0.99 | 0.98 |
| Structural Evasion | 99.42% | 0.9809 | 0.97 | 0.99 |
| Graph Poisoning | 98.85% | 0.9624 | 0.94 | 0.98 |

---

### 5. Discussion

#### 5.1 Impact of Structural Evasion
- **Mechanism**: Local graph topology manipulation around flagged nodes
- **Detection Impact**: Minimal degradation (0.17% accuracy drop), but improved recall for automated accounts
- **Feature Perturbation**: Clustering coefficient increased by 15-25%, betweenness centrality reduced
- **Evasion Efficacy**: 20 attacked nodes showed improved camouflage, precision dropped to 0.97

**Observations:**
- Clustering enhancement reduces structural anomaly scores
- High-centrality attachments provide legitimacy signals
- Feature distributions converge toward legitimate account patterns
- Attack successfully made automated accounts appear more embedded in social communities
- Triangle creation increased local network density around targeted nodes

#### 5.2 Impact of Graph Poisoning
- **Mechanism**: Training data corruption via strategic node/edge injection
- **Detection Impact**: More significant degradation (0.74% accuracy drop), precision reduced to 0.94
- **Classifier Degradation**: Model learned from 15 mislabeled sybil nodes, creating false negatives
- **Attack Efficiency**: 30 total modifications (15 nodes + 15 edges) affected global performance

**Observations:**
- Sybil nodes create ambiguous decision boundaries
- Adversarial edges reduce class separability
- Classifier learns corrupted patterns affecting all predictions
- More effective than structural evasion with 4.4x greater accuracy degradation
- Affects model's ability to distinguish between legitimate and automated accounts broadly

#### 5.3 Comparison of Attack Strategies

**Structural Evasion Advantages:**
- Targets specific bot nodes with surgical precision
- Harder to detect as malicious activity (appears as organic network evolution)
- Preserves overall graph structure and statistics
- Minimal impact allows continued operation

**Structural Evasion Limitations:**
- Requires modifications to many nodes for significant impact
- Limited effectiveness (only 0.17% degradation)
- May still be detectable by temporal analysis

**Graph Poisoning Advantages:**
- More efficient (30 modifications vs. 20 targeted nodes)
- Corrupts model training directly with 4.4x greater impact
- Harder to defend against without data validation
- Affects global model performance

**Graph Poisoning Limitations:**
- May be detected by graph anomaly detection
- Requires ability to inject new nodes (platform access)
- Sybil nodes may be identified through behavioral analysis

#### 5.4 Defense Mechanisms
Potential countermeasures:
1. **Robust Feature Engineering**: Perturbation-resistant metrics
2. **Anomaly Detection**: Structural change point detection
3. **Temporal Analysis**: Longitudinal behavioral patterns
4. **Ensemble Methods**: Multiple detection paradigms
5. **Data Sanitization**: Pre-training anomaly filtering

---

### 6. Visualizations

#### 6.1 Graph Visualizations
[Include the three generated graph images]
- Figure 1: Baseline Network
- Figure 2: After Structural Evasion
- Figure 3: After Graph Poisoning

#### 6.2 Performance Comparison
[Include performance_comparison.png]
- Figure 4: Accuracy and F1-Score comparison across conditions

#### 6.3 Feature Distribution Changes
[Optional: Add plots showing how feature distributions change after attacks]

---

### 7. Conclusions

#### 7.1 Key Findings
1. Baseline bot detection achieved 99.59% accuracy using graph features
2. Structural evasion reduced accuracy by 0.17% through local modifications
3. Graph poisoning reduced accuracy by 0.74% through training data corruption
4. Graph poisoning was 4.4x more effective at degrading model performance
5. Both attacks successfully reduced precision while maintaining high recall
6. The Facebook social network exhibited high clustering (0.606) and community structure (77 communities)

#### 7.2 Implications
- Graph-based classifiers exhibit vulnerability to adversarial manipulation
- Attack paradigms exploit distinct classifier weaknesses
- Robust detection requires multi-faceted defense strategies
- Structural features alone prove insufficient in adversarial environments

#### 7.3 Limitations
- Heuristic-based labeling may not capture authentic automated account behavior
- Analysis restricted to structural features (content-agnostic)
- Single classifier architecture evaluated
- Simplified attack implementations

#### 7.4 Future Research Directions
- Validation using authenticated bot datasets
- Implementation of adversarially robust classifiers
- Graph Neural Network architectures
- Multi-modal feature integration (structural + behavioral + content)
- Adaptive detection systems with continuous learning

---

### 8. References

1. Stanford SNAP: Facebook Social Circles Dataset
   - URL: https://snap.stanford.edu/data/egonets-Facebook.html

2. NetworkX Documentation
   - URL: https://networkx.org/documentation/stable/

3. Graph-based Bot Detection Literature
   - [Add relevant academic papers]

4. Adversarial Machine Learning
   - [Add relevant academic papers]

5. Social Network Analysis
   - [Add relevant books/papers]

---

### Appendix A: Code Implementation
[Link to GitHub repository or include key code snippets]

### Appendix B: Additional Visualizations
[Include any supplementary figures]

### Appendix C: Detailed Results Tables
[Include full classification reports and confusion matrices]

---

**Report Prepared By**: Ahmed
**Date**: December 10, 2025
**Project**: Social Network Analysis with Adversarial Bot Detection
**Dataset**: Facebook Egonets - Stanford SNAP
