# Text Mining of COVID-19 Research Papers using Word and Sentence Embeddings
<p align="center">
	<img src="https://www.lsvbw.de/wp-content/uploads/2020/02/2802_Corona.jpg">
</p>

# Purpose
The CORD-19 dataset is a vast collection of literature on the novel coronavirus. We can apply text and data mining approaches to find answers to questions in the literature in support of the ongoing COVID-19 response efforts worldwide.

##### What do we know about COVID-19 risk factors?
  - Smoking, pre-existing pulmonary disease
  - Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
  - Neonates and pregnant women
  - Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
  - Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
  - Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
  - Susceptibility of populations
  - Public health mitigation measures that could be effective for control


# Method
First, the documents on COVID-19 are retrieved using a BM-25 search engine. Then, to find answers to the questions above, two methods are used to find sentences in the papers that talk about those topics.

**Method 1:**
1. Create TF-IDF vectors for all sentences from all papers
2. For a particular Search Query, get the TF-IDF vector.
3. Find the highest Cosine Similarity between the Search Query and all the sentences from the papers.

 - Pros: Fast and accurate.
 - Cons: Not able to capture semantic relationships between words.

**Method 2:**
1. Train Word Embeddings (Word2Vec) on the papers' texts.
2. For a particular Search Query, get the embedded Word Vectors.
3. Find the lowest Word Mover's Distance between the Search Query and all the sentences from the papers.

 - Pros: Able to capture semantic relationships between words.
 - Cons: Distance calculations are slow.

# Results

*Question: Risk factors of smoking, pre-existing pulmonary disease*
![Smoking1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/smoking1.png)
![Pulmonary1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/pulmonary1.png)
---
*Question: Co-infections and other co-morbidities*
![Comorbidities1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/comorbidities1.png)
---
*Question: Neonates and pregnant women*
![Neonates1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/neonates1.png)
---
*Question: Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.*
![Comorbidities1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/comorbidities1.png)
---
*Question: Susceptibility of populations*
![Susceptibility1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/susceptibility1.png)
---
*Question: Public health mitigation measures that could be effective for control*
![Mitigation1](https://raw.githubusercontent.com/tchanda90/cord-19/master/img/mitigation1.png)

 