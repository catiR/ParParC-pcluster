
Requires to install [SPTK](https://github.com/sp-nitech/SPTK)

Get DBA_multivariate.py [here](https://github.com/fpetitjean/DBA/blob/master/DBA_multivariate.py).


#### Validate cluster-similarity-based classifications of sentence prosody

Edit `ppc_oracle.py` to adjust the sentence inventory, acoustic feature set, and options including: number of clusters per focus position, discard very small clusters after initial clustering, handle missing values and normalise acoustic features, distance metric for computing DTW local cost matrix, whether to apply [DBA](https://github.com/fpetitjean/DBA) reducing clusters to centroids before performing classification.


#### Cluster sentences based on prosodic similarity

Edit `textfilter` in `ppc_pcluster.py` to change the sample of sentences to cluster.

## Output

Audio file IDs like `0104_1_j` include sentence text ID `0104`, focus position `1` (indexing startd at `0`, `N` is no focus specified), and speaker ID `j`

Ideal prosodic clusters should consist mainly of a single focus position, with optionally some N.

Clustering across focus conditions by speaker ID (or gender), or by sentence text, could mean sensitivity to (para)linguistic distractors. Uninterpretable clustering, especially with high loss, could mean failure of the acoustic feature set or an incoherent choice of input sample.

### Data

All audio and metadata files in this experiment use data from the [Paralinguistic Paraphrase Corpus](https://dsc-nlp.naist.jp/data/speech/paralinguistic_paraphrase/) which is distributed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA) license: 

Naoaki Suzuki, Satoshi Nakamura. (2022)
Representing 'how you say' with 'what you say': English corpus of focused speech and text reflecting corresponding implications.
Proc. Interspeech 2022, 4980-4984.

Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. 2010. Collecting Image Annotations Using Amazon’s Mechanical Turk. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon’s Mechanical Turk, pages 139–147, Los Angeles. Association for Computational Linguistics.
