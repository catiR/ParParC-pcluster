from ppc_pcluster import *
import random
from collections import defaultdict

# oracle preliminary evaluation
# 
# - create clean clusters of known labels
# - compare held-out human data (with known labels) to the clusters
# --> compare TTS to the clusters? TODO

# simplest evaluation: label of lowest-distance cluster is assigned to Query
# alternate: train something to predict Query's label from a vector of its distances to all clusters
#   -- if there's enough data to ever do that

# ( in the future: 
#  would be nice if something works with automatically induced clusters
#  so that it doesn't require fully human labelled corpora to evaluate TTS )





def test_heldout_oracle(sent_ids, nwords, n_clusters, feat_list, audioc_dir, mfaln_dir, json_path, tmp_dir, remove_small=True):
	
	with open(json_path,'r') as handle:
		db = json.load(handle)

	
	# this only works for handmade batches of sentences with same word counts & directly comparable words
	# and it's only like this to abuse functions from ppc-pcluster to save time today
	# and waste three times longer doing it right later
	def _construct_batches(nwords,slist):
		slist = set(slist)
		batches = {str(wix) : { sid : [wix] for sid in slist } for wix in range(nwords) }
		return batches
		
		
	batches = _construct_batches(nwords, sent_ids)
	
	
	batche2 = {label_name: extract_sample(db, label_spec) for label_name, label_spec in batches.items()} 
	
	batch_feats = {label_name: get_sample_feats(db_sample, mfaln_dir, tmp_dir) 
					for label_name,db_sample in batche2.items()}
	
	batch_feats = {label_name : prep_feats(sample_feats, feature_list=feat_list)
					for label_name , sample_feats in batch_feats.items()}
	
	
	# set up random 5fold cross validation sets
	def _construct_5fold(batch_feats):
		ids_listed = {label_name : list(label_data.keys()) for label_name, label_data in batch_feats.items()}
		ids_listed = {k : random.sample(v,k=len(v)) for k,v in ids_listed.items()}
		
		ids_listed = {k :  [v[i::5] for i in range(5)] for k,v in ids_listed.items() }
		return ids_listed
	
	kfolds = _construct_5fold(batch_feats)



	def test_one(testr, clusters, feats, remove_small = True):
		
		# TODO not this if batches are named something besides word position
		def _b(r):
			return r.split('_')[1]
	
		rec,ground = testr
		
		if remove_small:
			clusters = [(r,l) for r,l in clusters if len(r)>2]

		cdists = [ (np.nanmean( 
						[ dtw_distance(feats[_b(rec)][rec], feats[_b(refrec)][refrec]) for refrec in cluster_recs ] 
						) ,
					cluster_label)
					for cluster_recs, cluster_label in clusters]
		cdists = sorted(cdists, key=lambda x: x[0])
		
		
		result = [rec, ground, cdists, cdists[0][1]==ground, [x[1] for x in cdists].index(ground) + 1]
		return result
		

	def report_fold(results):
		print(len(results), "test recordings")
		print(len([r for r in results if r[3]])/len(results)*100, '% correctly classified at top-1' )
		print(np.nanmean([r[4] for r in results]), "- average rank of best correct cluster (perfect = 1)")
		# TODO break down by position, confusion matrix etc
		
	# run the whole train + test
	def run_fold(knum,fold_spec,all_feats):
	
		print(' - - - - Partition =',knum,'- - - -')
		trainparts = [0,1,2,3,4]
		trainparts.remove(knum)
		
		testset = {k : v[knum] for k,v in fold_spec.items()}
		
		testids = set([x for b in testset.values() for x in b])
		train_batches = {k: {kk:vv for kk,vv in v.items() if kk not in testids} for k,v in all_feats.items() }
		
		# make a few clusters within each label
		batch_clusters = {k: make_clusters(v, n_clusters = n_clusters) for k,v in train_batches.items()}
		
		def _reformat_clstr(clist):
			cl = defaultdict(list)
			for rec,clid in clist:
				cl[clid].append(rec)
			return cl.values()

		
		# testset is list of (recording_id, true_label) pairs
		testset = [i for s in [[(rec,k) for rec in v] for k,v in testset.items()] for i in s]
		batch_clusters = {k: _reformat_clstr(v) for k,v in batch_clusters.items()}
		batch_clusters = [[(clstr,labl) for clstr in clstrs] for labl, clstrs in batch_clusters.items()]
		batch_clusters = [i for s in batch_clusters for i in s]
		
		fold_eval = [test_one(rec,batch_clusters,all_feats) for rec in testset]
		return(fold_eval)
		

	fold_evals = [run_fold(knum,kfolds,batch_feats) for knum in range(5)]
	
	for f in fold_evals:
		report_fold(f)
		print('\n - - - - - - - - - - - - - - - -\n')
		
	print('\n\n\nThe FEATURE TYPES were', feat_list)
	if remove_small:
		print('* Excluded very small clusters from candidates')
		
		
		




if __name__ == "__main__":

	# original corpus audio
	# https://dsc-nlp.naist.jp/data/speech/paralinguistic_paraphrase/
	audioc_dir = '~/work/cc/parpc/data/original_dl_speech/'
	
	# mfa alignments
	# csv, directories in mfa corpus structure
	# ! ask caitlinr@ru.is for these
	mfaln_dir = './epc_aligns_adapted/'
	
	# output provided from setupmetadata.collectinfo()
	json_path = './paraphrase.json'
	
	#sptk_bin = '~/work/env/SPTK/bin/'
	
	# store extracted speech features here
	#!this script assumes acoustic features for the whole corpus already exist
	# extracted by featurise_sample in ppc_pcluster.py
	#-if they don't exist yet, do that first.
	feats_dir = './tmp2/'
	
	audioc_dir, mfaln_dir, json_path, feats_dir = [os.path.expanduser(x) 
		for x in [audioc_dir, mfaln_dir, json_path, feats_dir]]
		
	
	# ---------------- some potential subsets of data to work with ----------------
	# for now, all the data in a subset needs the same number of words & they should be reasonably parallel
	#   e.g. probably don't mix 6 word sentences with verbs from 2nd-4th position.
	# TODO read input groups from files do not hardcode kludges here 
	
	# 3 words DET N VB
	threeword = ['0122','0082','0117','0149','0070','0034']
	
	# 4 words DET N V (preposition, adverb, or object)
	dnvx = ['0144', '0006', '0060', '0033', '0112', '0185', '0011', '0152']
	
	# 6 words DET N V DET (modifier) N
	dnvdmn = ['0178', '0042', '0177', '0187', '0047', '0097', '0124', '0015', '0142', '0133', '0184', '0139', '0120', '0071'] 
	
	# 6 words DET N V PREP DET N (mostly)
	dnvpdn = ['0161', '0025', '0009', '0193', '0076', '0194', '0016', '0039', '0080', '0099', '0077', '0153', '0087', '0165', '0095', '0168', '0156', '0073', '0180', '0083', '0043', '0092', '0171', '0008', '0001', '0110', '0170', '0048', '0116', '0056', '0023', '0104', '0049', '0046', '0127', '0126', '0096', '0072', '0069', '0143', '0064', '0030', '0107', '0160', '0038', '0164', '0192', '0075', '0004', '0012', '0141', '0128', '0162', '0086', '0176', '0053', '0094', '0146', '0181', '0045', '0136', '0003', '0022', '0085', '0159', '0154', '0102', '0017', '0041', '0020', '0098', '0079', '0188', '0111'] 
	
	
	feature_types=['rmse','swipe.f0']
	# options: rmse, ene, swipe.f0, harvest.f0, reaper.f0, rapt.f0
	#   bad F0 tracking can cause DTW errors, if that happens use a different f0 for now
	

	# comment in one ---------------
	# for n_clusters try anything reasonable for the amount of sentences
	#sent_ids, nwords, n_clusters = dnvdmn, 6, 4
	#sent_ids, nwords, n_clusters  = threeword, 3, 3 # this with features rmse & rapt.f0 is not terrible for a start
	#sent_ids, nwords, n_clusters = dnvx, 4, 5
	sent_ids, nwords, n_clusters  = dnvpdn, 6, 10
	
	
	
	# this function does 5fold cross validation with the above selected data
	test_heldout_oracle(sent_ids, nwords, n_clusters, feature_types, 
			audioc_dir, mfaln_dir, json_path, feats_dir)
			
			
			
	
	
