import json, os, re, glob, subprocess
import librosa
import numpy as np
from copy import deepcopy
from dtw import dtw
from scipy import stats
import kmedoids as kmd

#ppc-pcluster
# - perform automatic prosodic clustering based on acoustic features
# - evaluate the quality of these clusters
# - TODO: save/output the clusters


# clean
# basic sentence text cleaner for paralinguistic paraphrase corpus
def cln(s):
	s = re.sub(r'[^a-z ]+', '', s.lower())
	return s


# emphasise text
# apply capslock to word at index i in sentence s
def emt(s,i):
	if i.lower() == 'n':
		return s
	i = int(i)
	s = s.split(' ')
	#s[i] = s[i].upper()
	s[i] = f'<{s[i].upper()}>'
	return ' '.join(s)


# to zscore a piece of data by factors calculated externally from larger data
# and/or custom handle invalid feature values
def z_score(x, mean, std):
	return (x - mean) / std
	
	
# extract sample
# of recordings from db according to textfilter
#  textfilter: dict of {sentence_id : [optional, emphasised, positions, ...]}
#   positions are word indices or 'N' unspecified/reader's default emphasis
#   empty list matches all positions
def extract_sample(db, textfilter):

	sample = {}
	
	for sent, poss in textfilter.items():
		sentsample=db
		if sent:
			sentsample = {k:v for k,v in db.items() if sent == v["sentence_id"]}
		if poss:
			poss = [str(p) for p in poss]
			sentsample = {k:v for k,v in sentsample.items() if v["focus_index"] in poss}

		for k,v in sentsample.items():
			recs = v.pop('recordings')
			v['condition_id'] = k
			for rec in recs:
				rec.update(v)
				sample[rec['file_id']] = rec

	return sample



# librosa rmse
def get_rmse(wpath, khz, fs, znorm = True):
    """
    Returns an array of RMSE values for a given speech.
    """
    
    audio, sr = librosa.load(wpath, sr=khz*1000)
    hop = khz * fs # hop size in number of samples
    fl = khz * 30 # frame length (window) in number of samples 
    # set to 30ms window following cheng 2013, maybe change??
    rmse = librosa.feature.rms(y=audio,frame_length=fl,hop_length=hop)
    rmse = rmse[0] 
    if znorm:
        rmse = stats.zscore(rmse)
    return rmse
    
    
    

# utility to run sptk pitch commands 
# on sptk-formatted data,
# given sptk algorithm number ID + executable path
# sr: sample rate in khz
# fs: frame shift in ms
# ts: voicing threshold parameters for each algorithm
#    defaults t0 0.0 t1 0.3 t2 0.9 t4 0.01
#    note - swipe t1 had lot of problems finding any voicing
def run_pitch(sptk_fmt,lp,hp,anum,sptk,sr=16, fs=10, ts = " -t0 0.4 -t1 0.2 -t2 1.1 -t4 0.15"):
	f0_raw_data = subprocess.run(
		f"{sptk}x2x +sd {sptk_fmt} | \
		{sptk}pitch -a {anum} -s 16 -p {sr*fs} {lp}{hp}{ts}\
		| {sptk}x2x +da", shell=True, capture_output=True).stdout
	f0_raw_data = f0_raw_data.decode()
	return f0_raw_data
				
				
# compute acoustic features from sample's audio files
#	(corpus is provided as 16khz mono wav)
#  and save everything in feature_dir
def featurise_sample(sample, audio_dir, sptk, feature_dir):
	
	if not os.path.exists(feature_dir):
		os.makedirs(feature_dir)
		
	sr = 16  # Sample rate in kHz
	fs = 10  # Frame shift in ms
	
	lp = " -L 50"  # initial low pitch bound arg for 2pass
	hp = " -H 700" # initial high pitch bound arg
	
	vf_min = 50 # minimum number voiced frames to keep 2ndpass data
	# TODO something better after implementing sth to be a ratio of e.g. nonsilence

	for rec in sample.values():
		fid = rec["file_id"]
		spk = rec["speaker"]
		wav = os.path.join(audio_dir,f'{fid}.wav')

		assert os.path.exists(wav)
		
		# sptk wants no headers input
		sptk_fmt = f"{feature_dir}{fid}.SPTK.TMP"
		_ = subprocess.run(
				f"sox -t wav {wav} -c 1 -t s16 -r 16000 {sptk_fmt}", 
				shell=True)
		
		
		# pitch - - - - - - - - - - -
		# 2-pass method of Hirst,
		# falling back to SPTK's defaults if too narrow/untrackable.
		sptk_pitch_algos = [('rapt','0'), ('swipe','1'), ('reaper','2'), ('harvest','4')]
		
		
		for aname, anum in sptk_pitch_algos:
			save_path = f'{feature_dir}{fid}.{aname}.f0'
			if not os.path.exists(save_path):
				f0_1pass_rd = run_pitch(sptk_fmt, lp, hp, anum, sptk, ts = " -t0 0.7 -t1 0.2 -t2 1.6 -t4 0.2")
				f0_1pass = [f for f in map(float,f0_1pass_rd.splitlines()) if f != 0]
				
				q1 = np.quantile(f0_1pass,0.25)
				q3 = np.quantile(f0_1pass,0.75)
				pceil = 2.5 * q3
				pfloor = 0.75 * q1
				
				f0_rd = run_pitch(sptk_fmt, f' -L {pfloor}', f' -H {pceil}', anum, sptk)
				
				
				#print(len(f0_1pass_rd),len(f0_1pass))
				if len([f for f in map(float,f0_rd.splitlines()) if f != 0]) < 50:
					f0_def = run_pitch(sptk_fmt, '', '', anum, sptk, ts='')
					print(f'{aname} 2ndpass failed for {fid}, default got {len(f0_def)}')
					f0_rd = f0_def
				
				with open(save_path, 'w') as handle:
					handle.write(f0_rd)
		
		if not os.path.exists(f'{feature_dir}{fid}.ene'):
			mfcc = subprocess.check_output(
				f"{sptk}x2x +sd {sptk_fmt} | {sptk}frame -l {sr*25} -p {sr*fs} -n 1 \
				| {sptk}dfs -b 1 -0.97 | {sptk}window -l {sr*25} -L 512 -w 1 -n 0 | \
				{sptk}mfcc -l 512 -n 40 -c 22 -m 12 -L 64 -H 4000 -o 1 | \
				{sptk}x2x +da", 
				shell=True).decode().splitlines()
			energy = mfcc[12::13] # this is the only energy output from sptk
			with open(f"{feature_dir}{fid}.ene",'w') as handle:
				handle.write('\n'.join(energy)+'\n')
			# otherwise use librosa rmse or something
		# probably will end up using parselmouth praat anyway
		
		#plp = subprocess.check_output(
		#		f"{sptk}x2x +sd {sptk_fmt} | {sptk}frame -l {sr*25} -p {sr*fs} -n 1 \
		#		| {sptk}dfs -b 1 -0.97 | {sptk}window -l {sr*25} -L 512 -w 1 -n 0 | \
		#		{sptk}plp -l 512 -n 40 -c 22 -m 12 -L 64 -H 4000 -f 0.33 -o 1 | \
		#		{sptk}x2x +da", 
		#		shell=True).decode().splitlines()
		#plpenergy = plp[12::13] # confirmed same as energy from equivalent mfcc command.
		
		
		# librosa rmse
		if not os.path.exists(f'{feature_dir}{fid}.rmse'):
			rmse = get_rmse(wav, sr, fs)
			with open(f"{feature_dir}{fid}.rmse",'w') as handle:
				handle.write('\n'.join([str(x) for x in rmse])+'\n')
		
		print('Features:', wav)
		_ = subprocess.run(["rm", sptk_fmt])
	


def feature_reader(path,ext):
# for now theyre all 1-dimensional & ext doesnt matter
#  some have -inf values

	with open(path,'r') as handle:
		feat_data = handle.read().splitlines()
	feat_data = [float(x) for x in feat_data]
	
	if ext == 'rmse':
		feat_data = feat_data[:-1] # !! they handle frames differently
	
	return feat_data
	


def get_sample_feats(sample, aligns_dir, feats_dir):
	
	def _ext(feat_file):
		ext = feat_file.split('/')[-1].split('.',1)[-1]
	
	sample_data = {}
	for rec in sample.values():
		fid = rec["file_id"]
		spk = rec["speaker"]
		
		align_path = os.path.join(aligns_dir,spk,f'{fid}.csv')
		with open(align_path) as f:
			lines = f.read().splitlines()
		lines = [l.split(',') for l in lines[1:]]
		word_aligns = [(l,float(s),float(e)) for s,e,l,t,_ in lines
						if t.lower()=='words']
		phone_aligns = [(l,float(s),float(e)) for s,e,l,t,_ in lines
						if t.lower()=='phones']
		
		rec_data = {"mfa_words": word_aligns, "mfa_phones": phone_aligns}
		
		
		feat_files = glob.glob(os.path.join(feats_dir,fid)+'.*')
		
		FLEN = None # if feature step sizes will vary, do something else...
		
		for feature_path in feat_files:
			ext = feature_path.split('/')[-1].split('.',1)[-1]
			feature_data = feature_reader(feature_path,ext)
			rec_data[ext] = feature_data
			if FLEN:
				assert len(feature_data) == FLEN
			else:
				FLEN = len(feature_data)
			
		sample_data[fid] = rec_data

	return sample_data



	
	
def prep_feats(sample_data, feature_list=['ene','swipe.f0'], pitch_rep = 0):

	feature_list = sorted(feature_list)
	
	# may be unnecessary depending how rmse and pitch window/hop are calculated already
	def downsample_f2f(feature,reference_len):
		idx = np.round(np.linspace(0, len(feature) - 1, reference_len)).astype(int)
		return feature[idx]
	
	# TODO better things
	# - cleane, normalise features etc
	# - handle better when some features are undefined at some timepoints
	# -   possible custom DTW local distance function
	def _preprocess(data,ext):
	
		# sptk trackers by default say 0 pitch when not exists
		if '.f0' in ext:
			mean = np.mean([x for x in data if x!= 0])
			std = np.std([x for x in data if x!= 0])
			
			# replace missing pitch with mean
			if not pitch_rep:
				data = [0 if x == 0 else z_score(x, mean, std) for x in data]
			
			# replace missing pitch with low value
			else:
				low_val = min(data) - 1
				data = [z_score(low_val, mean, std) if x == 0 
					else z_score(x, mean, std) for x in data]
				
		elif '.ene' in ext:
			# replace -inf with a low value?
			if -np.inf in data:
				low_val = min(data) - 1
				data = [low_val if x == -np.inf else x for x in data]
		
		return data
	
	
	clusterable_feats = {}
	for fid, fdata in sample_data.items():
	
		word_al = fdata['mfa_words']
		start_time = word_al[0][1]
		end_time = word_al[-1][2]
		# TODO allow other audio slicing
		
		# as long as features sampled every 10ms --
		s_ix,e_ix = int(start_time*100), int(end_time*100)
		
		cpy = [np.array(deepcopy(_preprocess(fdata[x],x)[s_ix:e_ix]))
					 for x in feature_list]
			
		d, ok = [], True
		while ok:
			for t in zip(*cpy):
				if -np.inf not in t:
					d.append(list(t))
			else:
				ok = False
		try:
			assert len(cpy[0])-3 <= len(d) <= len(cpy[0])
		except:
			raise Exception("There is a problem")
			# TODO something better if necessary
			# for now allow to just cut off a few frame of trailing silence
			# --> ending timestamps in word,phone aligns will not quite match
		
		clusterable_feats[fid] = d
	return clusterable_feats



def dtw_distance(x, y):
	"""
	Returns the DTW distance between two pitch sequences.
	"""  
	alignment = dtw(x, y, keep_internals=True)
	return alignment.normalizedDistance
	
	
# it will do pairs of file with self where dtw == 0
def pair_dists(data):

	dtw_dists = []
	
	fids = sorted(list(data.keys()))
	
	for fid1 in fids:
		val1 = data[fid1]
		for fid2 in fids:
			val2 = data[fid2]
			dtw_dists.append((f"{fid1}**{fid2}", dtw_distance(val1, val2)))

	return dtw_dists, fids
	
	

def make_clusters(cluster_feats, n_clusters = None):

	dtw_dists, dindex = pair_dists(cluster_feats)
	
	nrecs = len(dindex)
	X = [d[1] for d in dtw_dists]
	X = [X[i:i+nrecs] for i in range(0, len(X), nrecs)]
	X = np.array(X)

	print('finished dtw, now clustering')
	
	# TODO more clustering options?
	if n_clusters:
		cluster_range = range(n_clusters, n_clusters+1)
	else:
		maxcl = min(max(3,int(nrecs/6)),12)
		cluster_range = range(3,maxcl)
	
	# skip dynmsc for now
	#dm = kmd.dynmsc(X, maxcl, 3)
	#for k,l in zip(dm.rangek, dm.losses):
	#	print(f'\t{k}\t{l}')
	#print(f'Best k {dm.bestk}, L {dm.loss}')
	#print(dm.labels)
	
	print('====================')
	print(f'N Recordings = {nrecs}')
	cs = [(k,kmd.fasterpam(X,k)) for k in cluster_range]
	best_k, best_c = min(cs, key = lambda x: x[1].loss)
	for k,c in cs:
		bb = ' <--*' if k==best_k else ''
		print(f' k {k}, L {c.loss:.2f}{bb}')


	print('')
	
	clusters_labels = zip(dindex,best_c.labels)
	assert len(dindex)==len(best_c.labels)
	
	return list(clusters_labels)
	
	
	

# a function to do things
def f1(audioc_dir, mfaln_dir, json_path, sptk_path, tmp_dir):

	with open(json_path,'r') as handle:
		db = json.load(handle)
		
	
	# make all potential acoustic features
	# TODO put this somewhere else since it only happens once
	#db_corpus = extract_sample(db, {'':[]}) #get whole corpus
	#_ = featurise_sample(db_corpus, audioc_dir, sptk_path, tmp_dir)

	
	# define the sample for this run
	# TODO not hardcode this here
	#fc = [] # retrieve all focuses
	focus_pos = [1,4] # word indices start 0
	sent_ids = ['0025', '0193', '0180', 'ö171', '0008', '0096', '0030', '0162', '0104']
	#sent_ids = ['0082', '0117', '0122']#, '0149']
	
	#! may need to vary focus positions by sentence e.g. different word counts
	textfilter = {sid : focus_pos for sid in set(sent_ids)}
					
	db_sample = extract_sample(db, textfilter)
		
	# read acoustic features for clustering
	sample_feats = get_sample_feats(db_sample, mfaln_dir, tmp_dir)
	#for k,v in sample_feats['0008_4_i'].items():
	#	print(k)
	#	print(v)
	#import fish
	
	#clusterable_feats = prep_feats(sample_feats)
	clusterable_feats = prep_feats(sample_feats, feature_list=['rmse','swipe.f0'])
	#clusterable_feats = prep_feats(sample_feats, feature_list=['ene','reaper.f0', 'harvest.f0'])
	#clusterable_feats = prep_feats(sample_feats, feature_list=['swipe.f0'])
	# pitch rep 0 seems to be better than low for now
	
	clusters = make_clusters(clusterable_feats, n_clusters = 8)
	sc = sorted(clusters, key = lambda x: x[1])
	for r,c in sc:
		print(r,c,emt(db_sample[r]['sentence_text'],db_sample[r]['focus_index']))
	
	
	# TODO spectral centroids next?
	# TODO extract local relative rate as feature
	# TODO visual display results
	
# measures to quickly estimate clustering quality...
# average number of unique speakers per cluster,
#  unique sentences per cluster,
#  average percent of each cluster's sentences whose focus was that cluster's most common focus?
# TODO learn optimal weights of each feature/dimension
#  - > need custom local dist func in DTW whose parameters fit based on clustering outome metric,
#     but the DTW probably allows that



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
	
	sptk_bin = '~/work/env/SPTK/bin/'
	
	# store extracted speech features here
	feats_dir = './tmp2/'
	
	audioc_dir, mfaln_dir, json_path, sptk_bin, feats_dir = [os.path.expanduser(x) 
		for x in [audioc_dir, mfaln_dir, json_path, sptk_bin, feats_dir]]
		
	f1(audioc_dir, mfaln_dir, json_path, sptk_bin, feats_dir)





