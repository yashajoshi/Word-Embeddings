import os, sys, re, csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from nltk.corpus import stopwords

nltk.download
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit



#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.

#... (4) Test your model. Compare cosine similarities between learned word vectors.

#...(5) Feel free make minor change to the structure or input of the skeleton code (P.S you still need to finish all the task and implemenent word2vec correctly)










#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from






#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................



def loadData(filename):
	global uniqueWords, wordcodes, wordcounts
	override = False
	if override:
		#... for debugging purposes, reloading input file and tokenizing is quite slow
		#...  >> simply reload the completed objects. Instantaneous.
		fullrec = pickle.load(open("w2v_fullrec.p","rb"))
		wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
		uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
		wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
		return fullrec


	# ... load in first 15,000 rows of unlabeled data file.  You can load in
	# more if you want later (and should do this for the final homework)
	handle = open(filename, "r", encoding="utf8")
	fullconts = handle.read().split("\n")
	fullconts = fullconts[1:15000]  # (TASK) Use all the data for the final submission
	#... apply simple tokenization (whitespace and lowercase)
	fullconts = [" ".join(fullconts).lower()]





	print ("Generating token stream...")
	#... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
	#... ignore stopwords in this process
	#... for simplicity, you may use nltk.word_tokenize() to split fullconts.
	#... keep track of the frequency counts of tokens in origcounts.
	word_tokens = nltk.word_tokenize((fullconts[0]))
	stop_words = set(stopwords.words('english'))
	min_count = 50

	fullrec = [w for w in word_tokens if not w in stop_words]
	origcounts = Counter(fullrec)





	print ("Performing minimum thresholding..")
	#... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
	#... replace other terms with <UNK> token.
	#... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
	fullrec_filtered = [w if origcounts[w] >= min_count else "<UNK>" for w in fullrec]


	#... after filling in fullrec_filtered, replace the original fullrec with this one.
	fullrec = fullrec_filtered
	wordcounts = Counter(fullrec)






	print ("Producing one-hot indicies")
	#... (TASK) sort the unique tokens into array uniqueWords
	#... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
	#... replace all word tokens in fullrec with their corresponding one-hot indices.
	uniqueWords = set(fullrec)  # ... fill in
	for i, word in enumerate(uniqueWords):
		wordcodes[word] = i
	fullrec = list(map(lambda x: wordcodes[x], fullrec))






	#... close input file handle
	handle.close()



	#... store these objects for later.
	#... for debugging, don't keep re-tokenizing same data in same way.
	#... just reload the already-processed input data with pickles.
	#... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows
	
	pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
	pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
	pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
	pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


	#... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
	return fullrec







#.................................................................................
#... compute sigmoid value
#.................................................................................
@jit
def sigmoid(x):
	return 1.0/(1+np.exp(-x))









#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
	#global wordcounts
	#... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
	max_exp_count = 0


	print ("Generating exponentiated count vectors")
	#... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
	#... store results in exp_count_array.
	exp_count_array = [math.pow(wordcounts[t], exp_power) for t in uniqueWords]
	max_exp_count = sum(exp_count_array)



	print ("Generating distribution")

	#... (TASK) compute the normalized probabilities of each term.
	#... using exp_count_array, normalize each value by the total value max_exp_count so that
	#... they all add up to 1. Store this corresponding array in prob_dist
    # prob_dist = exp_count_array / max_exp_count
	prob_dist = list(map(lambda x: float(x / max_exp_count), exp_count_array))





	print ("Filling up sampling table")
	#... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
	#... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
	#... multiplied by table_size. This table should be stored in cumulative_dict.
	#... we do this for much faster lookup later on when sampling from this table.

	#cumulative_dict = #... fill in
	table_size = 1e7
	word_freqs = [int(p * table_size) for p in prob_dist]
	cumulative_dict = {}

	j = 0
	for ind, freq in enumerate(word_freqs):
		i = 0
		while i < freq:
			cumulative_dict[j] = ind
			i += 1
			j += 1
	return cumulative_dict






#.................................................................................
#... generate a specific number of negative samples
#.................................................................................


def generateSamples(context_idx, num_samples):
	global samplingTable, uniqueWords, randcounter
	results = []
	#... (TASK) randomly sample num_samples token indices from samplingTable.
	#... don't allow the chosen token to be context_idx.
	#... append the chosen indices to results
	while len(results) < num_samples:
		indice = np.random.randint(low=0, high=len(samplingTable))
		val = samplingTable[indice]
		if context_idx == val:
			continue
		else:
			results.append(val)

	return results




@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, context_words,W1,W2,negative_indices):
	# sequence chars was generated from the mapped sequence in the core code
	nll_new = 0
		#... (TASK) implement gradient descent. Find the current context token from context_words
		#... and the associated negative samples from negative_indices. Run gradient descent on both
		#... weight matrices W1 and W2.
		#... compute the total negative log-likelihood and store this in nll_new.
		#... You don't have to use all the input list above, feel free to change them
		
# 	for k in range(0, len(sequence_chars)):
# 		h = W1[center_token]

# 		v_j_old_dash_context_word = np.copy(W2[sequence_chars[k]])
# 		v_j_old_dash_negative_one = np.copy(W2[negative_indices[num_samples * k]])
# 		v_j_old_dash_negative_two = np.copy(W2[negative_indices[num_samples * k + 1]])

# 		W2[sequence_chars[k]] = W2[sequence_chars[k]] - learning_rate * (
#             sigmoid(np.dot(W2[sequence_chars[k]], h)) - 1) * h
# 		W2[negative_indices[num_samples * k]] = W2[negative_indices[num_samples * k]] - learning_rate * (
#             sigmoid(np.dot(W2[negative_indices[num_samples * k]], h)) - 0) * h
# 		W2[negative_indices[num_samples * k + 1]] = W2[negative_indices[num_samples * k + 1]] - learning_rate * (
#             sigmoid(np.dot(W2[negative_indices[num_samples * k + 1]], h)) - 0) * h

# 		W1[center_token] = W1[center_token] - learning_rate * (
#             (sigmoid(np.dot(v_j_old_dash_context_word, h)) - 1) * v_j_old_dash_context_word + sigmoid(
#                 np.dot(v_j_old_dash_negative_one, h)) * v_j_old_dash_negative_one +
#             sigmoid(np.dot(v_j_old_dash_negative_two, h)) * v_j_old_dash_negative_two)

# 		nll_first_term = -np.log(sigmoid(np.dot(W2[sequence_chars[k]], W1[center_token])))
# 		nll_second_term = 0
# 		for i in range(num_samples):
# 			nll_second_term += np.log(sigmoid(-np.dot(W2[negative_indices[num_samples * k + i]], W1[center_token])))
# 		nll_new += nll_first_term - nll_second_term
	chunks = [negative_indices[x:x+2] for x in range(0, len(negative_indices), 2)]
	for k in range(0, len(context_word_ids)):
		neg_ll_total = 0
		context_index = context_word_ids[k]
		h = np.array(W1[center_token])
		W2_p = np.array(W2[context_index])

		#  Updating W2 for the postive context sample
		s = sigmoid(np.dot(W2[context_index], h))
		W2[context_index] = W2_p - (learning_rate * ((s - 1) * h))

		tot_p_neg = 0

		# iterating over the negative samples for the given context word
		for neg in chunks[k]:
			#  Updating W prime for the two negtive samples
			W2_p_neg = np.array(W2[neg])
			s_neg = sigmoid(np.dot(W2[neg], h))
			W2[neg] = W2_p_neg - (learning_rate * ((s_neg - 0) * h))

			tot_p_neg += (sigmoid(np.dot(W2_p_neg, h) - 0) * W2_p_neg)

			# Negative LL for both the negative samples
			nsig = sigmoid(np.negative(np.dot(W2[neg], h)))
			neg_ll_total += np.log(nsig)

		#  Updating W1 for the center token
		s2_pos = sigmoid(np.dot(W2_p, h))
		pos_vj = (s2_pos - 1)* W2_p
		total_vj = (pos_vj + tot_p_neg)
		W1[center_token] = h - (learning_rate * total_vj)

		#  calculating the negtive LL for the postive context token
		pos = (np.negative(np.log(sigmoid(np.dot(W2[context_index], h)))))
		nll_new += pos - neg_ll_total


	return [nll_new]









#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1 = None, curW2=None):
	global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
	vocab_size = len(uniqueWords)           #... unique characters
	hidden_size = 100                       #... number of hidden neurons
	context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
	nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations
	iteration_number = []

	#... determine how much of the full sequence we can use while still accommodating the context window
	start_point = int(math.fabs(min(context_window)))
	end_point = len(fullsequence)-(max(max(context_window),0))
	mapped_sequence = fullsequence



	#... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
	if curW1==None:
		np_randcounter += 1
		W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
		W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
	else:
		#... initialized from pre-loaded file
		W1 = curW1
		W2 = curW2



	#... set the training parameters
	epochs = 5
	num_samples = 2
	learning_rate = 0.05
	nll = 0
	iternum = 0




	#... Begin actual training
	for j in range(0,epochs):
		print ("Epoch: ", j)
		prevmark = 0

		#... For each epoch, redo the whole sequence...
		for i in range(start_point,end_point):

			if (float(i) / len(mapped_sequence)) >= (prevmark + 0.1):
				print("Progress: ", round(prevmark + 0.1, 1))
				prevmark += 0.1
			if iternum % 10000 == 0:
				print("Negative likelihood: ", nll)
				print("Iteration Number: ", iternum)
				nll_results.append(nll)
				iteration_number.append(iternum)
				nll = 0


			#... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
			if wordcodes["<UNK>"] == mapped_sequence[i]:
				continue
			center_token = mapped_sequence[i] #... fill in
			#... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.





			iternum += 1
			#... now propagate to each of the context outputs
			#for k in range(0, len(context_window)):

				#... (TASK) Use context_window to find one-hot index of the current context token.
				#context_index = #... fill in
			mapped_context = [mapped_sequence[i + ctx] for ctx in context_window]
			negative_indices = []
			for q in mapped_context:
				negative_indices += generateSamples(q, num_samples)


				#... construct some negative samples
				#negative_indices = generateSamples(context_index, num_samples)

				#... (TASK) You have your context token and your negative samples.
				#... Perform gradient descent on both weight matrices.
				#... Also keep track of the negative log-likelihood in variable nll.

			[nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1, W2,negative_indices)
			nll += nll_new


	for i in range(len(nll_results)):
		stri += str(nll_results[i]) + "\t" + str(iteration_number[i]) + "\n"
	file_handle.write(stri)
    
	return [W1,W2]



#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
	handle = open("saved_W1.data(4)","rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2.data(4)","rb")
	W2 = np.load(handle)
	handle.close()
	return [W1,W2]






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
	handle = open("saved_W1.data","wb+")
	np.save(handle, W1, allow_pickle=False)
	handle.close()

	handle = open("saved_W2.data","wb+")
	np.save(handle, W2, allow_pickle=False)
	handle.close()






#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.






#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
	global word_embeddings, proj_embeddings
	if preload:
		[curW1, curW2] = load_model()
	else:
		curW1 = None
		curW2 = None
	[word_embeddings, proj_embeddings] = trainer(curW1,curW2)
	save_model(word_embeddings, proj_embeddings)









#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

# def morphology(word_seq):
# 	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
# 	embeddings = word_embeddings
# 	learned_xfix = defaultdict(list)
# 	for x in word_seq:
# 		if x[1].find(x[0]) < 0:
# 			continue
# 		key = x[1].replace(x[0], '<WORD>')
# 		vec = embeddings[wordcodes[x[1]]] - embeddings[wordcodes[x[0]]]
# 		learned_xfix[key].append(vec)
# 	for dictkey in learned_xfix:
# 		learned_xfix[dictkey] = np.asarray(learned_xfix[dictkey]).mean(0)
# 	return learned_xfix
# 	# vector_math = vectors[0]+vectors[1]	
# 	#... find whichever vector is closest to vector_math
# 	#... (TASK) Use the same approach you used in function prediction() to construct a list
# 	#... of top 10 most similar words to vector_math. Return this list.

# def get_or_inpute_vector(testword, traineddata, do_inpute=False):
# 	global uniqueWords, word_embeddings, wordcodes
# 	if do_inpute == True:
# 		pass
# 	if testword in uniqueWords:
# 		return word_embeddings[wordcodes[testword]]
# 	else:







#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [embeddings[wordcodes[word_seq[0]]],
	embeddings[wordcodes[word_seq[1]]],
	embeddings[wordcodes[word_seq[2]]]]
	vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
	#... find whichever vector is closest to vector_math

	vectorized_list = []
	for word in uniqueWords:
		word_idx = uniqueWords.index(word)
		full_vector_math = vector_math + sum(proj_embeddings[word_idx])
		vectorized_list.append((word, full_vector_math))

	vectorized_list.sort(key=lambda x: abs(x[1]))
	new_vectorized_list = vectorized_list[:10]

	return new_vectorized_list


#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................


def get_neighbors(target_word):
	global word_embeddings, uniqueWords, wordcodes
	targets = [target_word]
	outputs = []
	#... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
	#... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
	#... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
	#... return a list of top 10 most similar words in the form of dicts,
	#... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}
	pred = {}

	for uniqueWord in uniqueWords:
		cos_similarity = abs(1 - scipy.spatial.distance.cosine(word_embeddings[wordcodes[uniqueWord]],
                                                               word_embeddings[wordcodes[target_word]]))
		pred[uniqueWord] = cos_similarity

	return dict(Counter(pred).most_common(11))


def calculate_cosine_similarity(word1, word2):
	global word_embeddings, uniqueWords, wordcodes

	cos_similarity = abs(1 - scipy.spatial.distance.cosine(word_embeddings[wordcodes[word1]],
                                                           word_embeddings[wordcodes[word2]]))
	return cos_similarity

def output(filename):
    global word_embeddings, uniqueWords, wordcodes
    file = open(filename)
    output = open('intrinsic-output.csv', 'w')
    output.write('id,sim\n')
    for x in file.readlines()[1:]:
        words = x.rstrip('\n').split('\t')
        s1_idx = uniqueWords.index(words[1])
        s2_idx = uniqueWords.index(words[2])
        distance = cosine(word_embeddings[s1_idx], word_embeddings[s2_idx])
        word_similarity = 1 - distance
        output.write('{},{},{}\n'.format(words[0], word_similarity))
    intr.close()
    output.close()

def learn_morphology(train_data):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	s_suffix = []
	for d in train_data:
		s_suffix.append(word_embeddings[wordcodes[d[0]]] - word_embeddings[wordcodes[d[1]]])
	return np.mean(s_suffix)


def knearest(target_word_vector, k, morphological_variation):
	global word_embeddings, uniqueWords, wordcodes
	pred = {}
	for uniqueWord in uniqueWords:
		cos_similarity = abs(1 - scipy.spatial.distance.cosine(word_embeddings[wordcodes[uniqueWord]],
                                                               target_word_vector))
		pred[uniqueWord] = cos_similarity

	k_nearest = dict(Counter(pred).most_common(k + 1))
	print(k_nearest)
	i = 0
	for key in k_nearest:
		i += 1
		if morphological_variation == key:
			return i


def get_or_impute_vector(test_data, suffix_vector):
	global word_embeddings, uniqueWords, wordcodes
	precision_k = []
	for d in test_data:
		total_word_vector = word_embeddings[wordcodes[d[1]]] + suffix_vector
		precision_k.append(knearest(total_word_vector, 20, d[0]))
	return precision_k




if __name__ == '__main__':
	if len(sys.argv)==2:
		filename = sys.argv[1]  # feel free to raed the file in a different way
		#... load in the file, tokenize it and assign each token an index.
		#... the full sequence of characters is encoded in terms of their one-hot positions

		fullsequence= loadData(filename)
		print ("Full sequence loaded...")
		#print(uniqueWords)
		#print (len(uniqueWords))



		#... now generate the negative sampling table
		print ("Total unique words: ", len(uniqueWords))
		print("Preparing negative sampling table")
		samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


		#... we've got the word indices and the sampling table. Begin the training.
		#... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
		#... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
		#... ... and uncomment the load_model() line

		train_vectors(preload=False)
		[word_embeddings, proj_embeddings] = load_model()
		output(filename)


		#... we've got the trained weight matrices. Now we can do some predictions
		#...pick ten words you choose
# 		targets = ["good", "bad", "food", "apple",'tasteful','unbelievably','uncle','tool','think']
# 		for targ in targets:
# 			print("Target: ", targ)
# 			bestpreds= (prediction(targ))
# 			for pred in bestpreds:
# 				print (pred["word"],":",pred["score"])
# 			print ("\n")
		targets = ["good", "bad", "food", "apple","fresh","yummy","water","meal","look","amazing"]
		file_handle = open("prob7_output.txt", "w")
		stri = "target_word,similar_word,similar_score" + "\n"
		for targ in targets:
			bestpreds = prediction(targ)
			for word in bestpreds:
				stri += targ + "," + word + "," + str(bestpreds[word]) + "\n"
		file_handle.write(stri)


		#... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
		#print (analogy(["apple", "fruit", "banana"]))



		#... try morphological task. Input is averages of vector combinations that use some morphological change.
		#... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
		#... the morphology() function.
		#... this is the optional task, if you don't want to finish it, common lines from 545 to 556

# 		s_suffix = [word_embeddings[wordcodes["banana"]] - word_embeddings[wordcodes["bananas"]]]
# 		others = [["apples", "apple"],["values", "value"]]
# 		for rec in others:
# 			s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
# 		s_suffix = np.mean(s_suffix, axis=0)
# 		print (morphology([s_suffix, "apples"]))
# 		print (morphology([s_suffix, "pears"]))






	else:
		print ("Please provide a valid input filename")
		sys.exit()


