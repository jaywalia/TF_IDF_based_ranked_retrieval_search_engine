#							TF-IDF BASED RANKED RETRIEVAL
#								CS F469 ASSIGNMENT 1
#									-TARUN RAHEJA
#									-2015A7PS0106H
#
############################################################################################################################################

import nltk
import time
from math import log, sqrt

# To avoid the hassle of creating a new empty dictionary entry everytime in case of new word, defaultdict is used.
from collections import defaultdict

total_doc_count = 14577
inv_index = defaultdict(list) # Returns empty list whenever non-existent element is accessed.
all_doc_vectors = []  # Each element of list is a dictionary, there will exist a vector for each doc.
doc_freq = {}  

#Adds token-frequency dicts as list elements to the list called all_doc_vectors.
def read_all_docs():
	for doc_id in range(total_doc_count - 1):
       		doc_text = doc_string(doc_id)
        	token_list = stem_and_tokenize(doc_text)
        	v = create_vector(token_list)
        	all_doc_vectors.append(v)

#Creates token-frequecy dictionary from the input query.
def input_vector(query):
	v = {}
	for word in query:
        	if word in v:
        		v[word] += 1.0 # Floats since they will be converted to TF-IDF later.
        	else:
        		v[word] = 1.0
	return v

#Generates inverted index for all documents.
def inv_index_all_docs():
	count = 0
	for doc_vector in all_doc_vectors:
	        for word in doc_vector:
        		inv_index[word].append(count)# Here defaultdict shows its value, returns 0.
        	count += 1

#Changes all token-frequency vectors to TF-IDF vectors.
def tf_idf_vectorize():
	length = 0.0
	for doc_vector in all_doc_vectors:
        	for word in doc_vector:
        		frequency = doc_vector[word]
        		score = tf_idf_score(word, frequency)
    			doc_vector[word] = score
        		length += score ** 2
        	length = sqrt(length)
        	for word in doc_vector:
        		doc_vector[word] /= length

#Calculates the TF-IDF vector for the query in specific.
def tf_idf_query(query_vector):
	length = 0.0
	for word in query_vector:
		frequency = query_vector[word]
   		if word in doc_freq: 
        		query_vector[word] = tf_idf_score(word, frequency)
        	else:
        		query_vector[word] = log(1 + frequency) * log(total_doc_count)  
        	length += query_vector[word] ** 2
	length = sqrt(length)
	if length != 0:
	        for word in query_vector:
        		query_vector[word] /= length

#Calculates TF-IDF score, give TF and DF values.
def tf_idf_score(word, frequency):
	return log(1 + frequency) * log(total_doc_count / doc_freq[word])

#Calculates the dot product of two given vectors.
def dot_product(vector_a, vector_b):
	if len(vector_a) > len(vector_b): # Swapping to ensure that left dict is always smaller.
	        temp = vector_a
	        vector_a = vector_b
	        vector_b = temp
	key_list_a = vector_a.keys()
	key_list_b = vector_b.keys()
	sum = 0
	for key in key_list_a:
	        if key in key_list_b:
	        	sum += vector_a[key] * vector_b[key]
	return sum


# Returns list of string tokens after stemming.
def stem_and_tokenize(doc_text):
	token_list = nltk.word_tokenize(doc_text)
	ps = nltk.stem.PorterStemmer()
	result = []
	for word in token_list:
	        result.append(ps.stem(word))
	return result

#Creates token-frequency vector from input string.
def create_vector(token_list):
	v = {}
	global doc_freq
	for token in token_list:
	        if token in v:
        		v[token] += 1
	        else:
        		v[token] = 1
            		if token in doc_freq:
                		doc_freq[token] += 1
            		else:
               		 	doc_freq[token] = 1
	return v

#Reads data from a bunch of files in the Dataset in the folder called Files in the same directory.
def doc_string(doc_id):
	try:
	        file_text = str(open("Files/" + str(doc_id)).read())
	except:
	        file_text = ""
	file_text = unicode(file_text, errors='replace')
	return file_text

#Returns a list of document IDs sorted on basis of cosine similarity.
def query_result(query_vector):
	answer = []
	for doc_id in range(total_doc_count - 1):
	        dp = dot_product(query_vector, all_doc_vectors[doc_id])
        	answer.append((doc_id, dp))
        answer = sorted(answer, key=lambda x: x[1], reverse = True)
	return answer

#Execution starts from here.
start_index_time = time.time()
read_all_docs()
inv_index_all_docs()
tf_idf_vectorize()
#Preprocessing ends here.

print "Indexing took " + str(time.time()-start_index_time) + " seconds."

#Query input to be taken.
while True:
	query = raw_input("Enter a search query:")    
	if len(query) == 0:
		break
	start_search_time = time.time()
	token_list = stem_and_tokenize(query)
	query_vector = input_vector(token_list)
	tf_idf_query(query_vector)
	result = query_result(query_vector)
	print "This query took " + str(time.time()-start_search_time) + " seconds."
	f_result = result[:10]
	for element in f_result:
		print("The DocID " + str(element[0]) + " matches, with weight " + str(element[1]))
