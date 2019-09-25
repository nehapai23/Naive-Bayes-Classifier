import sys
import re
import random
import math 
import json
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------
# FUNCTION: read_txt(filename)
# OUTPUT: Dictionary
# DESCRIPTION: Reads data in filename and pre-processes it to return a dictionary
#              of documents with sentences as keys and labels as values
#-----------------------------------------------------------------------------------
def read_txt(filename):
    dataset_dictionary = {}
    with open(filename, 'r') as f:
        for document in f:
            document = re.sub(r'[^a-zA-Z0-9 \t]','',document)
            document,label = document.lower().rstrip().split('\t')
            dataset_dictionary[document] = label
    return dataset_dictionary


#-----------------------------------------------------------------------------------
# FUNCTION: split_data(dataset, train_percentage)
# OUTPUT: 4 Lists
# DESCRIPTION: Reads dataset and training set split percentage and returns training 
#              and test sets
#-----------------------------------------------------------------------------------
# def split_data(dataset, train_percentage):
#     _keys = list(dataset.keys())
#     random.shuffle(_keys)
#     _values = [dataset[_key] for _key in _keys]
#     split_per = int(train_percentage/100 * len(_keys))
#     return _keys[:split_per], _keys[split_per:], _values[:split_per], _values[split_per:]


#-----------------------------------------------------------------------------------
# FUNCTION: stratified_split_data(dataset, train_percentage)
# OUTPUT: X_train, X_test, y_train, y_test 
# DESCRIPTION: Reads dataset, iteration number and total number of folds(k) for 
#              k-fold validation and returns train_i and test_i
#-----------------------------------------------------------------------------------
def stratified_split_data(dataset,i,k):
    ### Split dataset into positive and negative keys
    _posKeys = [doc for doc in dataset if dataset[doc] == '1']
    _negKeys = [doc for doc in dataset if dataset[doc] == '0']
    ### Perform random shuffling 
    random.shuffle(_posKeys)
    random.shuffle(_negKeys)
    _posPortion = int(len(_posKeys)/k)
    _negPortion = int(len(_negKeys)/k)
    _keys,_values = [],[] 
    pos1,pos2 = 0,0    
    ### Take x-part of postive and negative parts 
    for x in range(0,k):
        _keys.insert(x,_posKeys[pos1:pos1+_posPortion])
        pos1 = pos1+_posPortion
        _keys[x] = _keys[x] + (_negKeys[pos2:pos2+_negPortion])
        pos2 = pos2+_negPortion
        _values.insert(x,[dataset[_key] for _key in _keys[x]]) 
    _keys = [x for x in _keys if x != []]
    _values = [x for x in _values if x != []]
    ### For i-th iteration ith part becomes the test set and remaining are training sets
    return utility_flattenlist(_keys[:i]+_keys[i+1:]), _keys[i], utility_flattenlist(_values[:i]+_values[i+1:]), _values[i]

#-----------------------------------------------------------------------------------
# FUNCTION: logProduct(ipList, estimates)
# OUTPUT: Float value
# DESCRIPTION: Calculates log product of Estimate of Token Parameters for prediction
#-----------------------------------------------------------------------------------
def logProduct(ipList, estimates): 
    result = 1
    for x in ipList: 
        if x in estimates:
            if estimates[x] != 0.0:
                result = result * math.log(estimates[x])
            else: 
                result = 0.0  
    return result

#-----------------------------------------------------------------------------------
# FUNCTION: utility_flattenlist(ip_list)
# OUTPUT: list
# DESCRIPTION: Returns a flattened list from a list of lists
#-----------------------------------------------------------------------------------
def utility_flattenlist(ip_list):
    return [item for sublist in ip_list for item in sublist]

#-----------------------------------------------------------------------------------
# FUNCTION: utility_log(num)
# OUTPUT: float value
# DESCRIPTION: Returns log of input number
#-----------------------------------------------------------------------------------
def utility_log(num):
    if num == 0:
        return -float("inf")
    else: 
        return math.log(num)

#-----------------------------------------------------------------------------------
# FUNCTION: calculate_estimate(vocab, pos_tokens, neg_tokens, smoothing_param)
# OUTPUT: dictionary with keys 'word|pos' and 'word|neg'
# DESCRIPTION: Calculates the estimates of token parameters in the vocab according 
#              to smoothing parameter given by, p(w|c) = #(w^c)+m / #(c)+mV where V 
#              is vocab size, c is class, m is smoothing parameter and w is word
#-----------------------------------------------------------------------------------
def calculate_estimate(vocab, pos_tokens, neg_tokens, smoothing_param):
    estimate = {'word|pos':{}, 'word|neg':{}}
    for token in vocab:
        if len(pos_tokens) != 0:
            estimate['word|pos'][token] = (pos_tokens.count(token)+smoothing_param)/(len(pos_tokens)+smoothing_param*len(vocab))
        else:
            estimate['word|pos'][token] = 0
        if len(neg_tokens) != 0:
            estimate['word|neg'][token] = (neg_tokens.count(token)+smoothing_param)/(len(neg_tokens)+smoothing_param*len(vocab))
        else:
            estimate['word|neg'][token] = 0
    return estimate

#-----------------------------------------------------------------------------------
# FUNCTION: predict(X_test,y_test,pos_maxL,neg_maxL,estimate)
# OUTPUT: dictionary
# DESCRIPTION: Returns an accuracy of predictions using estimate values
#-----------------------------------------------------------------------------------
def predict(X_test,y_test,pos_maxL,neg_maxL,estimate):
    result_dict = {'Correct':0, 'Incorrect':0, 'Accuracy':0}
    for i,doc in enumerate(X_test):
        prediction = -1
        pos_class_prediction = utility_log(pos_maxL) * logProduct(doc.split(" "),estimate['word|pos'])
        neg_class_prediction = utility_log(neg_maxL) * logProduct(doc.split(" "),estimate['word|neg'])
        actual = int(y_test[i])
        if pos_class_prediction > neg_class_prediction:
            prediction = 1
        elif pos_class_prediction == neg_class_prediction:
            prediction = -1
        else:
            prediction = 0

        if actual == prediction:
            result_dict['Correct'] += 1
        else:
            result_dict['Incorrect'] += 1  
        result_dict['Accuracy'] = result_dict['Correct']*100/len(X_test)         
    return result_dict['Accuracy']

#-----------------------------------------------------------------------------------
# FUNCTION: naive_bayes(X_train, X_test, y_train, y_test, m)
# OUTPUT: list
# DESCRIPTION: Function to implement naive bayes classification model
#-----------------------------------------------------------------------------------
def naive_bayes(X_train, X_test, y_train, y_test, m):
    ### Creating set of all words in training set
    vocab = set()
    neg_docs, pos_docs, pos_tokens, neg_tokens = 0,0,[],[]  
    for i,doc in enumerate(X_train):
        vocab.update(doc.split(" "))
        vocab.discard(" ")
        ### Finding count of tokens in positive and negative classes
        if y_train[i] == '0':
            neg_docs += 1
            neg_tokens.extend(doc.split(" "))
            neg_tokens = [x for x in neg_tokens if x!='']
        else:   
            pos_docs += 1
            pos_tokens.extend(doc.split(" "))
            pos_tokens = [x for x in pos_tokens if x!='']
    pos_maxL, neg_maxL = 0.0, 0.0
    if len(vocab) != 0:
        pos_maxL, neg_maxL = pos_docs/len(vocab), neg_docs/len(vocab)

    results = []
    for i in range(0,len(m)):
        estimate = calculate_estimate(vocab, pos_tokens, neg_tokens, m[i])
        results.append(predict(X_test,y_test,pos_maxL,neg_maxL,estimate))
    return results

#-----------------------------------------------------------------------------------
# FUNCTION: mean(data)
# OUTPUT: float value
# DESCRIPTION: Returns mean
#-----------------------------------------------------------------------------------
def mean(data):
    return sum(data)/len(data)


#-----------------------------------------------------------------------------------
# FUNCTION: sum_of_squares(data)
# OUTPUT: float value
# DESCRIPTION: Returns sum of squares
#-----------------------------------------------------------------------------------
def sum_of_squares(data):
    avg = mean(data)
    return sum((d-avg)**2 for d in data)


#-----------------------------------------------------------------------------------
# FUNCTION: stddev(data)
# OUTPUT: float value
# DESCRIPTION: Returns standard deviation
#-----------------------------------------------------------------------------------
def stddev(data):
    return sum_of_squares(data)/len(data)


#-----------------------------------------------------------------------------------
# FUNCTION: experiment1(result,k,sample_size_list)
# OUTPUT: Graph Plot
# DESCRIPTION: Plots learning curves of model
#-----------------------------------------------------------------------------------
def experiment1(result,k,sample_size_list):
    maxl = result['MaxL'] 
    mapl = result['MAP']
    maxLAccuracies, mapAccuracies, maxlsd, mapsd = [],[],[],[]
    for i in range(0,k):
        t1,t2 = [],[]
        for j in range(0,k):
            t1.append(maxl['Fold_'+str(j+1)][i]['Sample_'+str(i+1)])
            t2.append(mapl['Fold_'+str(j+1)][i]['Sample_'+str(i+1)])
        maxLAccuracies.append(mean(t1))    
        maxlsd.append(stddev(t1))    
        mapAccuracies.append(mean(t2))    
        mapsd.append(stddev(t2))    

    print("Accuracies : ", maxLAccuracies, mapAccuracies)
    print("STD :", maxlsd, mapsd)
    plt.errorbar(x=sample_size_list[:k], y=maxLAccuracies, yerr=maxlsd, color='blue', marker='s', mfc='blue', mec='blue', ms=4, mew=4)
    plt.errorbar(x=sample_size_list[:k], y=mapAccuracies, yerr=mapsd, color = 'red', marker='s', mfc='red', mec='red', ms=4, mew=4)
    plt.xlabel('Training set Size')
    plt.ylabel("Classification Accuracy(%)")
    plt.title("Stratifed Cross Validation Learning Curves with m = 0 and m = 1")
    plt.show()


#-----------------------------------------------------------------------------------
# FUNCTION: experiment2(result_exp2,k,list_m_exp2)
# OUTPUT: Graph Plot
# DESCRIPTION: Plots Stratified cross validation with m= 0.1,0.2,0.3…,0.9,1,2,3… 9
#-----------------------------------------------------------------------------------
def experiment2(result_exp2,k,list_m_exp2):
    accuracies, sd = [],[]
    for i in range(0,len(list_m_exp2)):
        t1 = []
        for j in range(0,k):
            t1.append(result_exp2['Fold_'+str(j+1)][i])
        accuracies.append(mean(t1))    
        sd.append(stddev(t1))    
        
    print("Accuracies : ",accuracies)
    print("STD :", sd)
    plt.errorbar(x=list_m_exp2, y=accuracies, yerr=sd, color='blue', marker='s', mfc='blue', mec='blue', ms=4, mew=4)
    plt.xlabel('Smoothing Parameter')
    plt.ylabel("Classification Accuracy(%)")
    plt.title("Stratifed Cross Validation with m = 0,0.1,0.2..1,2,3,...9,10")
    plt.show()


if __name__ == '__main__':
    input_dataset = read_txt(sys.argv[1])
    
    ### Splitting data in 70:30 ratio   
    # X_train, X_test, y_train, y_test = split_data(input_dataset, 70)
    # result = naive_bayes(X_train, X_test, y_train, y_test, [0])

    ### Splitting data using 10 fold stratified cross validation and applying 
    k = 10
    list_m_exp2 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
    result = {'MaxL':{}, 'MAP':{}}
    result_exp2 = {}
    sample_size_list = []
    X_train, X_test, y_train, y_test = [],[],[],[]
    sample_result_dict1,sample_result_dict2,sample_result_dict3  = dict(), dict(), dict()
    for i in range(0,k):
        ### Generating train_i and test_i for i = 1,...,k
        X_tr, X_te, y_tr, y_te = stratified_split_data(input_dataset,i,k)
        X_train.append(X_tr)
        X_test.append(X_te)
        y_train.append(y_tr)
        y_test.append(y_te)
    for p in range(0,k):
        ### Generating train_i and test_i for i = 1,...,k
        N = len(X_train[p])
        subsample_result_dict1,subsample_result_dict2  = [], []
        for subsample in range(0,10,1):
            sample_size = int((subsample+1)*N/10)
            sample_size_list.append(sample_size)
            result_sample = naive_bayes(X_train[p][:sample_size], X_test[p], y_train[p][:sample_size], y_test[p], [0,1])
            subsample_result_dict1.append({'Sample_'+str(subsample+1) : result_sample[0]})
            subsample_result_dict2.append({'Sample_'+str(subsample+1) : result_sample[1]})
        sample_result_dict1['Fold_'+str(p+1)] = subsample_result_dict1
        sample_result_dict2['Fold_'+str(p+1)] = subsample_result_dict2
        sample_result_dict3['Fold_'+str(p+1)] = naive_bayes(X_train[p], X_test[p], y_train[p], y_test[p], list_m_exp2)
    result['MaxL'] = sample_result_dict1 
    result['MAP'] = sample_result_dict2
    result_exp2 = sample_result_dict3
    # print(result)
    experiment1(result,k,sample_size_list)
    experiment2(result_exp2,k,list_m_exp2)