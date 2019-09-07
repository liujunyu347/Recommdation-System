from pyspark import SparkContext
from pyspark import SparkConf
import math
import time
import itertools
import random
from sys import argv
start = time.time()
input_file = "yelp_train.csv"
output_file = "junyu_liu_task1.csv"
# input_file = argv[1]
# output_file = argv[2]
def strip_header(line):
    if "user_id, business_id, stars" not in line:
        return line
def build_hash_functions():
    functions = dict()
    for i in range(30):
        a = [283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,389,397,401,409,419,421,431,433,439,443,449,479,487,491,499,503,509]
        b = [983,953,947,941,937,929,911,907,887,883,881,877,853,839,829,827,823,821,811,809,797,787,773,769,761,757,751,743,739,733,727,719]
        p = [327,277,673,773,419,421,431,433,439,443,449,479,487,491,499,503,509,521,523,541,547,557,587,593,599,601]
        m = 11270
        functions[i] = [a[i], b[i], m]
    return functions
def cosine_similarity(l1,l2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(l1)):
        x = l1[i]
        y = l2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/(math.sqrt(sumxx)*math.sqrt(sumyy))
def get_candidate(iterater, users):#iterater(business_id, user_id)
    output = list()
    all_users = dict()
    user_index = 0
    for user in users:
        if user not in all_users:
            all_users[user] = user_index
            user_index += 1
    # print("all_users: ",end="")
    # print(all_users)
    business_is1_index = dict()
    for business_users in iterater:
        #if this user rated save his user_index
        for rated_user in set(business_users[1]):
            if business_users[0] not in business_is1_index:
                business_is1_index[business_users[0]] = set()
            business_is1_index[business_users[0]].add(all_users[rated_user])
    #build hash functions
    hash_functions = build_hash_functions()
    #build signature matrix
    band = dict()
    hash_value = dict()

    for business_id in list(business_is1_index.keys()):
        #reset to inf
        for i in range(len(hash_functions.keys())):
            hash_value[i] = math.inf
        #for each index, calculate the hash_value
        for index in business_is1_index[business_id]:
            for i in range(len(hash_functions.keys())):
                # f(x) = ((ax + b) % 11270
                hash_value[i] = min(hash_value[i], (hash_functions[i][0] * index + hash_functions[i][1]) % hash_functions[i][2])# % hash_functions[i][3])
        #put them into 15 bands 2 rows each
        for band_num in range(0, len(hash_functions.keys()), 2):
            band_key = (hash_value[band_num], hash_value[band_num+1], int(band_num/2))
            if band_key not in band:
                band[band_key] = list()
            band[band_key].append(business_id)
    #find the candidate pair
    candidate = dict()
    for business_ids in band.values():
        if len(business_ids) >= 2:
            combination = list(itertools.combinations(sorted(set(business_ids)), 2))
            # print(len(combination))
            for pair in combination:
                # cos_similarity = cosine_similarity(list(business_is1_index[pair[0]]), list(business_is1_index[pair[1]]))
                intersection_len = len(set(business_is1_index[pair[0]]).intersection(set(business_is1_index[pair[1]])))
                union_len = len(set(business_is1_index[pair[0]]) | set(business_is1_index[pair[1]]))
                similarity = intersection_len/union_len
                if similarity >= 0.5:
                    candidate[pair] = similarity

    #print("candidate: ", end="")
    #print(candidate)
    for i in sorted(candidate.keys()):
        temp_tuple = (i[0], i[1], candidate[i])
        output.append(temp_tuple)
    print(len(candidate.keys()))
    yield output
def output_csv(output_file, pair_similarity):
    with open(output_file, "w", encoding='utf-8') as output:
        output.write("business_id_1, business_id_2, similarity\n")
        for triple in sorted(pair_similarity[0]):
            output.write(str(triple[0]) + "," + str(triple[1]) + "," + str(triple[2]) + "\n")
        output.close()
sc = SparkContext("local[*]")
sf = SparkConf()
sf.set("spark.executor.memory", "4g")
yelp_train = sc.textFile(input_file).repartition(1)
yelp_train = yelp_train.filter(strip_header).map(lambda s: s.strip()).map(lambda s: s.split(","))
user = yelp_train.map(lambda x: x[0]).collect()
pair_rating = yelp_train.map(lambda x: [x[1], x[0]]).groupByKey().mapPartitions(lambda x: get_candidate(x, user)).collect()
#print(business_business)
output_csv(output_file, pair_rating)
end = time.time()
whole_time = end - start
print("Duration: " + str(whole_time))

