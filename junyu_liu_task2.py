from pyspark import SparkContext
from pyspark import SparkConf
import math
import time
import itertools
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from sys import argv
from pyspark.sql import SparkSession
start = time.time()
train_file = "yelp_train.csv"
test_file = "yelp_test.csv"
CASE = 1
output_file = "junyu_liu_task2.csv"
# train_file = argv[1]
# test_file = argv[2]
# CASE = int(argv[3])
# output_file = argv[4]
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
                intersection_len = len(set(business_is1_index[pair[0]]).intersection(set(business_is1_index[pair[1]])))
                union_len = len(set(business_is1_index[pair[0]]) | set(business_is1_index[pair[1]]))
                similarity = intersection_len/union_len
                if similarity >= 0.3:
                    candidate[pair] = similarity

    #print("candidate: ", end="")
    #print(candidate)
    for i in sorted(candidate.keys()):
        temp_tuple = (i[0], i[1], candidate[i])
        output.append(temp_tuple)
    print(len(candidate.keys()))
    yield output
def user_base_prediction(iterater, user_businesses, business_users, pair_rating, user_average):
    active_user_businesses = user_businesses[iterater[0]]
    active_user_average = user_average[iterater[0]][0]/user_average[iterater[0]][1]
    if iterater[1] in business_users:
        similar_users = business_users[iterater[1]]
        pred_numerator = 0
        pred_denominator = 0
        for user in similar_users:
            similar_user_rate_active_business = pair_rating[(user, iterater[1])]
            similar_user_businesses = user_businesses[user]
            similar_user_average = user_average[user][0]/user_average[user][1]
            co_rated = active_user_businesses.intersection(similar_user_businesses)
            if len(co_rated) >= 1:
                weight_numerator = 0
                weight_denominator1 = 0
                weight_denominator2 = 0
                for business in co_rated:
                    active_user_rating = pair_rating[(iterater[0], business)]
                    similar_user_rating = pair_rating[(user, business)]
                    # calculate the weight between these two users
                    weight_numerator += (active_user_rating - active_user_average) * (similar_user_rating - similar_user_average)
                    weight_denominator1 += (active_user_rating - active_user_average) ** 2
                    weight_denominator2 += (similar_user_rating - similar_user_average) ** 2
                if weight_denominator1 == 0 or weight_denominator2 == 0:
                    weight = 0
                else:
                    weight = weight_numerator / (math.sqrt(weight_denominator1) * (math.sqrt(weight_denominator2)))
            else:
                weight = 0
            #prediction
            pred_numerator += (similar_user_rate_active_business - similar_user_average) * weight
            pred_denominator += math.fabs(weight)
        if pred_denominator == 0:
            prediction = active_user_average
        else:
            prediction = pred_numerator / pred_denominator + active_user_average
    else:
        prediction = 3.5
        # print("prediction: ",end='')
        # print(prediction)
    return prediction
def item_base_prediction(iterater, user_businesses, business_users, pair_rating, business_average):
    try:
        active_business_average = business_average[iterater[1]][0]/business_average[iterater[1]][1]
    except KeyError:
        return 3.5
    similar_businesses = user_businesses[iterater[0]]
    pred_numerator = 0
    pred_denominator = 0
    weight_numerator = 0
    weight_denominator1 = 0
    weight_denominator2 = 0
    for business in similar_businesses:
        active_user_rate_other_business = pair_rating[(iterater[0], business)]
        similar_business_average = business_average[iterater[1]][0]/business_average[iterater[1]][1]
        active_business_rating = pair_rating[(iterater[0], business)]
        try:
            similar_business_rating = pair_rating[(iterater[0], business)]
        except KeyError:
            continue
        # calculate the weight between these two users
        weight_numerator += (active_business_rating - active_business_average) * (similar_business_rating - similar_business_average)
        weight_denominator1 += (active_business_rating - active_business_average) ** 2
        weight_denominator2 += (similar_business_rating - similar_business_average) ** 2
        if weight_denominator1 == 0 or weight_denominator2 == 0:
            weight = 0
        else:
            weight = weight_numerator / (math.sqrt(weight_denominator1) * (math.sqrt(weight_denominator2)))

        #prediction
        pred_numerator += (active_user_rate_other_business - similar_business_average) * weight
        pred_denominator += math.fabs(weight)
    if pred_denominator == 0:
        prediction = active_business_average
    else:
        prediction = pred_numerator / pred_denominator + active_business_average
        # print("prediction: ",end='')
        # print(prediction)
    return prediction
def item_base_prediction_with_Jaccard_based_LSH(iterater, business_pair, business_pair_inverse, pair_rating, business_average, business_pair_rating):
    numerator = 0
    denominator = 0
    similar_businesses = set()
    if iterater[1] in business_pair:
        similar_businesses |= business_pair[iterater[1]]
    elif iterater[1] in business_pair_inverse:
        similar_businesses |= business_pair_inverse[iterater[1]]
    else:
        similar_businesses = []
    if similar_businesses:
        for similar_business in similar_businesses:
            pair = tuple(sorted([iterater[0], similar_business]))
            try:
                similar_business_rating = pair_rating[pair]
            except KeyError:
                return 3.75
            try:
                weight = business_pair_rating[(iterater[1], similar_business)]
            except KeyError:
                weight = business_pair_rating[(similar_business, iterater[1])]
            numerator += similar_business_rating * weight
            denominator += weight
    if denominator == 0:
        prediction = 3.75
    else:
        prediction = numerator / denominator
    return prediction
def output_csv(output_file, pair_prediction):
    with open(output_file, "w", encoding='utf-8') as output:
        output.write("user_id, business_id, prediction\n")
        for triple in pair_prediction:
            output.write(str(triple[0]) + "," + str(triple[1]) + "," + str(triple[2]) + "\n")
        output.close()
sc = SparkContext("local[*]")
sc.setLogLevel(logLevel="OFF")
spark = SparkSession(sc)
sf = SparkConf()
sf.set("spark.executor.memory", "4g")
if CASE == 1:
    yelp_train = sc.textFile(train_file).filter(strip_header).map(lambda s: s.split(","))
    yelp_test = sc.textFile(test_file).filter(strip_header).map(lambda s: s.split(","))
    all_users = dict()
    all_business = dict()
    user_index = 0
    business_index = 0
    yelp_train_collect = yelp_train.collect()
    yelp_test_collect = yelp_test.collect()
    for line in yelp_train_collect:
        if line[0] not in all_users:
            all_users[line[0]] = user_index
            user_index += 1
        if line[1] not in all_business:
            all_business[line[1]] = business_index
            business_index += 1
    for line in yelp_test_collect:
        if line[1] not in all_business:
            all_business[line[1]] = business_index
            business_index += 1
    all_users_inverse = {y: x for x, y in all_users.items()}
    all_business_inverse = {y: x for x, y in all_business.items()}
    #start
    ratings = yelp_train.map(lambda l: Rating(all_users[l[0]], all_business[l[1]], float(l[2])))
    ratings_test = yelp_test.map(lambda l: Rating(all_users[l[0]], all_business[l[1]], float(l[2])))
    # Build the recommendation model using Alternating Least Squares
    rank = 3
    numIterations = 6
    model = ALS.train(ratings, rank, numIterations)
    # Evaluate the model on training data
    testdata = ratings_test.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), min(5.0, abs(r[2]))))
    ratesAndPreds = ratings_test.map(lambda r: ((r[0], r[1]), r[2])).leftOuterJoin(predictions)
    ratesAndPreds_collect = ratesAndPreds.collect()
    sum_error = 0
    count = 0
    for line in ratesAndPreds_collect:
        if line[1][1] is not None:
            sum_error += (line[1][0] - line[1][1])**2
            count += 1
    triple_result = ratesAndPreds.map(lambda r: (all_users_inverse[r[0][0]], all_business_inverse[r[0][1]], r[1][1])).collect()
    RMSE = math.sqrt(sum_error/count)
    print("Root Mean Squared Error = " + str(RMSE))
    output_csv(output_file, triple_result)
    #end
#prepare the dictionarys for case 2,3,4
user_businesses = dict()
business_users = dict()
pair_rating = dict()
user_average = dict()
business_average = dict()
validation = open(test_file, 'r')
train = open(train_file, 'r')
validation.readline()
train.readline()
for line in train:
    line = line.strip().split(",")
    if line[0] not in user_businesses:
        user_businesses[line[0]] = set()
    user_businesses[line[0]].add(line[1])#{user:(business1, business2)}
    if line[1] not in business_users:
        business_users[line[1]] = set()
    business_users[line[1]].add(line[0])#{business:(user1, user2)}
    pair_rating[(line[0], line[1])] = float(line[2].strip("'"))#{(user, business): rating}
for user in user_businesses.keys():
    if user not in user_average:
        user_average[user] = [0, 0]
    for business in user_businesses[user]:
        user_average[user][0] += pair_rating[(user, business)]
        user_average[user][1] += 1#{user: [sum, count]}
for business in business_users.keys():
    if business not in business_average:
        business_average[business] = [0, 0]
    for user in business_users[business]:
        business_average[business][0] += pair_rating[(user, business)]
        business_average[business][1] += 1#{business: [sum, count]}
if CASE == 2:
    with open(output_file, "w", encoding='utf-8') as output:
        output.write("user_id, business_id, prediction\n")
        sum_error = 0
        count = 0
        for line in validation:
            line = line.strip().split(",")
            test_true = float(line[2].strip("'"))
            test_x = (line[0], line[1])
            count += 1
            prediction = user_base_prediction(test_x, user_businesses, business_users, pair_rating, user_average)
            sum_error += (prediction - test_true)**2
            output.write(str(line[0]) + "," + str(line[1]) + "," + str(prediction) + "\n")
    RMSE = math.sqrt(sum_error/count)
    print("Root Mean Squared Error = " + str(RMSE))
    output.close()
if CASE == 3:
    with open(output_file, "w", encoding='utf-8') as output:
        output.write("user_id, business_id, prediction\n")
        sum_error = 0
        count = 0
        for line in validation:
            line = line.strip().split(",")
            test_true = float(line[2].strip("'"))
            test_x = (line[0], line[1])
            count += 1
            prediction = item_base_prediction(test_x, user_businesses, business_users, pair_rating, business_average)
            sum_error += (prediction - test_true)**2
            output.write(str(line[0]) + "," + str(line[1]) + "," + str(prediction) + "\n")
    RMSE = math.sqrt(sum_error/count)
    print("Root Mean Squared Error = " + str(RMSE))
    output.close()
if CASE == 4:
    business_pair_rating = dict()
    business_pair = dict()
    business_pair_inverse = dict()
    yelp_train = sc.textFile(train_file).repartition(1)
    yelp_train = yelp_train.filter(strip_header).map(lambda s: s.strip()).map(lambda s: s.split(","))
    user = yelp_train.map(lambda x: x[0]).collect()
    pair_rating_collect = yelp_train.map(lambda x: [x[1], x[0]]).groupByKey().mapPartitions(lambda x: get_candidate(x, user)).collect()[0]
    for triple in pair_rating_collect:
        if triple[0] not in business_pair:
            business_pair[triple[0]] = set()
        business_pair[triple[0]].add(triple[1])
        if triple[1] not in business_pair_inverse:
            business_pair_inverse[triple[1]] = set()
        business_pair_inverse[triple[1]].add(triple[0])
        business_pair_rating[(triple[0], triple[1])] = triple[2]
    #business_pair_inverse = {y: x for x, y in business_pair.items()}
    with open(output_file, "w", encoding='utf-8') as output:
        output.write("user_id, business_id, prediction\n")
        sum_error = 0
        count = 0
        for line in validation:
            line = line.strip().split(",")
            test_true = float(line[2].strip("'"))
            test_x = (line[0], line[1])
            count += 1
            prediction = item_base_prediction_with_Jaccard_based_LSH(test_x, business_pair, business_pair_inverse, pair_rating, business_average, business_pair_rating)
            #prediction = 3.5
            sum_error += (prediction - test_true)**2
            output.write(str(line[0]) + "," + str(line[1]) + "," + str(prediction) + "\n")
    RMSE = math.sqrt(sum_error/count)
    print("Root Mean Squared Error = " + str(RMSE))
    output.close()
    with open("junyu_liu_explanation", "w", encoding='utf-8') as output:
        output.write("Item base recommendation system with Jaccard similarity based LSH will shorter the running time"
                     "because after running Jaccard similarity base LSH, we have a prebuild dictionary to predict the "
                     "rating for an active user.")
    output.close()
end = time.time()
whole_time = end - start
print("Duration: " + str(whole_time))


