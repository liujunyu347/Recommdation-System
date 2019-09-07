from surprise import SVD
from surprise import Reader, Dataset
import sys
import io
import math
import time
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

start = time.time()
train_file = "yelp_train.csv"
test_file = "yelp_val.csv"
business_file = "business.json"
# user_file = "user.json"
output_file = "competition.txt"
description_file = 'junyu_liu_description.txt'
temp_file = "temp_file.csv"
# train_file = argv[1]
# test_file = argv[2]
# output_file = argv[3]
temp = open(temp_file, 'r', encoding='UTF-8')
business = open(business_file, 'r', encoding='UTF-8')
# user = open(user_file, 'r', encoding='UTF-8')
business_id_stars = dict()
business_id_review_count = dict()
user_id_stars = dict()
user_id_review_count = dict()
for line in business:
    business_id = json.loads(line)['business_id']
    business_stars = json.loads(line)['stars']
    # business_review_count = json.loads(line)['review_count']
    business_id_stars[business_id] = business_stars
    # business_id_review_count[business_id] = business_review_count
# for line in user:
#     user_id = json.loads(line)['user_id']
#     user_stars = json.loads(line)['average_stars']
#     user_review_count = json.loads(line)['review_count']
#     user_id_stars[user_id] = user_stars
#     user_id_review_count[user_id] = user_review_count


# #prepare the dictionarys for case 2,3,4
user_businesses = dict()
business_users = dict()
pair_rating = dict()
user_average = dict()
business_average = dict()
vali_list = []
train = open(train_file, 'r', encoding='UTF-8')
validation = open(test_file, 'r', encoding='UTF-8')

train.readline()
validation.readline()
train_str = ''
for line in validation:
    line = line.split(",")
    vali_list.append(line)
for line in train:
    train_str += line
    line = line.strip().split(",")
    # if line[0] not in user_businesses:
    #     user_businesses[line[0]] = set()
    # user_businesses[line[0]].add(line[1])#{user:(business1, business2)}
    if line[1] not in business_users:
        business_users[line[1]] = set()
    business_users[line[1]].add(line[0])#{business:(user1, user2)}
    pair_rating[(line[0], line[1])] = float(line[2].strip("'"))#{(user, business): rating}
# for user in user_businesses.keys():
#     if user not in user_average:
#         user_average[user] = [0, 0]
#     for business in user_businesses[user]:
#         user_average[user][0] += pair_rating[(user, business)]
#         user_average[user][1] += 1#{user: [sum, count]}
for business in business_users.keys():
    if business not in business_average:
        business_average[business] = [0, 0]
    for user in business_users[business]:
        business_average[business][0] += pair_rating[(user, business)]
        business_average[business][1] += 1#{business: [sum, count]}

reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
# fold_path = [(train_file, test_file)]
# data = Dataset.load_from_folds(fold_path, reader=reader)
data = Dataset.load_from_file(train_file, reader=reader)
trainset = data.build_full_trainset()
# min_rmse = 2
# min_rs = 51
# for rs in range(50,100,1):
algo = SVD(n_factors=20, lr_bu=0.008, lr_bi=0.008, lr_pu=0.009, lr_qi=0.01, reg_all=0.2, n_epochs=23, random_state=21).fit(trainset)

sum_error = 0
count = 0
round_one = ''
for line in vali_list:
    test_true = float(line[2])
    business = line[1]
    if business not in business_average.keys():
        pred = business_id_stars[business]
    elif business_average[business][1] <= 3:
        pred = business_id_stars[business]
    else:
        pred = algo.predict(line[0], line[1], verbose=False).est

    round_one += str(line[0]) + "," + str(line[1]) + "," + str(pred) + "\n"

data = Dataset.load_from_file(train_file, reader=reader)


error_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
with open(output_file, "w", encoding='utf-8') as output:
    output.write("user_id, business_id, prediction\n")
    for line in vali_list:
        # line = line.split(",")
        test_true = float(line[2])
        business = line[1]
        if business not in business_average.keys():
            pred = business_id_stars[business]
        elif business_average[business][1] <= 3:
            pred = business_id_stars[business]
        else:
            pred = algo.predict(line[0], line[1], verbose=False).est

        absolute_differences = abs(pred - test_true)
        count += 1
        sum_error += (pred - test_true) ** 2
        error_distribution[int(absolute_differences)] += 1
        output.write(str(line[0]) + "," + str(line[1]) + "," + str(pred) + "\n")
output.close()

print("Error Distribution:")
print(error_distribution)
RMSE = math.sqrt(sum_error / count)

print("Root Mean Squared Error = " + str(RMSE))
#     if RMSE < min_rmse:
#         min_rmse = RMSE
#         min_rs = rs
# print(min_rmse)
# print(min_rs)
end = time.time()
whole_time = end - start

writer = ""
writer += "Method Description:" + "\n"
writer += ""
writer += "\n" + "Error Distribution:" + "\n"
writer += ">=0 and <1: " + str(error_distribution[0]) + "\n"
writer += ">=1 and <2: " + str(error_distribution[1]) + "\n"
writer += ">=2 and <3: " + str(error_distribution[2]) + "\n"
writer += ">=3 and <4: " + str(error_distribution[3]) + "\n"
writer += ">=4 and <5: " + str(error_distribution[4]) + "\n"
writer += "\n" + "RMSE:" + "\n"
writer += str(RMSE) + "\n"
writer += "\n" + "Execution Time:" + "\n"
writer += str(whole_time) + "s"

with open(description_file, "w", encoding='utf-8') as output:
    output.write(writer)
output.close()

end = time.time()
whole_time = end - start
print("Duration: " + str(whole_time))

# print(math.sqrt(mse / count))
# for line in validation:
#     line = line.strip().split(",")
#     test_true = float(line[2].strip("'"))
#     test_x = (line[0], line[1])
#     count += 1
#     prediction = item_base_prediction(test_x, user_businesses, pair_rating, business_average, business_id_stars, business_id_review_count, user_id_stars, user_id_review_count)
#     sum_error += (prediction - test_true) ** 2
#     absolute_differences = abs(prediction - test_true)
#     if absolute_differences > 4:
#         print(test_x[0])
#         absolute_differences = 4
#     error_distribution[int(absolute_differences)] += 1