import sys
import os
import json
import time
import catboost as cb
from pyspark import SparkContext

def get_tip_sentiment(text):
    if not text: return 0.0
    text = text.lower()
    
    ext_pos_words = {'perfection', 'phenomenal', 'incredible', 'superb', 'paradise', 'gem'}
    pos_words = {'great', 'good', 'awesome', 'amazing', 'love', 'best', 'delicious', 'excellent', 'perfect', 'nice', 'friendly', 'recommend', 'delightful', 'welcoming', 'addicting', 'addicted', 'excited', 'flavours', 'buttery', 'craving', 'authentic', 'tender', 'crispy', 'fresh', 'favorite', 'homemade', 'savory', 'melt', 'locally', 'fantastic'}
    
    ext_neg_words = {'worst', 'scam', 'unacceptable', 'refused', 'zero', 'dump', 'disgusting', 'horrible', 'terrible'}
    neg_words = {'bad', 'awful', 'hate', 'poor', 'slow', 'rude', 'never', 'avoid', 'mediocre', 'waste', 'nasty', 'ghetto', 'gross', 'sucks', 'disappointing', 'overpriced', 'joke', 'bland', 'attitude', 'fail', 'broken', 'greasy', 'management', 'manager', 'charged', 'beware', 'waited', 'dirty', 'minutes', 'forever', 'walked', 'unprofessional', 'ignored', 'worse', 'elsewhere', 'cold', 'apology', 'leave', 'bother'}
    
    words = ''.join(e for e in text if e.isalnum() or e.isspace()).split()
    
    pos_count = sum(2 if w in ext_pos_words else 1 for w in words if w in pos_words or w in ext_pos_words)
    neg_count = sum(2 if w in ext_neg_words else 1 for w in words if w in neg_words or w in ext_neg_words)
    
    if pos_count > neg_count: 
        return 2.0 if any(w in ext_pos_words for w in words) else 1.0
    elif neg_count > pos_count: 
        return -2.0 if any(w in ext_neg_words for w in words) else -1.0
    else: 
        return 0.0

def extract_user_features(line):
    data = json.loads(line)
    return (data['user_id'], [
        float(data.get('average_stars', 3.0)),
        float(data.get('review_count', 0)),
        float(data.get('useful', 0)),
        float(data.get('funny', 0)),
        float(data.get('cool', 0)),
        float(data.get('fans', 0))
    ])

def extract_business_features(line):
    data = json.loads(line)
    return (data['business_id'], [
        float(data.get('stars', 3.0)),
        float(data.get('review_count', 0))
    ])

def compute_pearson(bus1_ratings, bus2_ratings):
    corated_users = set(bus1_ratings.keys()).intersection(set(bus2_ratings.keys()))
    if len(corated_users) < 3:
        return 0.0
        
    r1 = [bus1_ratings[u] for u in corated_users]
    r2 = [bus2_ratings[u] for u in corated_users]
    
    avg1 = sum(r1) / len(r1)
    avg2 = sum(r2) / len(r2)
    
    numerator = sum((x - avg1) * (y - avg2) for x, y in zip(r1, r2))
    denominator1 = sum((x - avg1)**2 for x in r1)
    denominator2 = sum((y - avg2)**2 for y in r2)
    
    if denominator1 == 0 or denominator2 == 0:
        return 0.0
        
    weight = numerator / ((denominator1 ** 0.5) * (denominator2 ** 0.5))
    
    support = len(corated_users)
    shrinkage_factor = min(support / 30.0, 1.0)
    
    return weight * shrinkage_factor

def predict_rating_cf(user_id, business_id, bus_user_dict, user_bus_dict, bus_avgs, user_avgs, global_avg):
    if user_id not in user_bus_dict and business_id not in bus_user_dict:
        return (global_avg, 0.0)
    if user_id not in user_bus_dict:
        return (bus_avgs.get(business_id, global_avg), 0.0)
    if business_id not in bus_user_dict:
        return (user_avgs.get(user_id, global_avg), 0.0)

    baseline = user_avgs[user_id] + bus_avgs[business_id] - global_avg
    user_history = user_bus_dict[user_id] 
    target_bus_profile = bus_user_dict[business_id] 
    
    valid_neighbors = []
    
    for other_bus, rating in user_history.items():
        if other_bus == business_id: continue
            
        other_bus_profile = bus_user_dict[other_bus]
        weight = compute_pearson(target_bus_profile, other_bus_profile)
        
        if weight > 0.0:
            valid_neighbors.append((weight, rating, other_bus))
            
    valid_neighbors.sort(key=lambda x: x[0], reverse=True)
    
    top_n_neighbors = valid_neighbors[:60]
    
    numerator = 0.0
    denominator = 0.0
    
    for w, r, other_bus in top_n_neighbors:
        neighbor_baseline = user_avgs[user_id] + bus_avgs[other_bus] - global_avg
        deviation = r - neighbor_baseline
        numerator += deviation * w
        denominator += w
            
    if denominator < 1.0:
        return (baseline, denominator)
    else:
        return (baseline + (numerator / denominator), denominator)

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) != 4:
        sys.exit(-1)

    folder_path = sys.argv[1]
    val_file = sys.argv[2]     
    output_file = sys.argv[3]

    train_file = os.path.join(folder_path, "yelp_train.csv")
    user_file = os.path.join(folder_path, "user.json")
    bus_file = os.path.join(folder_path, "business.json")
    tip_file = os.path.join(folder_path, "tip.json")

    sc = SparkContext(appName="competition", master="local[*]")
    sc.setLogLevel("WARN")

    user_dict = dict(sc.textFile(user_file).map(extract_user_features).collect())
    bus_dict = dict(sc.textFile(bus_file).map(extract_business_features).collect())
    
    tip_dict = dict(sc.textFile(tip_file)
                    .map(json.loads)
                    .map(lambda x: ((x['user_id'], x['business_id']), get_tip_sentiment(x.get('text', ''))))
                    .reduceByKey(lambda a, b: a + b) 
                    .collect())

    def_u_feat = [3.5, 10.0, 0.0, 0.0, 0.0, 0.0] 
    def_b_feat = [3.5, 10.0]

    raw_train = sc.textFile(train_file)
    train_header = raw_train.first()
    train_rdd = raw_train.filter(lambda x: x != train_header).map(lambda x: x.split(","))

    train_cf_rdd = train_rdd.map(lambda x: (x[0], x[1], float(x[2])))
    bus_user_dict = train_cf_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    user_bus_dict = train_cf_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()

    global_avg = sum(sum(r.values()) for r in bus_user_dict.values()) / sum(len(r) for r in bus_user_dict.values())

    damping = 3.0
    bus_avg_dict = {b: (sum(r.values()) + (global_avg * damping)) / (len(r) + damping) for b, r in bus_user_dict.items()}
    user_avg_dict = {u: (sum(r.values()) + (global_avg * damping)) / (len(r) + damping) for u, r in user_bus_dict.items()}

    bus_user_bc = sc.broadcast(bus_user_dict)
    user_bus_bc = sc.broadcast(user_bus_dict)
    bus_avg_bc = sc.broadcast(bus_avg_dict)
    user_avg_bc = sc.broadcast(user_avg_dict)

    train_data = train_rdd.collect()
    X_train, y_train = [], []
    for row in train_data:
        u_id, b_id, rating = row[0], row[1], float(row[2])
        
        u_feat = user_dict.get(u_id, def_u_feat)
        b_feat = bus_dict.get(b_id, def_b_feat)
        tip_score = tip_dict.get((u_id, b_id), 0.0)
        
        X_train.append([u_id, b_id] + u_feat + b_feat + [tip_score])
        y_train.append(rating)
        
    cat_features_indices = [0, 1]

    model = cb.CatBoostRegressor(
        iterations=400,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3,
        cat_features=cat_features_indices,
        loss_function='RMSE',
        verbose=0, 
        random_seed=42
    )
    model.fit(X_train, y_train)

    raw_val = sc.textFile(val_file)
    val_header = raw_val.first()
    val_rdd = raw_val.filter(lambda x: x != val_header).map(lambda x: x.split(","))
    
    cf_results_dict = val_rdd.map(
        lambda x: (
            (x[0], x[1]), 
            predict_rating_cf(x[0], x[1], bus_user_bc.value, user_bus_bc.value, bus_avg_bc.value, user_avg_bc.value, global_avg)
        )
    ).collectAsMap()

    val_list = val_rdd.collect()
    X_val = []
    
    for row in val_list:
        u_id, b_id = row[0], row[1]
        u_feat = user_dict.get(u_id, def_u_feat)
        b_feat = bus_dict.get(b_id, def_b_feat)
        
        tip_score = tip_dict.get((u_id, b_id), 0.0)
        
        X_val.append([u_id, b_id] + u_feat + b_feat + [tip_score])

    cat_predictions = model.predict(X_val)

    with open(output_file, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        
        for i in range(len(val_list)):
            u_id, b_id = val_list[i][0], val_list[i][1]
            
            cat_pred = float(cat_predictions[i])
            cf_pred, cf_weight = cf_results_dict[(u_id, b_id)]
            
            alpha = min(0.20, cf_weight / (cf_weight + 10.0))
            final_pred = (alpha * cf_pred) + ((1.0 - alpha) * cat_pred)
            
            if final_pred > 5.0: final_pred = 5.0
            elif final_pred < 1.0: final_pred = 1.0
                
            f.write(f"{u_id},{b_id},{final_pred}\n")

    sc.stop()

    execution_time = time.time() - start_time
    print(f"Total Execution Time: {execution_time:.2f} seconds")