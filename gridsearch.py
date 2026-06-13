import sys
import os
import json
import math
import itertools
import time
import catboost as cb
from pyspark import SparkContext

"""
Note on AI Usage: Google Gemini was used as an interactive programming assistant 
during the development of this script.
"""

def extract_user_features(line, default_stars):
    data = json.loads(line)
    return (data['user_id'], [
        float(data.get('average_stars', default_stars)),
        float(data.get('review_count', 0)),
        float(data.get('useful', 0)),
        float(data.get('funny', 0)),
        float(data.get('cool', 0)),
        float(data.get('fans', 0))
    ])

def extract_business_features(line, default_stars):
    data = json.loads(line)
    return (data['business_id'], [
        float(data.get('stars', default_stars)),
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
    if len(sys.argv) != 4:
        print("Usage: spark-submit recommendation.py <folder_path> <val_file> <output_file>")
        sys.exit(-1)

    folder_path = sys.argv[1]
    val_file = sys.argv[2]     
    output_file = sys.argv[3]

    train_file = os.path.join(folder_path, "yelp_train.csv")
    user_file = os.path.join(folder_path, "user.json")
    bus_file = os.path.join(folder_path, "business.json")

    sc = SparkContext(appName="competition_CatBoost_GridSearch", master="local[*]")
    sc.setLogLevel("WARN")

    # 1. LOAD TRAIN DATA FIRST
    raw_train = sc.textFile(train_file)
    train_header = raw_train.first()
    train_rdd = raw_train.filter(lambda x: x != train_header).map(lambda x: x.split(","))

    train_cf_rdd = train_rdd.map(lambda x: (x[0], x[1], float(x[2])))
    train_cf_rdd.cache() 

    bus_user_dict = train_cf_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    user_bus_dict = train_cf_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()

    # 2. COMPUTE TRUE GLOBAL AVERAGE
    global_avg = sum(sum(r.values()) for r in bus_user_dict.values()) / sum(len(r) for r in bus_user_dict.values())

    # 3. LOAD JSON DATA USING GLOBAL AVERAGE AS FALLBACK
    user_dict = dict(sc.textFile(user_file).map(lambda line: extract_user_features(line, global_avg)).collect())
    bus_dict = dict(sc.textFile(bus_file).map(lambda line: extract_business_features(line, global_avg)).collect())
    
    # 4. UPDATE FALLBACK FEATURES FOR UNSEEN DATA
    def_u_feat = [global_avg, 0.0, 0.0, 0.0, 0.0, 0.0] 
    def_b_feat = [global_avg, 0.0]

    damping = 3.0
    bus_avg_dict = {b: (sum(r.values()) + (global_avg * damping)) / (len(r) + damping) for b, r in bus_user_dict.items()}
    user_avg_dict = {u: (sum(r.values()) + (global_avg * damping)) / (len(r) + damping) for u, r in user_bus_dict.items()}

    bus_user_bc = sc.broadcast(bus_user_dict)
    user_bus_bc = sc.broadcast(user_bus_dict)
    bus_avg_bc = sc.broadcast(bus_avg_dict)
    user_avg_bc = sc.broadcast(user_avg_dict)

    # 5. PREPARE TRAINING DATA
    train_data = train_rdd.collect()
    X_train, y_train = [], []
    for row in train_data:
        u_id, b_id, rating = row[0], row[1], float(row[2])
        u_feat = user_dict.get(u_id, def_u_feat)
        b_feat = bus_dict.get(b_id, def_b_feat)
        X_train.append([u_id, b_id] + u_feat + b_feat)
        y_train.append(rating)
        
    cat_features_indices = [0, 1]

    # 6. PREPARE VALIDATION DATA (Execute CF Once)
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
        X_val.append([u_id, b_id] + u_feat + b_feat)

    # 7. LOAD GROUND TRUTH FOR IN-MEMORY SCORING
    ground_truth = {}
    with open(val_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                ground_truth[(parts[0], parts[1])] = float(parts[2])

    # 8. AGGRESSIVE HYPERPARAMETER GRID
    param_grid = {
        'iterations': [1500, 2000],
        'learning_rate': [0.08, 0.1],
        'depth': [8, 9, 10],
        'l2_leaf_reg': [20, 35, 50]
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_rmse = float('inf')
    best_params = None
    best_predictions = []

    print(f"\nStarting Aggressive Grid Search with {len(combinations)} combinations...")
    print("-" * 50)

    # 9. EXECUTE GRID SEARCH WITH TIMER
    for i, params in enumerate(combinations):
        print(f"Training {i+1}/{len(combinations)}: {params}")
        
        # Start the clock
        start_time = time.time()
        
        model = cb.CatBoostRegressor(
            **params,
            cat_features=cat_features_indices,
            loss_function='RMSE',
            verbose=0,
            random_seed=42
        )
        model.fit(X_train, y_train)

        cat_predictions = model.predict(X_val)
        
        sum_sq_err = 0.0
        n_eval = 0
        current_preds = []

        # Blend and Score
        for j in range(len(val_list)):
            u_id, b_id = val_list[j][0], val_list[j][1]
            
            cat_pred = float(cat_predictions[j])
            cf_pred, cf_weight = cf_results_dict[(u_id, b_id)]
            
            # Hybrid Weighting
            alpha = min(0.20, cf_weight / (cf_weight + 10.0))
            final_pred = (alpha * cf_pred) + ((1.0 - alpha) * cat_pred)
            
            # Bound Predictions
            if final_pred > 5.0: final_pred = 5.0
            elif final_pred < 1.0: final_pred = 1.0
                
            current_preds.append((u_id, b_id, final_pred))
            
            # Calculate RMSE
            if (u_id, b_id) in ground_truth:
                diff = ground_truth[(u_id, b_id)] - final_pred
                sum_sq_err += (diff ** 2)
                n_eval += 1
                
        # Stop the clock and format the output
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        mins, secs = divmod(int(elapsed_seconds), 60)
        
        current_rmse = math.sqrt(sum_sq_err / n_eval) if n_eval > 0 else float('inf')
        
        print(f"--> RMSE: {current_rmse:.4f}  |  Time: {mins}m {secs}s\n")
        
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params
            best_predictions = current_preds

    # 10. FINAL OUTPUT
    print("=" * 50)
    print(f"BEST RMSE: {best_rmse:.4f}")
    print(f"BEST PARAMS: {best_params}")
    print("=" * 50)

    with open(output_file, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for u_id, b_id, pred in best_predictions:
            f.write(f"{u_id},{b_id},{pred}\n")

    sc.stop()