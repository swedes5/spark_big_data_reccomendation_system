#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import json
import re
from pyspark import SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit mine_tips.py <folder_path>")
        sys.exit(-1)

    folder_path = sys.argv[1]
    train_file = os.path.join(folder_path, "yelp_train.csv")
    tip_file = os.path.join(folder_path, "tip.json")

    sc = SparkContext(appName="TipKeywordMiner", master="local[*]")
    sc.setLogLevel("WARN")

    # 1. Map (user, business) -> rating from training data
    raw_train = sc.textFile(train_file)
    header = raw_train.first()
    
    train_ratings_dict = raw_train.filter(lambda x: x != header)        .map(lambda x: x.split(","))        .map(lambda x: ((x[0], x[1]), float(x[2])))        .collectAsMap()

    ratings_bc = sc.broadcast(train_ratings_dict)

    # 2. Extract words from tips and link to known ratings
    def process_tip(line):
        try:
            data = json.loads(line)
            u_id = data['user_id']
            b_id = data['business_id']
            text = data.get('text', '')
        except:
            return []
            
        rating = ratings_bc.value.get((u_id, b_id))
        if not rating or not text:
            return []
            
        # Grab words 4+ letters long (filters out 'and', 'the', etc.)
        words = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
        
        results = []
        if rating >= 4.0: # Positive tip
            for w in words: results.append((w, (1, 0)))
        elif rating <= 2.0: # Negative tip
            for w in words: results.append((w, (0, 1)))
            
        return results

    # (word) -> (Pos_Count, Neg_Count)
    word_counts = sc.textFile(tip_file)        .flatMap(process_tip)        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))        .filter(lambda x: (x[1][0] + x[1][1]) >= 10) # Must appear at least 10 times total
        
    # Calculate positive ratio: Pos / (Pos + Neg)
    word_stats = word_counts.map(lambda x: (x[0], x[1][0], x[1][1], x[1][0] / (x[1][0] + x[1][1]))).collect()

    # Sort for pure positive (ratio near 1.0) and pure negative (ratio near 0.0)
    # Adding a minimum count threshold to ensure the word isn't just a 1-off typo
    top_pos = sorted([x for x in word_stats if x[1] >= 15], key=lambda x: x[3], reverse=True)
    top_neg = sorted([x for x in word_stats if x[2] >= 15], key=lambda x: x[3])

    print("\n--- COPY/PASTE INTO POS_WORDS SET ---")
    for w in top_pos[:50]:
        print(f"'{w[0]}',  # {w[1]} pos, {w[2]} neg")

    print("\n--- COPY/PASTE INTO NEG_WORDS SET ---")
    for w in top_neg[:50]:
        print(f"'{w[0]}',  # {w[1]} pos, {w[2]} neg")

    sc.stop()

