import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load and preprocess the user-item rating dataset
data = pd.read_csv('ratings.csv')  # Replace 'ratings.csv' with your dataset file
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Train the model using matrix factorization
model = SVD()
model.fit(trainset)

# Generate recommendations for a given user
user_id = 1  # Replace with the desired user ID
user_items = data[data['user_id'] == user_id]['item_id']
items_to_predict = [item for item in dataset.df['item_id'].unique() if item not in user_items]
testset = [[user_id, item_id, 0] for item_id in items_to_predict]
predictions = model.test(testset)

# Output the recommendations
recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]  # Get top 10 recommendations
for recommendation in recommendations:
    print(f"Item ID: {recommendation.iid}, Estimated Rating: {recommendation.est}")

# Evaluate the model's performance
test_predictions = model.test(testset)
accuracy.rmse(test_predictions)
accuracy.mae(test_predictions)
