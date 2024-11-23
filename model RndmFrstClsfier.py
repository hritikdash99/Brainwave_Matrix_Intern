import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
import string

# Step 1: Load the Data
liar_data = pd.read_csv(
    "liar.csv",
    header=None,
    names=[
        "ID",
        "Label",
        "Statement",
        "Subjects",
        "Speaker",
        "Position",
        "State",
        "Party",
        "Barely_True_Count",
        "False_Count",
        "Half_True_Count",
        "Mostly_True_Count",
        "Pants_on_Fire_Count",
        "Context",
    ],
)

true_data = pd.read_csv("True.csv")
fake_data = pd.read_csv("Fake.csv")

# Step 2: Preprocess Data
# Combine true and fake datasets with the liar dataset
true_data["Label"] = "TRUE"
fake_data["Label"] = "FALSE"

true_data = true_data.rename(columns={"text": "Statement"})
fake_data = fake_data.rename(columns={"text": "Statement"})

# Combine all datasets
liar_data_subset = liar_data[["Statement", "Label"]]
combined_data = pd.concat(
    [
        liar_data_subset,
        true_data[["Statement", "Label"]],
        fake_data[["Statement", "Label"]]
    ]
)

# Drop rows with missing labels or statements
combined_data = combined_data.dropna(subset=["Statement", "Label"])

# Clean text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\[.*?\]", "", text)  # Remove text in brackets
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation), "", text
    )  # Remove punctuation
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words containing numbers
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    return text


combined_data["Statement"] = combined_data["Statement"].apply(preprocess_text)

# Encode labels
combined_data["Label"] = combined_data["Label"].map({"TRUE": 1, "FALSE": 0})

# Drop rows with invalid labels (after mapping)
combined_data = combined_data.dropna(subset=["Label"])

# Step 3: Feature Extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(combined_data["Statement"]).toarray()
y = combined_data["Label"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Model Training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Step 7: Prediction Function
def predict_news(news):
    news_processed = preprocess_text(news)
    news_vectorized = tfidf.transform([news_processed]).toarray()
    prediction = model.predict(news_vectorized)
    return "True News" if prediction[0] == 1 else "Fake News"


# Example Usage
user_input = input("Enter a news statement: ")
print("Prediction:", predict_news(user_input))
