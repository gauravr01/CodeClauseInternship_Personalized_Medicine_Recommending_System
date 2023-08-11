import streamlit as st
import random
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved recommendation system data from the pickle file
with open("1.pkl", "rb") as f:
    recommendation_data = pickle.load(f)

# # Unpack the data
tfidf_vectorizer = recommendation_data["tfidf_vectorizer"]
tfidf_matrix = recommendation_data["tfidf_matrix"]
# validate_user_input = recommendation_data["validate_user_input"]
#recommend_drug = recommendation_data["recommend_drug"]
filtered_data = recommendation_data["filtered_data"]

tips_dict = recommendation_data["tips_dict"]


# Streamlit UI
st.title("Personalized Medicine Recommendation System")

# User input for reason and description

# # Create a dropdown list
# selected_option = st.selectbox("Select an option:", ["Acne", "Fever", "Infection"])
# user_reason = selected_option
user_reason = st.text_input("Enter a reason:")
user_description = st.text_area("Enter a description:")

def recommend_drug(user_reason, user_description):
    # Preprocess and transform user input
    user_input_tfidf = tfidf_vectorizer.transform([user_reason + " " + user_description])

    # Calculate cosine similarity between user input and existing reasons + descriptions
    similarity_scores = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()

    # Find the index of the most similar reason + description
    most_similar_index = similarity_scores.argmax()
    
    # Get the recommended drug name

    recommended_drug = filtered_data.loc[most_similar_index, 'Drug_Name1']
    
    return recommended_drug


def validate_user_input(user_reason, user_description):
    # Basic validation for user input
    if not user_reason:
        raise ValueError("Reason cannot be empty.")
    if not user_description:
        raise ValueError("Description cannot be empty.")
    
    # Add more validation criteria if needed
    
    return user_reason, user_description


# Validate user input
if st.button("Recommend"):
    try:
        print("-------------------Working -----------------------")
        validated_reason, validated_description = validate_user_input(user_reason, user_description)
        
        print("-------------------Working Success----------------------")
        recommended_drug = recommend_drug(validated_reason, validated_description)

        if recommended_drug[-1] == ",":
            st.success(f"Recommended Drug: {recommended_drug[:-1]}")
        else:
            st.success(f"Recommended Drug: {recommended_drug}")

        if validated_reason in tips_dict:
            random_tip = random.choice(tips_dict[validated_reason])
            st.write(f"Random Tip for Treating {validated_reason}:")
            st.info("- " + random_tip)
        else:
            random_tip = random.choice(tips_dict["General_health"])
            st.write(f"Random Tip :")

            st.info("- " + random_tip)

    except ValueError as e:
        st.error("Validation Error:", e)





# from sklearn.preprocessing import QuantileTransformer

# rf_regressor,X_train = pickle.load(open('PROFITMODEL.pkl','rb'))
# #,x_train
# #Streamlit is a light weight web app making framework.
# st.title ("Price Prediction")
# with st.form("my_form"):
#     bmi = st.text_input('R&D Spend')
#     bp = st.text_input('Administration')
#     glu =st.text_input('Marketing Spend')
  
#     submitted = st.form_submit_button("Submit")

# #Model Only takes scaled values. So we have to scale those values before giving input into our model.
# if submitted:
#     # scaler = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
#     # values = scaler.fit_transform(rf_regressor,X_train)
#     glu,bp,bmi = int(glu),int(bp),int(bmi)
#     # values = scaler.transform([[glu,bp,bmi]])
#     values = [[bmi,bp,glu]]
#     pred = rf_regressor.predict(values)
#     st.success(f'Price {pred[0]}')

        
