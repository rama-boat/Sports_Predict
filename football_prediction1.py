
import pickle
import streamlit as st
import numpy as np


# Opens the deployed model
model = pickle.load(open('C:/Users/Ewurama Boateng/Desktop/Group9_SportsPrediction/deployed_model.pkl', 'rb'))
with open('scale_model.pkl', 'wb') as scaler_file:
    scaler,scaling = pickle.load(model,scaler_file)
    model.tree_.missing_go_to_left = model.tree_.missing_go_to_left.astype(np.uint8)





def predict(user_inputs):
    st.title('Player Rating Sports Prediction')

    user_inputs = np.array(user_inputs)

    scaled_inputs = scaling.transform([user_inputs])

    makeprediction = model.predict(scaled_inputs)

    return makeprediction

    # Features
def main():
    cf = st.number_input('cf')
    gk = st.number_input('gk')
    lm = st.number_input('lm')
    lcb = st.number_input('lcb')
    rm = st.number_input('rm')
    potential = st.number_input('potential')
    value_eur = st.number_input('value_eur')
    wage_eur = st.number_input('wage_eur')
    release_clause_eur = st.number_input('release_clause_eur')
    movement_reactions = int(st.number_input('movement_reactions'))
    age = int(st.number_input('age'))

    
    if st.button('Predict', key='predict_button'):
        
        user_inputs = [gk, lcb, cf, lm, rm, potential, value_eur, wage_eur, release_clause_eur, movement_reactions, age]


        output = round(predict(user_inputs), 2)
        st.success("The player's overall rating is {output}".format(output))


        #this is the  code that we will use for prediction
    # if st.button('Predict', key='predict_button'):
    #     makeprediction = model.predict([[gk,lcb,cf,lm,rm,potential,value_eur,wage_eur,release_clause_eur,movement_reactions,age]])
    #     output=round(makeprediction[0],2)
    #     st.success('The player overall performance is {}'.format(output))


if __name__ == '__main__':
    main()


#go to terminal and type (streamlit run football_prediction.py)
