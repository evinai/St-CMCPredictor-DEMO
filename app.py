# Core Pkgs
import streamlit as st
import os

# EDA
import pandas as pd
import numpy as np

# Visualization
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')#
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)


st.sidebar.image('togaylogogri.jpg', width=250)
st.sidebar.caption('Demo-Uygulama')


import joblib
import pickle

######### FUNCTIONS #########

# Get Keys
# Will look thru the keys and get the value

def get_value(val,my_dicts):
	for key, value in my_dicts.items():
		if val == key:
			return value


def get_key(val,my_dicts):
	for key, value in my_dicts.items():
		if val == value:
			return key

# Load Models
def load_prediction_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def main():
	st.title('CMC Predictor App')

	

######### PAGE SELECTION #########
	activity = ["Descriptive Analysis", "Predictive Analysis"]
	choice = st.sidebar.selectbox("Choose Analytics Type",activity)

	if choice == ("Descriptive Analysis"):
		st.subheader("Exploratory Data Analysis")
		# Preview Dataset

		df = pd.read_csv("data_cmc/cmc_dataset.csv")

		if st.checkbox("Preview Dataset"):
			 number = st.number_input('Select Number of Rows To View', value=0)
			 st.dataframe(df.head(number))


		if st.checkbox("Shape of Data Set"):
			st.write(df.shape)
			data_dim = st.radio('Show Dimensions By ', ('Rows', 'Cloumns'))
			if data_dim == 'Rows':
				st.text("Showing the Rows")
				st.write(df.shape[0])
			if data_dim == 'Columns':
				st.text("Showing the Columns")
				st.write(df.shape[1])

		if st.checkbox("Select Columns"):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect("Select Colums",all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.button('Summary of Dataset'):
			st.write(df.describe())

		if st.button("Values Counts"):
			st.text("Value Counts By Target")
			st.write(df.iloc[:,-1].value_counts())

		st.subheader('Data Visualization')
		# Correlation Plot
		if st.checkbox("Correlation Plot with Matplotlib"):
			plt.matshow(df.corr())
			st.pyplot()

		if st.checkbox("Correlation Plot with Seaborn"):
			st.write(sns.heatmap(df.corr(),annot=True))
			st.pyplot()

		if st.checkbox("Pie Chart"):
			if st.button("Generate Pie Chart"):
				st.write(df.iloc[:,-1].value_counts().plot(kind='pie', autopct='%1.1f%%'))
				st.pyplot()

		if st.checkbox("Plot Value Counts"):
			st.write(df.iloc[:,-1].value_counts().plot(kind='pie', autopct='%1.1f%%'))
			st.pyplot()

		if st.checkbox('Plot Value Counts by Columns'):
			st.text("Value Counts By Target/Class")

			all_columns_names = df.columns.tolist()
			primary_col = st.selectbox('Select Primary Column To Group By', all_columns_names)
			selected_column_names = st.multiselect('Select Columns',all_columns_names)
			if st.button("Plot"):
				st.text("Generating Plot for: {} and {}".format(primary_col,selected_column_names))
				if selected_column_names:
					vc_plot = df.groupby(primary_col)[selected_column_names].count()
				else:
					vc_plot = df.iloc[:,-1].value_counts()
				st.write(vc_plot.plot(kind='bar'))
				st.pyplot()




	if choice ==("Predictive Analysis"):
		st.subheader("Machine Learning Predict Model")
		df = pd.read_csv("data_cmc/cmc_dataset.csv")
		st.write(df.head(5))
		st.write(df.columns)

		age = st.slider("Select Age",16,60)
		wife_education = st.number_input("Select Wife Education Level(low2high[1,4])",1,4)
		husband_education = st.number_input("Select Husbands Education Level(low2high[1,4])",1,4)
		num_of_born_children = st.number_input("Number of Born Children",1,20)

		wife_reg = {"Non_Religious":0,"Religious":1}
		choice_wife_reg = st.radio("wife's religion",tuple(wife_reg.keys()))
		result_wife_reg = get_value(choice_wife_reg,wife_reg)

		wife_working = {"Yes":0,"No":1}
		choice_wife_working = st.radio("is the wife working",tuple(wife_working.keys()))
		result_wife_working = get_value(choice_wife_working,wife_working)


		husband_occupation = st.number_input("Husbands Occupation Level[1,4])",1,4)
		standard_of_living = st.slider("Standard of Living",1,4)
		

		media_exposure = {"Good":1, "Not-Good":0}
		choice_media_exposure = st.radio('media exposure',tuple(media_exposure.keys()))
		result_media_exposure = get_value(choice_media_exposure,media_exposure)


		# we will take all the values above and put it inside a list
		# Result and in json format
		results = [age,wife_education,husband_education,num_of_born_children,result_wife_reg,result_wife_working,husband_occupation,standard_of_living,result_media_exposure]
		displayed_results = [age,wife_education,husband_education,num_of_born_children,choice_wife_reg,choice_wife_working,husband_occupation,standard_of_living,choice_media_exposure]
		prettified_result = {"age":age,
		"wife_education":wife_education,
		"husband_education":husband_education,
		"num_of_born_children":num_of_born_children,
		"result_wife_reg":choice_wife_reg,
		"result_wife_working":choice_wife_working,
		"husband_occupation":husband_occupation,
		"standard_of_living":standard_of_living,
		"media_exposure":choice_media_exposure}
		sample_data = np.array(results).reshape(1, -1)

		
		st.json(prettified_result)
		st.text("Vectorized as ::{}".format(results))

		st.subheader("Prediction Aspects")
		if st.checkbox("Make Prediction"):
			all_ml_dict = ('LR',"Decision Tree","Naive Bayes","Random Forest")
			model_choice = st.selectbox("Model Choice",all_ml_dict)

			if st.button("Predict"):
				prediciton_label = {"Not_using":1, "Long_term_use":2, "short_term_use":3}
				if model_choice == 'LR':
					predictor = load_prediction_model("models/contraceptives_logit_model.pkl")
					prediction = predictor.predict(sample_data)
					st.write(prediction)

				elif model_choice == 'Decision Tree':
					predictor = load_prediction_model("models/contraceptives_dcTree_model.pkl")
					prediction = predictor.predict(sample_data)
					st.write(prediction)

				elif model_choice == 'Naive Bayes':
					predictor = load_prediction_model("models/contraceptives_nv_model.pkl")
					prediction = predictor.predict(sample_data)
					st.write(prediction)

				elif model_choice == 'Random Forest':
					predictor = load_prediction_model("models/contraceptives_rf_model.pkl")
					prediction = predictor.predict(sample_data)
					st.write(prediction)


				final_result= get_key(prediction,prediciton_label)
				st.success(final_result)



































if __name__ == '__main__':
	main()
















