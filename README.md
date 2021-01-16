# MIMIC-III
Resources scheduling is an important task of Healthcare Management Systems. The right scheduling of doctors and nurses aids on the improved resources utilization.  But the volume of newly admitted patients and unrelenting hospital operations make the task challenging.  The different types of patients tend to have different types of medical histories.  Hence, building a generalized model  to  read  records  of  all  the  patients  effectively  is  a  challenge.   This  project  discusses about  a  generalized  neural  network  model  to  predict  patients’  span  in  the  hospital  by using  doctors'  natural  language  writing  data  and  general  numeric  inputs.   The  model  is useful to represent most of the patients’ maximum information available via doctor’s writings.  Also, numeric data aids more information for majority of the patients.  The model is implemented for ICU patients’ database.  Different experiments are done in different hyper-parameters setups to find the better model.  Mean Squared Error, Mean Absolute Deviation and R squared score are used as the measures of performance to choose the most efficient model.

# Introduction
Healthcare Management System (HMS) is responsible for an efficient resources allocation plan on a daily basis in a hospital. Everyday many new patients admit into the hospital and the resulting continuous hospital operations make the resources planning both interesting and challenging.The information available on database management system increases with the increasing newly admitted patients and the regular hospital operation. Maximum insights can be drawn from the surfeit of information. One important information about the patients is their length of stay (LOS) in a hospital after they are admitted. This information is essential for HMS planning purpose. Knowing the LOS of a patient in advance can help HMS in following aspects:
- Helps to plan the hospital operations for patients in advance.
- Helps in the planning of doctors' and nurses' scheduling.
- Helps to manage the inventory and to keep anticipated tools furnished.
- Helps to plan and manage the beds.

In this paper, we propose a forecast model which uses neural networks to predict the patients' length of stay in a hospital. The modern trends of hospital database systems are known to be adopting electronic health records (EHR) to store the patients' data. The popularity of EHR has created a massive opportunity to explore the data and build the efficient prediction models. Deep Learning offers a variety of data modeling algorithms. Neural networks are amongst the popular algorithms to model healthcare data.

# Challenges in data
To build an efficient model, it is vital to consider the inputs which incorporate the maximum information needed to justify the output. EHR could fall short in that sense as the type of data it stores may not be sufficient to justify some patients' health conditions. For example, A pneumonia patient may have many X-rays to represent the health while a diabetic patient may have many lab reports. Converting the different types of data into a common type may result into the loss of important information. The following points summarize the challenges:
- The nature of information available for one type of patients can be different than that for another types.
- EHR data could fail to include all the important information in input features that can justify the output efficiently.
- Integrating together the data of different nature and importance is difficult.

# Opportunity of Natural Language
Building separate models to read separate data types is one way to address this issue. This will result into many different models for different patients. It will also create a new challenge of categorizing the patients before choosing the right model for them. Also, this approach results into the smaller training sets. Hence, we propose to build a more generalized model which can be applicable to ell types of patients. To build a generalized model representing all patients together, a common information is needed which incorporates the most of patients' medical conditions. Doctor's natural language is a common data for majority of the patients. In every visits to the patients, doctors update their record files. They often summarizes all the operations, lab reports, medications, and health condition in the report file. Following are the benefits of using doctor's natural language data.
- The information in doctors' writing reports can be extracted and converted into a digital format easily.
- Doctor's report can be represented in a common data type (STR/INT) for all patients.
- Doctor's reports contain important information in a summarized form.
- Doctor's report are sequential and hence the time-series information is structured.

# Limitations of Doctors' Reports & Solution
The doctor's writings sometimes possess the heavily aggregated information. If a data is an aggregate of many different features, then it may lose vital information upon aggregation. Hence, the natural language text data may fall short sometimes. To fill out the information that are lost in aggregation, some additional information sources may be needed. We propose to add some useful numeric feature inputs from EHR to the doctors' text feature input. The following are the main benefits of combing two data types together:
- EHR data together with doctors' natural language data incorporate the maximum information.
- EHR data helps to ascertain the information available in doctors' reports.
- The lost information in one data type is recovered by another.

The two different types of data with maximum information are used to feed the model. We propose a suitable neural network architecture to process the two types of data using the most effective algorithms.
