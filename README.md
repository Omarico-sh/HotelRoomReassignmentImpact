# Hotel Room Reassignment Impact

This study investigates the effect of assigning a different room type for the customer than initially reserved on the likelihood of booking cancellations in the hospitality industry. Utilizing the **Hotel Booking Demand** dataset, which includes over 120,000 observations from Lisbon and Algarve, the research addresses the significant challenge posed by hidden confounders and the complexities introduced by dependencies among measured variables, which can lead to biased estimates. By leveraging structural causal models, we determine whether and how the causal effect of interest can be estimated from the observed data based on the assumptions encoded in the causal model and graph. By identifying criteria to calculate the causal effect, we adopt various methods and machine learning models to compare the results, providing confidence levels and validation metrics for each. Our findings aim to provide valuable insights into managing booking cancellations, contributing to better decision-making in hotel operations by quantifying the impact of different room type assignment on customer booking cancellation.

## Files

- **HotelRoomReassignmentImpact.pdf**: Research paper for the study's methodology, analysis, and findings on the causal impact of room type reassignment on booking cancellations. This paper provides detailed insights and interpretations from the experiments conducted with various validations using advanced statistical methods.

- **Hotel_Demand_Experiment.ipynb**: Jupyter notebook containing the complete experimental process for assessing the impact of room type reassignment on booking cancellations. It includes data preprocessing, causal graph construction, and model evaluation.

- **Run.py**: Python script designed to execute all the experimental procedures in a streamlined format, suitable for reproducibility and integration.

- **hotel_bookings.csv**: Dataset containing hotel booking data used for our study, including information on room types, booking status, and customer demographics.
  
- **dataset_ready.csv**: Hotel bookings' dataset after applying EDA and ready for analysis.
