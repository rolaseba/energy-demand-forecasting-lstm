## ⚡ LSTM Energy Demand Prediction

**📌 Description**
Development of an LSTM-based deep learning model to forecast hourly electricity demand using national grid data from Bangladesh. The solution enables proactive grid management and resource optimization to ensure stability and reduce operational costs.

**💡 Business Value**
Predicts electricity demand with high accuracy to support grid stability, optimize generation across energy sources, and minimize wasteful generation—critical for managing ~25,700 MW of installed capacity in Bangladesh's national grid.

**📈 Key Results**

* Achieved **3.95% MAPE** and **320.21 MW MAE** in hourly demand forecasting
* Model captures weekly and diurnal patterns using 168-hour sequence input
* Supports 24-hour ahead forecasting for real-time operational decision-making

**🌟 Key Benefits**

* Enables proactive deployment of generation capacity before demand peaks
* Reduces operational costs through optimized resource allocation
* Provides early warning for supply-demand imbalances to prevent blackouts
* Scalable architecture for regional or national grid forecasting

**🧰 Tech Stack**
`Python`, `TensorFlow/Keras`, `pandas`, `scikit-learn`, `Streamlit`

**📊 Methods Used**
LSTM Neural Networks, Sequence Modeling, Time-Series Cross-Validation, MinMax Scaling, Early Stopping, Gradient Clipping

**🔗 Links**
[Repo](https://github.com/rolaseba/energy-demand-forecasting-lstm) | [Live App](https://energy-demand-forecasting-lstm.streamlit.app)

*Note: Built and validated on real-world hourly generation data from the Power Grid Company of Bangladesh (PGCB), with methodology transferable to other national or regional grid systems.*