# Will_it_be_late-
An ML-powered web app that predicts whether your Zomato food delivery will arrive on time, before you even place the order.

"Will It Be Late?" is a full-stack machine learning project trained on 45,584 real Zomato India delivery records. The user types their address, the system auto-detects their city, and surfaces nearby restaurants — each tagged with an ML prediction of on-time or late delivery. Four models were built and compared: Logistic Regression, Linear Regression, Random Forest, and XGBoost. Random Forest achieved the best accuracy at 92.7% with an AUC of 0.974. The project includes a Flask REST API backend and a single-file HTML dashboard with four tabs: Order Food, Overview, Model Comparison, and ML Pipeline.


What problem does it solve?

Food delivery delays are frustrating and unpredictable. This project turns the question "will my food be late?" from a guess into a data-driven prediction — surfacing the exact factors causing the delay (driver handling multiple orders, traffic jams, bad weather, festival days) and how much each one contributes.


Key Technical Highlights


17 features used for prediction — 12 raw + 5 engineered (Haversine distance, pickup wait time, peak hour flag, food complexity score, weekend flag)
4 ML models trained, compared, and evaluated on Accuracy, F1, Precision, Recall, and AUC-ROC
Random Forest selected as best model (92.7% accuracy, 84.7% F1, AUC 0.974)
Flask REST API with 4 endpoints serving real-time predictions
Offline city detection — user's city matched from typed address using a keyword map, no internet or API key needed
Interactive dashboard — restaurant cards, late probability bars, click-to-expand detail sheets



Dataset


Source: Zomato India (Kaggle, public dataset)
Size: 45,584 delivery orders
Target: Binary — Late (delivery > 32 min) or On-Time
Late threshold: 32 minutes (75th percentile of all delivery times)
Class split: 75.3% On-Time · 24.7% Late



Top Insights from the Model


Multiple Deliveries is the single biggest predictor of lateness (26.5% importance)
Festival days add significant delay due to traffic + demand surge
Driver rating is a strong proxy for speed — higher-rated drivers are faster
Traffic jams alone push estimated delivery time up by 14+ minutes
Peak hours (lunch 12–2 PM, dinner 7–10 PM) correlate strongly with delays



Tech Stack


ML: scikit-learn (Random Forest, Logistic Regression, Linear Regression), XGBoost
Backend: Python 3, Flask, Flask-CORS
Frontend: HTML5, CSS3, Vanilla JavaScript (no frameworks)
Data: pandas, numpy
