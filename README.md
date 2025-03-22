# Weaver

## Heart Failure Risk Prediction using Synthetic EHR Data
A machine learning pipeline that predicts heart failure risk using synthetic electronic health record (EHR) data from Synthea. The model is deployed via FastAPI and packaged in Docker for easy deployment.

📌 Project Features
Predicts risk of heart failure (HFYN) from patient data.

Uses Naive Bayes (GaussianNB) with custom priors for balanced classification.

Encodes categorical features using LabelEncoder + OneHotEncoder.

Evaluated with ROC-AUC, classification reports, and threshold tuning.

Deployed with FastAPI + Docker — includes interactive API docs via Swagger.

🗂️ Project Structure
bash
Copy
Edit
Weaver/
├── app/                      # FastAPI app
│   └── main.py
├── data/                     # Data loading & processing
│   ├── DataLoad.py
│   └── DataProcessing.py
├── model/                    # Saved model + encoders
│   ├── model_NB.pkl
│   ├── onehot_encoder.pkl
│   └── label_encoders.pkl
├── synthea/                  # Synthetic data
├── Dockerfile
├── requirements.txt
└── README.md
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/wellDeadpan/Weaver.git
cd Weaver
2. Build Docker Image
bash
Copy
Edit
docker build -t weaver-app .
3. Run Docker Container
bash
Copy
Edit
docker run -p 8000:8000 weaver-app
4. Access API Docs
Open http://localhost:8000/docs for Swagger UI.

Test the /predict endpoint interactively.

📊 Example API Request
Endpoint: POST /predict
Request Body:

json
Copy
Edit
{
  "features": [1.0, 0.5, 2.3, 4.4, ..., 29.1]  # Total 29 features expected
}
Response:

json
Copy
Edit
{
  "prediction": [1]
}
📈 Model Training Summary
Model: GaussianNB(priors=[0.5, 0.5])

Features: Encoded patient demographic + medical data

Target: HFYN (Heart Failure Yes/No)

Evaluation:

ROC-AUC: 

Threshold tuned at 0.3 for better recall

🧠 Future Improvements
Automate data preprocessing in FastAPI

Add logging and error handling

Explore deployment to cloud (e.g., AWS, Render)

Add CI/CD (e.g., GitHub Actions)

📄 License
MIT License

🙌 Acknowledgements
Synthetic data from Synthea

Built with love, FastAPI, and Docker 🐳

💬 Want Help With...
Writing unit tests?

Deploying to cloud?

Improving model performance?

Just ask — I’m happy to collaborate!

