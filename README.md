## Quick venv activation alias

To quickly activate your virtual environment, add this alias to your shell profile (e.g., `~/.zshrc` or `~/.bashrc`):

```sh
alias sdoc='source $(pwd)/venv/bin/activate'
```

Now, just run `sdoc` in your project directory to activate the venv.
# AI Doctor - Skin Lesion Analysis Platform

A comprehensive AI-powered healthcare platform for skin lesion analysis and patient-doctor communication, built with FastAPI, React, and advanced machine learning models.

## 🚀 Features

### 🔬 AI-Powered Skin Analysis
- **Deep Learning Model**: ResNet18-based skin lesion classification
- **Real-time Analysis**: Instant malignancy risk assessment
- **High Accuracy**: Trained on melanoma cancer datasets
- **Risk Scoring**: Detailed probability scores for different lesion types

### 💬 Advanced Messaging System
- **Real-time Chat**: WebSocket-based instant messaging between patients and doctors
- **File Attachments**: Share medical images and documents
- **Message Reactions**: Interactive communication features
- **Read Receipts**: Message delivery and read status tracking

### 🤖 AI Assistant Integration
- **Gemini AI**: Primary LLM provider for intelligent responses
- **Multi-provider Support**: Fallback to Azure OpenAI and Ollama
- **Context-aware**: Medical knowledge-based responses
- **Conversational UI**: Natural language interaction

### 👥 User Management
- **Role-based Access**: Patients, Doctors, and Administrators
- **Secure Authentication**: JWT-based authentication system
- **Doctor Application System**: Admin approval workflow for doctors
- **Patient Profiles**: Comprehensive medical history tracking

### 📊 Comprehensive Dashboard
- **Patient Dashboard**: Lesion history, appointments, medical records
- **Doctor Dashboard**: Patient management, analysis reviews, messaging
- **Admin Panel**: User management, system oversight, analytics

## 🛠 Technology Stack

### Backend
- **FastAPI**: Modern, fast Python web framework
- **SQLAlchemy**: SQL toolkit and ORM
- **MySQL**: Primary database
- **PyTorch**: Deep learning framework for AI models
- **WebSockets**: Real-time communication
- **Pydantic**: Data validation and serialization

### Frontend
- **React 18**: Modern frontend library
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client for API communication

### AI/ML
- **PyTorch**: Deep learning models
- **ResNet18**: Convolutional neural network architecture
- **Gemini AI**: Google's large language model
- **Azure OpenAI**: Backup LLM provider
- **Ollama**: Local LLM support

### Infrastructure
- **Azure**: Cloud deployment ready
- **Docker**: Containerization support
- **Bicep**: Infrastructure as Code
- **WebSocket**: Real-time features

## 📋 Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **MySQL 8.0+**
- **Git**

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-doctor-skin-lesion.git
cd ai-doctor-skin-lesion
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv ai_doctor_env
source ai_doctor_env/bin/activate  # On Windows: ai_doctor_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup
```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE ai_doc;
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configurations:
# - MySQL credentials
# - Gemini API key
# - Other service configurations
```

### 5. Frontend Setup
```bash
cd frontend-react
npm install
```

### 6. Database Migration
```bash
# Run from project root
python -c "from backend.models import Base; from backend.database import engine; Base.metadata.create_all(bind=engine)"
```

## 🚀 Running the Application

### Start Backend Server
```bash
source ai_doctor_env/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend Development Server
```bash
cd frontend-react
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 👤 Default Admin Account
```
Email: admin@example.com
Password: Admin@12345
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```bash
# Database Configuration
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DB=ai_doc
MYSQL_USER=root
MYSQL_PASSWORD=your_password

# AI Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
LLM_PROVIDER=gemini

# Optional: Azure OpenAI (fallback)
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key

# Optional: Ollama (local LLM)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
```

## 📱 API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/me` - Get current user info
- `POST /auth/logout_all` - Logout from all devices

### Patients
- `POST /patients/register` - Patient registration
- `GET /patients/` - List patients (doctors/admin only)
- `GET /patients/{id}/profile` - Get patient profile

### Doctors
- `POST /doctors/apply` - Doctor application
- `GET /doctors/` - List doctors
- `GET /doctors/{id}/profile` - Get doctor profile

### Lesions
- `POST /lesions/` - Upload and analyze lesion
- `GET /lesions/` - Get lesion history
- `GET /lesions/{id}` - Get specific lesion details

### Messaging
- `GET /chat/rooms` - Get chat rooms
- `POST /chat/rooms` - Create new chat room
- `GET /chat/rooms/{id}/messages` - Get messages
- `POST /chat/rooms/{id}/messages` - Send message
- `WebSocket /chat/ws/{room_id}` - Real-time messaging

## 🔐 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Role-based Access Control**: Fine-grained permissions
- **Input Validation**: Comprehensive data validation
- **SQL Injection Protection**: Parameterized queries
- **CORS Configuration**: Secure cross-origin requests
- **Password Hashing**: Bcrypt password security

## 🧪 Testing

### Run Backend Tests
```bash
pytest backend/tests/
```

### Run Frontend Tests
```bash
cd frontend-react
npm test
```

## 🚀 Deployment

### Azure Deployment
```bash
# Using Azure Developer CLI
azd init
azd up
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d
```

## 📊 Project Structure

```
ai-doctor-skin-lesion/
├── backend/                 # FastAPI backend
│   ├── routes/             # API endpoints
│   ├── models.py           # Database models
│   ├── database.py         # Database configuration
│   ├── ml/                 # Machine learning models
│   └── main.py             # Application entry point
├── frontend-react/         # React frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   └── services/       # API services
├── infra/                  # Infrastructure as Code
├── scripts/                # Utility scripts
└── requirements.txt        # Python dependencies
```

## 📚 Training Datasets

The machine learning models were trained on the following medical datasets:
- **HAM10000**: Harvard Dataverse skin lesion dataset
- **DermMel**: Dermatology melanoma detection dataset
- **Med Node**: Medical node classification dataset
- **SD260**: Skin disease 260 image dataset
- **Skin Cancer ISIC**: International Skin Imaging Collaboration dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, email your-email@example.com or create an issue in this repository.

## 🙏 Acknowledgments

- Medical dataset providers
- Open source AI/ML community
- Healthcare professionals who provided domain expertise
- Contributors and testers

---

**⚠️ Medical Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Download the pretrained models:
### Attention U-Net
```python
from huggingface_hub import snapshot_download

# Define the model repo
model_name = "Sharukesh/attention-unet"

# Download the model locally
snapshot_download(repo_id=model_name, local_dir="/content/attention-unet")
```

### GAN
```python
from huggingface_hub import snapshot_download

# Define the model repo
model_name = "Sharukesh/GAN-HAM10000-class-balancing"

# Download the model locally
snapshot_download(repo_id=model_name, local_dir="/content/GAN")
```

### SMOTE
On our implementation of GAN the outputs were not well featurized, so those images could not be used in the training of the model, hence we choose to do SMOTE (Synthetic Minority Oversampling Technique).

#### How SMOTE Works:

1. **Identify Minority Class:** It targets the minority class in an imbalanced dataset.
2. **Select a Sample:** Randomly picks a sample from the minority class.
3. **Find Nearest Neighbors:** Identifies its k-nearest neighbors in the feature space (typically using Euclidean distance).
4. **Generate Synthetic Samples:** Creates new synthetic data points by interpolating between the original sample and one of its nearest neighbors.
5. **Repeat:** This process is repeated until the desired class balance is achieved.

Find our implementation of smote down [here](https://github.com/gshyamv/Skin-Lesion-Classification/tree/main/SMOTE)
