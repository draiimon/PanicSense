# How to Run PanicSense Locally

This guide will help you set up and run PanicSense on your local machine, whether you're using Docker or running it directly.

## Option 1: Using Docker (Recommended for Consistency)

### Prerequisites
- Docker installed on your machine
- Docker Compose (optional, but recommended)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/draiimon/PanicSense.git
   cd PanicSense
   ```

2. **Build and run using Docker**
   ```bash
   # Build the Docker image
   docker build -t panicsense:latest -f Dockerfile.working .

   # Run the container
   docker run -p 5000:5000 --env-file .env panicsense:latest
   ```

3. **Using Docker Compose (Alternative)**
   ```bash
   # Make sure your docker-compose.yml is updated
   docker-compose up --build
   ```

4. **Access the application**
   Open your browser and go to: `http://localhost:5000`

## Option 2: Direct Installation (Without Docker)

### Prerequisites
- Node.js 20.x
- Python 3.11 or later
- pnpm installed globally (`npm install -g pnpm`)
- PostgreSQL database

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/draiimon/PanicSense.git
   cd PanicSense
   ```

2. **Set up environment variables**
   Copy `.env.example` to `.env` and update the variables:
   ```bash
   cp .env.example .env
   # Edit .env with your database connection and other settings
   ```

3. **Install Node dependencies**
   ```bash
   pnpm install
   ```

4. **Set up Python environment**
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install Python dependencies
   pip install -r server/python/requirements.txt
   
   # Download NLTK data
   python -c "import nltk; nltk.download('punkt')"
   ```

5. **Set up the database**
   ```bash
   # Create the database tables
   npm run db:push
   ```

6. **Start the development server**
   ```bash
   npm run dev
   ```

7. **Access the application**
   Open your browser and go to: `http://localhost:5000`

## Common Issues and Solutions

### Python Package Installation Issues

If you encounter issues with PyTorch or scikit-learn installation:

- For PyTorch: Consider installing the CPU-only version with:
  ```bash
  pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- For scikit-learn: Make sure you have the necessary C compiler and development libraries:
  - On Ubuntu/Debian:
    ```bash
    sudo apt-get install build-essential python3-dev libopenblas-dev
    ```
  - On macOS:
    ```bash
    brew install openblas
    ```
  - On Windows:
    Install Visual C++ Build Tools from the Microsoft Visual Studio Installer

### Database Connection Issues

- Make sure your PostgreSQL server is running
- Verify the DATABASE_URL in your .env file is correct
- For local development, a typical URL format is:
  ```
  DATABASE_URL=postgresql://username:password@localhost:5432/database_name
  ```

### Node.js Memory Issues

If you encounter memory issues when building:

```bash
export NODE_OPTIONS="--max-old-space-size=4096"
# or on Windows
set NODE_OPTIONS=--max-old-space-size=4096
```

## Additional Information

- The application uses port 5000 by default. You can change this in the .env file.
- For development, the API and frontend are both served from the same port.
- For any other issues, please check the GitHub repository issues section or submit a new issue.