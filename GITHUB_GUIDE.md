# GitHub Integration Guide

This guide provides step-by-step instructions for setting up and pushing this project to GitHub.

## Prerequisites

- A GitHub account
- Git installed on your local machine
- Basic knowledge of Git commands

## Setup Steps

### 1. Create a New Repository on GitHub

1. Log in to your GitHub account
2. Click the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "disaster-sentiment-analysis")
4. Optionally add a description
5. Choose whether the repository should be public or private
6. Do NOT initialize with README, .gitignore, or license (as we already have these files)
7. Click "Create repository"

### 2. Initialize Git in Your Local Project (if not already done)

```bash
# Navigate to your project directory
cd path/to/disaster-sentiment-analysis

# Initialize a git repository
git init
```

### 3. Add GitHub as a Remote Repository

```bash
# Add the GitHub repository as a remote
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git

# Verify the remote was added
git remote -v
```

### 4. Stage and Commit Files

```bash
# Add all files to staging
git add .

# Commit the changes
git commit -m "Initial commit: Disaster Sentiment Analysis Platform"
```

### 5. Push to GitHub

```bash
# Push to the main branch on GitHub
git push -u origin main
```

If your default branch is called "master" instead of "main", use:

```bash
git push -u origin master
```

### 6. Verify on GitHub

1. Go to your GitHub repository page
2. Refresh the page
3. You should see all your project files now on GitHub

## Using the Automated Setup Script

For convenience, you can use the included `setup.sh` script to automate these steps:

```bash
# Make the script executable (if not already)
chmod +x setup.sh

# Run the setup script
./setup.sh
```

Follow the prompts in the script to set up your GitHub repository and optionally build and run Docker containers.

## Workflow for Future Changes

After the initial setup, use this workflow for future changes:

1. **Pull latest changes:**
   ```bash
   git pull origin main
   ```

2. **Make your changes** to the project files

3. **Stage, commit, and push:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin main
   ```

## Working with Branches (Optional)

For larger features or collaborative work, consider using branches:

1. **Create a new branch:**
   ```bash
   git checkout -b feature-name
   ```

2. **Make your changes** on this branch

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Implement feature X"
   ```

4. **Push the branch to GitHub:**
   ```bash
   git push origin feature-name
   ```

5. **Create a Pull Request** on GitHub to merge your branch into main

## Troubleshooting

- **Authentication Issues**: If you're having trouble with authentication, consider setting up SSH keys or using a personal access token
- **Conflicts**: If you encounter merge conflicts, resolve them in your code editor and then complete the merge
- **Large Files**: GitHub has a file size limit of 100MB. For larger files, consider using Git LFS or keeping them out of the repository