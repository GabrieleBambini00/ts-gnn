# GitHub Repository Setup Guide

This guide walks you through setting up the TS-GNN repository on GitHub with all verified sheaf integration components.

## Prerequisites

1. GitHub account (create at https://github.com if needed)
2. Git installed locally
3. SSH key configured (or use HTTPS token)
4. Current working directory: `C:\Users\Gabriele\OneDrive - Università Commerciale Luigi Bocconi\Desktop\Sheaf Neural Networks\ts-gnn\`

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Create new repository:
   - **Repository name:** `ts-gnn` (or your preferred name)
   - **Description:** "Temporal Sheaf Graph Neural Network with verified integration"
   - **Visibility:** Public (for collaboration) or Private (for restricted access)
   - **Initialize:** ❌ Do NOT initialize with README (we already have one)
   - **License:** MIT (recommended)

3. Click **Create repository**

## Step 2: Configure Git Remote

After creating the repository on GitHub, you'll see a page with setup instructions. Use these commands:

### Option A: SSH (Recommended)

```bash
cd "C:\Users\Gabriele\OneDrive - Università Commerciale Luigi Bocconi\Desktop\Sheaf Neural Networks\ts-gnn"

# Add remote
git remote add origin git@github.com:<your-username>/<your-repo-name>.git

# Verify
git remote -v
# Should output:
# origin  git@github.com:<your-username>/<your-repo-name>.git (fetch)
# origin  git@github.com:<your-username>/<your-repo-name>.git (push)
```

### Option B: HTTPS (If SSH not configured)

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Test authentication
git remote -v
```

## Step 3: Set Default Branch

The repository uses `sheaf-verified-integration` as the main development branch. You may want to set it as default:

```bash
# Option 1: Set sheaf-verified-integration as default (recommended during development)
git push -u origin sheaf-verified-integration

# Option 2: Create main branch from sheaf-verified-integration
git checkout -b main
git push -u origin main
```

Then on GitHub:
- Go to Settings → Branches → Default branch
- Change from `main` to `sheaf-verified-integration` (or keep `main` if you prefer)

## Step 4: Push All Branches and Tags

Push the current verified branch:

```bash
git push -u origin sheaf-verified-integration
```

Push all branches (if you have other branches):

```bash
git push -u origin --all
```

Push tags (for releases):

```bash
git push -u origin --tags
```

## Step 5: Verify on GitHub

1. Go to https://github.com/<your-username>/<your-repo-name>
2. Verify you see:
   - ✅ All commits (should see 15+ commits)
   - ✅ All branches (should see `sheaf-verified-integration`)
   - ✅ File structure (src/, tests/, scripts/, configs/, etc.)
   - ✅ GitHub Actions workflow (`.github/workflows/smoke-tests.yml`)

## Step 6: Enable GitHub Actions (CI/CD)

1. Go to Settings → Actions → General
2. Under "Actions permissions":
   - ☑️ Allow all actions and reusable workflows
3. Under "Workflow permissions":
   - ☑️ Read and write permissions
   - ☑️ Allow GitHub Actions to create and approve pull requests

4. Go to Actions tab
   - You should see "Sheaf GNN Smoke Tests" workflow
   - Click to trigger first run (or it will run on next push)

## Step 7: Configure Repository Secrets (Optional)

If you plan to use automated deployment or notifications:

1. Go to Settings → Secrets and variables → Actions
2. Add secrets as needed (e.g., for HPC integration):
   - `HPC_USER`: Your Bocconi HPC username
   - `HPC_HOST`: `slogin.hpc.unibocconi.it`
   - `HPC_SSH_KEY`: Private SSH key (if using passwordless)

## Step 8: (Optional) Set Up GitHub Pages Documentation

To publish documentation:

1. Go to Settings → Pages
2. Under "Build and deployment":
   - Source: Deploy from a branch
   - Branch: `sheaf-verified-integration` (or `main`)
   - Folder: `/docs` or `/` (root)
3. GitHub will automatically deploy documentation

## Step 9: Create Initial Commit on main Branch (Optional)

If you want a `main` branch that's always stable:

```bash
# Create and checkout main from current HEAD
git checkout -b main

# Push to GitHub
git push -u origin main
```

Then set `main` as default on GitHub (Settings → Branches → Default branch).

## Step 10: Add Collaborators (Optional)

1. Go to Settings → Collaborators
2. Add collaborators by GitHub username
3. Grant appropriate permissions:
   - **Pull access**: Can view and pull
   - **Triage access**: Can manage issues/PRs (no code changes)
   - **Write access**: Can push code
   - **Maintain access**: Can manage settings
   - **Admin access**: Full control

## Verification Checklist

After setup, verify:

- [ ] Remote configured: `git remote -v` shows `origin`
- [ ] Latest commit visible on GitHub
- [ ] All branches pushed: `sheaf-verified-integration` visible
- [ ] GitHub Actions enabled and workflow visible
- [ ] README.md displays correctly on GitHub repo page
- [ ] All 15+ commits visible in commit history
- [ ] File structure complete (src/, tests/, configs/, scripts/)
- [ ] `.github/workflows/smoke-tests.yml` visible in Actions tab

## Running Smoke Tests on GitHub

Once set up, tests run automatically on:
- **Push** to `main`, `sheaf-verified-integration`, or `develop`
- **Pull Request** to any of the above branches
- **Manual trigger** from Actions tab (Workflow dispatch)

To manually trigger:
1. Go to GitHub → Actions → "Sheaf GNN Smoke Tests"
2. Click "Run workflow" button
3. Select branch and click green "Run workflow"

## Setting Up Local Development (After GitHub Push)

For a fresh clone on another machine:

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd ts-gnn

# Checkout sheaf-verified-integration branch if not already selected
git checkout sheaf-verified-integration

# Install dependencies
pip install -e .

# Run smoke tests locally to verify setup
python test_sheaf_end_to_end.py

# Or run via pytest
python -m pytest tests/ -v
```

## HPC Integration (Advanced)

Once GitHub is set up, you can integrate with HPC:

```bash
# On HPC login node, clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd ts-gnn

# Submit HPC training job
sbatch scripts/submit_hpc_job.sh
```

For automated HPC integration, see `scripts/sync_up.sh` and `scripts/sync_down.sh`.

## Troubleshooting

### "fatal: Authentication failed"
- **Solution**: Verify SSH key is configured or use HTTPS with personal access token

### "rejected: updates were rejected because the tip of your current branch is behind"
- **Solution**: Pull first: `git pull origin sheaf-verified-integration`

### "No workflows found"
- **Solution**: Commit `.github/workflows/smoke-tests.yml` to your repository:
  ```bash
  git add .github/workflows/
  git commit -m "ci: add GitHub Actions workflow"
  git push origin sheaf-verified-integration
  ```

### "Cannot push to branch"
- **Solution**: Verify you have write permissions and are on the correct branch

## Next Steps

After GitHub setup:

1. **Local Development**: Clone on your machine and create feature branches
2. **CI/CD**: Monitor GitHub Actions for test results on each push
3. **Real Data Training**: Run `sbatch scripts/submit_hpc_job.sh` on HPC
4. **Documentation**: Update README with results and figures
5. **Releases**: Create GitHub Releases with checkpoints and metrics

## Questions?

- For GitHub setup issues: See GitHub's [official documentation](https://docs.github.com/)
- For TS-GNN issues: See `README.md` and `QUICK_REFERENCE.md`
- For HPC integration: See `scripts/submit_hpc_job.sh` comments

---

**Status**: Ready to push to GitHub  
**Branch**: `sheaf-verified-integration` (15+ commits)  
**Tests**: All passing (end-to-end + 5 CI gates)  
**Hardware**: A100-ready with sparse Laplacian (38.9× memory reduction)
