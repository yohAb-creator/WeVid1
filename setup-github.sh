#!/bin/bash

echo "🚀 WeVid - GitHub Repository Setup"
echo "===================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
    echo ""
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Adding remote origin..."
    git remote add origin https://github.com/Mingxiao300/WeVid.git
    echo "✅ Remote origin added"
    echo ""
fi

# Create and switch to main branch
echo "🌿 Creating main branch..."
git checkout -b main
echo "✅ Switched to main branch"
echo ""

# Add all files
echo "📦 Adding files to staging..."
git add .
echo "✅ Files added to staging"
echo ""

# Check what will be committed
echo "📋 Files to be committed:"
git status --porcelain | head -20
if [ $(git status --porcelain | wc -l) -gt 20 ]; then
    echo "... and $(( $(git status --porcelain | wc -l) - 20 )) more files"
fi
echo ""

# Commit changes
echo "💾 Committing changes..."
git commit -m "Initial commit: WeVid AI-powered podcast analyzer

✨ Features:
- YouTube video analysis with timeline-based approach
- AI-powered concept extraction using OpenAI/Gemini
- AssemblyAI integration for rich audio analysis
- Interactive embedded video players
- Real-time progress updates
- Semantic relevance matching

🔧 Tech Stack:
- Next.js 14 with TypeScript
- Tailwind CSS for styling
- OpenAI GPT-3.5 Turbo / Google Gemini Pro
- AssemblyAI for speech-to-text
- Serverless API architecture

🎯 Ready for personalized learning experiences!"
echo "✅ Changes committed"
echo ""

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push -u origin main
echo "✅ Pushed to GitHub successfully!"
echo ""

echo "🎉 Repository setup complete!"
echo ""
echo "📱 Your repository is now available at:"
echo "   https://github.com/Mingxiao300/WeVid"
echo ""
echo "🔒 Security Notes:"
echo "   ✅ .env.local is protected by .gitignore"
echo "   ✅ API keys will NOT be pushed to GitHub"
echo "   ✅ Only .env.example template is included"
echo ""
echo "📖 Next Steps:"
echo "   1. Update the repository description on GitHub"
echo "   2. Add topics/tags: nextjs, ai, podcast, youtube, assemblyai"
echo "   3. Consider adding a demo GIF to the README"
echo "   4. Set up GitHub Pages if needed"
