# Instructions for Pushing to GitHub

The social media preview image has been updated to use your exact "PANICSENSE PH.png" file for all platforms. Here's what to push to GitHub:

## Files to Push

1. `client/index.html` - Contains the updated meta tags using your GitHub image URL
2. `client/public/favicon.png` - The rounded purple brain circuit icon for browser tabs
3. `client/public/images/panicsense-icon.png` - The same icon used in the images folder
4. `client/public/images/PANICSENSE PH.png` - Your social media preview image (already on GitHub)

## GitHub Push Commands

```bash
# Navigate to the repository directory first
cd /path/to/your/repository

# Add the files
git add client/index.html
git add client/public/favicon.png
git add client/public/images/panicsense-icon.png

# Commit the changes
git commit -m "Update social media meta tags to use PANICSENSE PH.png and add rounded purple brain icon"

# Push to GitHub
git push origin main
```

After pushing, your links shared on social media (Messenger, Twitter, Instagram, etc.) will show your exact PANICSENSE PH.png image as the preview!
