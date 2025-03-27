# Social Media Sharing Setup for PANICSENSE PH

## What was added

1. **Page Title**: Added "PANICSENSE PH" as the title that appears in browser tabs
2. **Favicon**: Added a favicon for browser tabs at `/client/public/images/favicon.png`
3. **OpenGraph Images**: Added preview images for social media at `/client/public/images/og-image.png`
4. **Meta Tags**: Added proper OpenGraph and Twitter card meta tags for social media sharing

## How it works

When someone shares your site link on platforms like:
- Facebook
- Messenger
- Twitter
- LinkedIn
- Discord

The platform will fetch the meta tags from your HTML and display:
- The title "PANICSENSE PH"
- The description about disaster sentiment analysis
- The preview image you provided

## GitHub Integration

The image URLs in the HTML now reference:
```
https://raw.githubusercontent.com/draiimon/PanicSense/main/client/public/images/og-image.png
```

This means:
1. When you push to GitHub, make sure the images are included in your repository
2. Social media platforms will fetch the images directly from GitHub (not your deployment)

## Important when pushing to GitHub

Make sure these files are pushed:
- `/client/public/images/favicon.png`
- `/client/public/images/og-image.png`
- `/client/public/images/logo.png` (optional)
- `/client/index.html` (with the updated meta tags)

## Testing social media cards

You can test your social media preview cards using:
- [Facebook Sharing Debugger](https://developers.facebook.com/tools/debug/)
- [Twitter Card Validator](https://cards-dev.twitter.com/validator)
- [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)

Simply enter your deployed URL and these tools will show you how your links will appear when shared.