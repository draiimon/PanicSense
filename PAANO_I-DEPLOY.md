# STEP-BY-STEP GUIDE: PAGDE-DEPLOY NG PANICSENSE SA RENDER

## Ano ang mga kailangan?

1. GitHub account na may PanicSense repository
2. Render.com account 
3. Neon.tech PostgreSQL database
4. Groq API key para sa AI functions

## Hakbang 1: Setup ng Render Account

1. Pumunta sa [render.com](https://render.com) at mag-sign up/log in
2. Click "New" at piliin ang "Web Service"

## Hakbang 2: I-connect ang GitHub Repository

1. Click "Connect account" at piliin ang GitHub
2. Hanapin ang PanicSense repository at i-connect ito

## Hakbang 3: Setup ng Web Service

Ilagay ang mga sumusunod:

- **Name**: PanicSense (o anumang pangalan na gusto mo)
- **Environment**: Node
- **Region**: Singapore (o pinakamalapit sa iyo)
- **Branch**: main

### NAPAKAIMPORTANTE NA SETTINGS:

- **Build Command**: `node render-setup.js`
- **Start Command**: `NODE_ENV=production node start-prod.js`
- **Plan**: Free (kalagitnaan ng page)

## Hakbang 4: Environment Variables

Click "Advanced" at idagdag ang mga sumusunod:

- `NODE_ENV` = production
- `DATABASE_URL` = (ang Neon PostgreSQL connection string mo)
- `GROQ_API_KEY` = (ang Groq API key mo)
- `SESSION_SECRET` = (kahit anong random string)

## Hakbang 5: Create Web Service

Click "Create Web Service"

## Hakbang 6: Monitor Deployment

- Tingnan ang logs para sa errors
- Kapag successful ang deployment, i-check ang web app sa <service-name>.onrender.com

## Hakbang 7: Pag-setup ng Database

Kung hindi mo pa nagagawa:

1. Pumunta sa [neon.tech](https://neon.tech)
2. Gumawa ng bagong PostgreSQL database
3. Kunin ang connection string at ilagay sa environment variables

## Hakbang 8: Custom Domain (Optional)

Kung gusto mo ng custom domain:

1. Click "Settings" sa Render dashboard
2. Scroll down sa "Custom Domains" section
3. Click "Add Custom Domain"
4. Sundin ang instructions

## Troubleshooting

Kung may errors:

1. Check logs sa Render dashboard
2. Tingnan kung tama ang lahat ng environment variables
3. Kung "vite not found" ang error, gagana pa rin ang fallback mechanism

## Mga Files Na Gumawa Tayo

Ito ang mga pinaka-importanteng files para sa Render deployment:

1. `render-setup.js` - Simpleng setup script
2. `start-prod.js` - Production startup script
3. `render.yaml` - Render configuration file

## Support

Kung may problema, i-check ang logs sa Render dashboard para makita kung ano ang error.

---

Congrats! Ikaw na ang may sariling PanicSense deployment!