# Fixing Render Deployment for PanicSense

Kung nagkakaroon ka ng "vite: not found" error sa Render deployment, sundin ang mga hakbang na ito para ma-fix ito.

## Step 1: I-clone ang repository sa Render

Gamitin ang GitHub repository URL mo: `https://github.com/draiimon/PanicSense`

## Step 2: I-configure ang Web Service

Gamitin ang mga settings na ito:

- **Name**: `panicsense` (o kahit anong pangalan na gusto mo)
- **Environment**: Node.js
- **Region**: Singapore (o mas malapit sa iyo)
- **Branch**: main (o anumang branch na gusto mong i-deploy)

## Step 3: Gamitin ang mga custom build commands

Sa halip na default settings, gamitin ang mga ito:

**Build Command:**
```
./render-build-bash.sh
```

**Start Command:**
```
node render-start.js
```

## Step 4: Siguruhin na nakatakda ang mga environment variables

Siguruhin na nakatakda ang mga ito:

```
NODE_ENV=production
PORT=10000
DATABASE_URL=your_database_connection_string
SESSION_SECRET=your_secure_session_secret
```

Palitan ang "your_database_connection_string" sa tunay na database URL ng PostgreSQL o Neon database mo.

## Step 5: Pag-click sa Deploy at Pagsubaybay ng Logs

Kapag na-click mo ang Deploy button, tingnan ang mga logs. Dapat makita mo na ang build ay gumagamit ng custom script natin na nag-iinstall ng devDependencies at gumagamit ng lokal na Vite.

## Kung May Errors Pa Rin

Kung may errors pa rin, subukan ang mga ito:

1. Tingnan kung nakita ng Render ang mga custom script files (`render-build-bash.sh` at `render-start.js`)
2. Baka kailangan mong bumalik sa Render dashboard at manu-manong i-restart ang deploy
3. Tingnan ang mga logs para sa anumang errors o warnings

## Bakit Ito Gumagana

Ang fix na ito ay gumagana dahil:

1. Tiyak na na-i-install natin ang `devDependencies` sa build
2. Ginagamit ang `npx` para patakbuhin ang lokal na Vite
3. Ginagamit natin ang custom start-up script (`render-start.js`) para maayos na i-handle ang environment variables

Kapag sinunod mo ang mga steps na ito, dapat ma-fix ang "vite: not found" error sa Render deployment mo.