# PanicSense - Render Deployment

## Deployment sa Render.com

### Step 1: Setup sa Render Dashboard

Sa Render dashboard, i-add ang new web service at i-point ito sa GitHub repository mo.

### Step 2: Configure ang deployment

#### OPTION 1: Gamit ang run.cjs (RECOMMENDED)

**Build Command:**
```
npm install
```

**Start Command:**
```
node run.cjs
```

#### OPTION 2: Gamit ang package.json scripts (Advanced setup, may issues)

**Build Command:**
```
npm install && npm install -g vite esbuild && npm run build
```

**Start Command:**
```
npm run start
```

⚠️ **PROBLEMA**: Marami itong issues sa Render dahil hindi laging available ang Vite at esbuild sa environment nila.

### Step 3: Environment Variables

I-add ang mga sumusunod na environment variables:

- `NODE_ENV` = `production`
- `DATABASE_URL` = [Your PostgreSQL URL]
- `GROQ_API_KEY` = [Your Groq API Key]

## Paano Gumagana Ang Deployment?

### Option 1: run.cjs (Simple at Reliable)
Simpleng file na nagko-connect sa main application:

- Gumagana kahit walang Vite installation
- CommonJS format para compatible kahit may "type": "module" sa package.json
- Direkta na tinatawag ang main index.js para patakbuhin ang application
- .cjs extension para masabi kay Node.js na iba ito sa ES modules

Advantages:
- Hindi kailangan ng Vite (na problema sa ibang hosting providers)
- Minimal build requirements
- Reliable, failsafe approach

### Option 2: package.json scripts (Advanced)
Kung kailangan ng full build gamit ang Vite at TypeScript:

- Requires pre-installation of Vite at esbuild
- May optimized build process for both client at server
- Better TypeScript integration
- Problema lang: Kailangan i-setup specifically sa Render

Lahat ng files ay makikita ni Render, walang hidden files o complex setup scripts!